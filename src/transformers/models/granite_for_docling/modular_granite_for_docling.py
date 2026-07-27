# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from transformers.models.idefics3, granitemoehybrid, and nanoVLM connector patterns
"""PyTorch GraniteForDocling model."""

from functools import lru_cache
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import BatchEncoding, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING
from ..granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridAttention,
    GraniteMoeHybridMLP,
    GraniteMoeHybridRotaryEmbedding,
)
from ..granitemoeshared.modeling_granitemoeshared import GraniteMoeSharedRMSNorm
from ..idefics3.configuration_idefics3 import Idefics3VisionConfig
from ..idefics3.modeling_idefics3 import (
    Idefics3BaseModelOutputWithPast,
    Idefics3CausalLMOutputWithPast,
    Idefics3Connector,
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3PreTrainedModel,
)
from ..idefics3.processing_idefics3 import Idefics3Processor

# The doclang tokenizer's tile-position vocab (<row_r_col_c>) covers r, c in 1..16 only.
# GotOcr2's get_all_supported_aspect_ratios has no per-dimension cap, so an extreme-aspect-ratio
# crop (e.g. a 322x19 equation line, with our max_patches=32) can pick a grid like (1, 24): the
# marker token for a column/row past 16 doesn't exist in the vocab, so _expand_image_tokens_in_ids
# silently drops that tile's image-token block instead of raising, producing an off-by-one-tile
# mismatch between the text and image embeddings. Capped local copies below fix the row/col count
# used for the prompt; _install_tile_grid_dim_cap keeps the shared GotOcr2 image processor's own
# pixel tiling (a separate code path) in agreement with it.
MAX_TILE_GRID_DIM = 16


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(min_image_tiles: int, max_image_tiles: int) -> list[tuple[int, int]]:
    """Same as GotOcr2's get_all_supported_aspect_ratios, restricted to grids that fit MAX_TILE_GRID_DIM."""
    max_dim = min(max_image_tiles, MAX_TILE_GRID_DIM)
    aspect_ratios = []
    for width in range(1, max_dim + 1):
        for height in range(1, max_dim + 1):
            if max_image_tiles >= width * height >= min_image_tiles:
                aspect_ratios.append((width, height))
    return sorted(aspect_ratios, key=lambda x: x[0] * x[1])


@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]:
    """Same selection logic as GotOcr2's get_optimal_tiled_canvas, over the capped candidates above."""
    possible_tile_arrangements = get_all_supported_aspect_ratios(min_image_tiles, max_image_tiles)

    original_height, original_width = original_image_size
    target_tile_height, target_tile_width = target_tile_size
    aspect_ratio = original_width / original_height
    area = original_width * original_height

    best_ratio_diff = float("inf")
    best_grid = (1, 1)
    for grid in possible_tile_arrangements:
        grid_aspect_ratio = grid[0] / grid[1]
        ratio_diff = abs(aspect_ratio - grid_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_grid = grid
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * target_tile_height * target_tile_width * grid[0] * grid[1]:
                best_grid = grid

    return best_grid


def _install_tile_grid_dim_cap() -> None:
    """Apply the same cap to the shared GotOcr2 image processor's own tiling.

    ``GraniteForDoclingProcessor.image_processor`` is a stock GotOcr2ImageProcessor(Fast)
    instance; its ``crop_image_to_patches`` calls ``get_optimal_tiled_canvas`` /
    ``get_all_supported_aspect_ratios`` from ``got_ocr2.image_processing_got_ocr2`` directly, a
    separate function object from the capped copies above. Without this, the actual pixel tiling
    and the prompt's row/col tokens (computed with the capped versions above) could disagree.
    """
    from ..got_ocr2 import image_processing_got_ocr2 as _got_ocr2_slow

    orig = _got_ocr2_slow.get_all_supported_aspect_ratios
    if getattr(orig, "_tile_grid_dim_capped", False):
        return

    @lru_cache(maxsize=32)
    def _capped(min_image_tiles: int, max_image_tiles: int):
        return [
            grid
            for grid in orig(min_image_tiles, max_image_tiles)
            if grid[0] <= MAX_TILE_GRID_DIM and grid[1] <= MAX_TILE_GRID_DIM
        ]

    _capped._tile_grid_dim_capped = True
    _got_ocr2_slow.get_all_supported_aspect_ratios = _capped
    _got_ocr2_slow.get_optimal_tiled_canvas.cache_clear()


def _prompt_split_image(image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token):
    """Prompt with expanded image tokens for when the image is split into patches.

    Only includes the global thumbnail token when there are multiple tiles (image_rows * image_cols > 1),
    matching the training-time behavior. GotOcr2ImageProcessor likewise only appends the
    thumbnail patch when the image is split into more than one tile.
    """
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}" + f"<row_{n_h + 1}_col_{n_w + 1}>" + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    if image_rows * image_cols > 1:
        text_split_images += (
            f"\n{fake_token_around_image}"
            + f"{global_img_token}"
            + f"{image_token}" * image_seq_len
            + f"{fake_token_around_image}"
        )
    return text_split_images


def _prompt_single_image(image_seq_len, fake_token_around_image, image_token, global_img_token):
    """Prompt with expanded image tokens for a single image (no tiles)."""
    return (
        f"{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows, image_cols, image_seq_len, fake_token_around_image, image_token, global_img_token
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_img_token=global_img_token,
        )
    return _prompt_split_image(
        image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token
    )


def _compact_prompt_split_image(image_rows, image_cols, fake_token_around_image, global_img_token):
    """Compact prompt without repeated <image> tokens; expansion happens post-tokenization."""
    text = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text += f"{fake_token_around_image}<row_{n_h + 1}_col_{n_w + 1}>"
        text += "\n"
    if image_rows * image_cols > 1:
        text += f"\n{fake_token_around_image}{global_img_token}{fake_token_around_image}"
    return text


def _compact_prompt_single_image(fake_token_around_image, global_img_token):
    """Compact prompt for a single image without repeated <image> tokens."""
    return f"{fake_token_around_image}{global_img_token}{fake_token_around_image}"


def get_compact_image_prompt_string(image_rows, image_cols, fake_token_around_image, global_img_token):
    """Returns a compact prompt string without repeated <image> tokens."""
    if image_rows == 0 and image_cols == 0:
        return _compact_prompt_single_image(fake_token_around_image, global_img_token)
    return _compact_prompt_split_image(image_rows, image_cols, fake_token_around_image, global_img_token)


if TYPE_CHECKING:
    from ...modeling_rope_utils import RopeParameters
    from ...tokenization_utils_base import PreTokenizedInput


logger = logging.get_logger(__name__)


## Configuration ##


class GraniteForDoclingVisionConfig(Idefics3VisionConfig):
    model_type = "granite_for_docling_vision"


@auto_docstring
class GraniteForDoclingTextConfig(PretrainedConfig):
    r"""
    Configuration for the dense Granite-style text decoder used in [`GraniteForDoclingModel`].

    shared_intermediate_size (`int`, *optional*, defaults to 1024):
        Intermediate size of the shared MLP (`shared_mlp`) in each decoder layer.
    position_embedding_type (`str`, *optional*, defaults to `"rope"`):
        Positional embedding type. Supported: `"rope"`.
    """

    model_type = "granite_for_docling_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: "RopeParameters | dict | None" = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    embedding_multiplier: float | int | None = 1.0
    logits_scaling: float | int | None = 1.0
    residual_multiplier: float | int | None = 1.0
    attention_multiplier: float | int | None = 1.0
    shared_intermediate_size: int = 1024
    position_embedding_type: str = "rope"

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


class GraniteForDoclingConfig(PretrainedConfig):
    r"""
    Configuration for [`GraniteForDoclingModel`]: a SigLIP vision tower, modality projector,
    and dense Granite-style text decoder with optional DeepStack visual injections.

    Args:
        image_token_id (`int`, *optional*, defaults to 128257):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`GraniteForDoclingVisionConfig` or `dict`, *optional*):
            Custom vision config or dict for the vision tower.
        text_config (`GraniteForDoclingTextConfig` or `dict`, *optional*):
            Custom text config or dict for the text model.
        scale_factor (`int`, *optional*, defaults to 4):
            The scale factor for the image encoder projector.
        mp_pooling_mode (`str`, *optional*, defaults to `"pixel_shuffle_mlp_v2"`):
            Modality-projector variant. Supported: `"pixel_shuffle"`, `"pixel_shuffle_mlp_v2"`.
        use_deepstack (`bool`, *optional*, defaults to `True`):
            Whether to inject intermediate ViT features into selected decoder layers at image-token
            positions.
        deepstack_visual_indexes (`list[int]`, *optional*, defaults to `[3, 7, 10]`):
            Vision encoder block indices whose outputs feed DeepStack. Slot `i` pairs with
            `deepstack_attn_layers[i]`.
        deepstack_attn_layers (`list[int]`, *optional*, defaults to `[0, 1, 2]`):
            Text decoder layer indices that receive the corresponding DeepStack visual tap.
    """

    model_type = "granite_for_docling"
    sub_configs = {"text_config": GraniteForDoclingTextConfig, "vision_config": GraniteForDoclingVisionConfig}

    def __init__(
        self,
        image_token_id=128257,
        tie_word_embeddings=False,
        vision_config=None,
        text_config=None,
        scale_factor=4,
        mp_pooling_mode="pixel_shuffle_mlp_v2",
        use_deepstack=True,
        deepstack_visual_indexes=None,
        deepstack_attn_layers=None,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings

        if vision_config is None:
            self.vision_config = GraniteForDoclingVisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = GraniteForDoclingVisionConfig(**vision_config)
        elif isinstance(vision_config, GraniteForDoclingVisionConfig):
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "granite_for_docling_text")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("text_config is None, using default GraniteForDocling text config")
            text_config = GraniteForDoclingTextConfig()

        self.text_config = text_config
        self.scale_factor = scale_factor

        if mp_pooling_mode not in ("pixel_shuffle", "pixel_shuffle_mlp_v2"):
            raise ValueError(
                f"Unknown mp_pooling_mode={mp_pooling_mode!r}. Supported: 'pixel_shuffle', 'pixel_shuffle_mlp_v2'."
            )
        self.mp_pooling_mode = mp_pooling_mode

        self.use_deepstack = use_deepstack
        self.deepstack_visual_indexes = (
            list(deepstack_visual_indexes) if deepstack_visual_indexes is not None else [3, 7, 10]
        )
        self.deepstack_attn_layers = (
            list(deepstack_attn_layers) if deepstack_attn_layers is not None else [0, 1, 2]
        )
        if len(self.deepstack_visual_indexes) != len(self.deepstack_attn_layers):
            raise ValueError(
                "deepstack_visual_indexes and deepstack_attn_layers must have the same length "
                f"(got {len(self.deepstack_visual_indexes)} vs {len(self.deepstack_attn_layers)})."
            )

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


## Processing ##


class GraniteForDoclingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
            "return_mm_token_type_ids": False,
        },
    }


@auto_docstring
class GraniteForDoclingProcessor(Idefics3Processor):

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 169, chat_template=None, **kwargs):
        # Must run before any tiling happens: caps the shared GotOcr2 image processor's
        # own crop grid to match the tokenizer vocab (see MAX_TILE_GRID_DIM above).
        _install_tile_grid_dim_cap()
        super().__init__(
            image_processor, tokenizer=tokenizer, image_seq_len=image_seq_len, chat_template=chat_template, **kwargs
        )
        self.row_col_ids = self._collect_row_col_ids(tokenizer)

    @staticmethod
    def _collect_row_col_ids(tokenizer) -> set[int]:
        if tokenizer is None:
            return set()
        unk_id = tokenizer.unk_token_id
        return {
            tid
            for tid in (
                tokenizer.convert_tokens_to_ids(f"<row_{i + 1}_col_{j + 1}>")
                for i in range(16)
                for j in range(16)
            )
            if tid != unk_id
        }

    def _expand_image_tokens_in_ids(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Insert image_seq_len <image> token IDs after each row/col and global-img marker."""
        image_token_id = self.image_token_id
        row_col_ids = self.row_col_ids
        global_id = self.global_image_token_id
        image_seq_len = self.image_seq_len
        image_fill = [image_token_id] * image_seq_len
        mask_fill = [1] * image_seq_len

        expanded_ids = []
        expanded_mask = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            new_ids: list[int] = []
            new_mask: list[int] = []
            for tok_id, m in zip(ids_row, mask_row):
                new_ids.append(tok_id)
                new_mask.append(m)
                if tok_id in row_col_ids or tok_id == global_id:
                    new_ids.extend(image_fill)
                    new_mask.extend(mask_fill)
            expanded_ids.append(new_ids)
            expanded_mask.append(new_mask)
        return expanded_ids, expanded_mask

    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[GraniteForDoclingProcessorKwargs],
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchEncoding.

        This method extends the Idefics3Processor to handle GotOcr2ImageProcessor specifics.
        """
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            GraniteForDoclingProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        n_images_in_text = []
        n_images_in_images = []
        inputs = {}

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")
            n_images_in_text = [sample.count(self.image_token) for sample in text]

        if images is not None:
            images = self.image_processor.fetch_images(images)
            n_images_in_images = [len(sample) if isinstance(sample, (list, tuple)) else 1 for sample in images]
            images = [
                [sample] if not isinstance(sample, (list, tuple)) else sample
                for sample in images
            ]

            # Skip CPU-side rescale and normalize so pixel_values are transferred as uint8
            # (1 byte/pixel). The model rescales and normalizes on GPU instead.
            uint8_kwargs = dict(output_kwargs["images_kwargs"])
            uint8_kwargs["do_rescale"] = False
            uint8_kwargs["do_normalize"] = False
            image_inputs = self.image_processor(images, **uint8_kwargs)
            pv = image_inputs.get("pixel_values")
            if pv is not None:
                image_inputs["pixel_values"] = pv.to(dtype=torch.uint8)
            inputs.update(image_inputs)

            if text is not None:
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )

                image_rows = []
                image_cols = []
                for sample_images in images:
                    sample_image_rows = []
                    sample_image_cols = []
                    for img in sample_images:
                        width, height = img.size
                        n_cols, n_rows = get_optimal_tiled_canvas(
                            (height, width),
                            (
                                self.image_processor.size["height"],
                                self.image_processor.size["width"],
                            ),
                            self.image_processor.min_patches,
                            self.image_processor.max_patches,
                        )
                        sample_image_rows.append(n_rows)
                        sample_image_cols.append(n_cols)
                    image_rows.append(sample_image_rows)
                    image_cols.append(sample_image_cols)

                num_patches_arr = inputs.pop("num_patches", None)
                pixel_values = inputs.get("pixel_values")
                if pixel_values is not None and len(pixel_values.shape) == 4:
                    batch_size = len(images)

                    if num_patches_arr is not None:
                        img_offset = 0
                        patches_per_sample = []
                        for sample_imgs in images:
                            n_imgs = len(sample_imgs)
                            patches_per_sample.append(int(sum(num_patches_arr[img_offset : img_offset + n_imgs])))
                            img_offset += n_imgs
                    else:
                        total = pixel_values.shape[0]
                        patches_per_sample = [total // batch_size] * batch_size

                    max_patches = max(patches_per_sample)
                    if all(p == max_patches for p in patches_per_sample):
                        inputs["pixel_values"] = pixel_values.reshape(batch_size, max_patches, *pixel_values.shape[1:])
                    else:
                        padded = torch.zeros(
                            batch_size,
                            max_patches,
                            *pixel_values.shape[1:],
                            dtype=pixel_values.dtype,
                            device=pixel_values.device,
                        )
                        pv_offset = 0
                        for i, count in enumerate(patches_per_sample):
                            padded[i, :count] = pixel_values[pv_offset : pv_offset + count]
                            pv_offset += count
                        inputs["pixel_values"] = padded

                    pam = torch.zeros(batch_size, max_patches, dtype=torch.bool)
                    for i, count in enumerate(patches_per_sample):
                        pam[i, :count] = True
                    inputs["pixel_attention_mask"] = pam

                fake_image_token = self.fake_image_token
                image_token = self.image_token
                global_img_token = self.global_image_tag

                prompt_strings = []
                compact_prompt_strings = []
                batch_image_seq_lengths = []
                for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
                    image_prompt_strings = []
                    compact_image_prompt_strings = []
                    image_seq_lengths = []
                    for n_rows, n_cols in zip(sample_rows, sample_cols):
                        image_prompt_string = get_image_prompt_string(
                            n_rows,
                            n_cols,
                            self.image_seq_len,
                            image_token=image_token,
                            fake_token_around_image=fake_image_token,
                            global_img_token=global_img_token,
                        )
                        compact_image_prompt_string = get_compact_image_prompt_string(
                            n_rows,
                            n_cols,
                            fake_token_around_image=fake_image_token,
                            global_img_token=global_img_token,
                        )
                        row_length = (self.image_seq_len + 2) * n_cols + 1
                        image_seq_lengths.append((self.image_seq_len + 3) + row_length * n_rows)
                        image_prompt_strings.append(image_prompt_string)
                        compact_image_prompt_strings.append(compact_image_prompt_string)

                    batch_image_seq_lengths.append(image_seq_lengths)
                    split_sample = sample.split(image_token)
                    if len(split_sample) == 0:
                        raise ValueError("The image token should be present in the text.")

                    full_sample = split_sample[0]
                    compact_sample = split_sample[0]
                    for i, (full_str, compact_str) in enumerate(
                        zip(image_prompt_strings, compact_image_prompt_strings)
                    ):
                        full_sample += full_str + split_sample[i + 1]
                        compact_sample += compact_str + split_sample[i + 1]
                    prompt_strings.append(full_sample)
                    compact_prompt_strings.append(compact_sample)

                text_inputs = self.tokenizer(compact_prompt_strings, **output_kwargs["text_kwargs"])

                input_ids = text_inputs["input_ids"]
                attention_mask = text_inputs.get(
                    "attention_mask", [[1] * len(ids) for ids in input_ids]
                )
                expanded_ids, expanded_mask = self._expand_image_tokens_in_ids(
                    input_ids, attention_mask
                )
                max_len = max(len(ids) for ids in expanded_ids)
                pad_id = self.tokenizer.pad_token_id
                expanded_ids = [
                    ids + [pad_id] * (max_len - len(ids)) for ids in expanded_ids
                ]
                expanded_mask = [
                    m + [0] * (max_len - len(m)) for m in expanded_mask
                ]

                text_inputs["input_ids"] = expanded_ids
                if "attention_mask" in text_inputs:
                    text_inputs["attention_mask"] = expanded_mask

                self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])
                inputs.update(text_inputs)

        elif text is not None:
            if any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            inputs.update(text_inputs)

        if return_mm_token_type_ids:
            array_ids = np.array(inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            for i, seq_lengths in enumerate(batch_image_seq_lengths):
                image_start_positions = np.where(array_ids[i] == self.fake_image_token_id)[0]
                j = 0
                for seq_len in seq_lengths:
                    if j >= len(image_start_positions):
                        break
                    start = image_start_positions[j]
                    end = start + seq_len
                    mm_token_type_ids[i, start:end] = 1
                    j = np.searchsorted(image_start_positions, end)

            inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data=inputs, tensor_type=return_tensors)


## Modeling ##


class GraniteForDoclingBaseModelOutputWithPast(Idefics3BaseModelOutputWithPast):
    pass


class GraniteForDoclingCausalLMOutputWithPast(Idefics3CausalLMOutputWithPast):
    pass


class GraniteForDoclingTextPreTrainedModel(PreTrainedModel):
    config_class = GraniteForDoclingTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GraniteForDoclingTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _is_stateful = True


class GraniteForDoclingTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GraniteForDoclingTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GraniteMoeHybridAttention(config, layer_idx)
        self.input_layernorm = GraniteMoeSharedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GraniteMoeSharedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.shared_mlp = GraniteMoeHybridMLP(config)
        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


class GraniteForDoclingTextModel(GraniteForDoclingTextPreTrainedModel):
    """Dense Granite-style text decoder with optional DeepStack residual injection."""

    def __init__(self, config: GraniteForDoclingTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GraniteForDoclingTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GraniteMoeSharedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = (
            GraniteMoeHybridRotaryEmbedding(config) if config.position_embedding_type == "rope" else None
        )
        self.embedding_multiplier = config.embedding_multiplier
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        deepstack_attn_layers: list[int] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
            Boolean mask marking image-token positions in the LM sequence. Only set during prefill.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            One tensor per ViT tap, shape `(num_image_tokens, hidden_size)`. Slot `i` is added at
            decoder layer `deepstack_attn_layers[i]`.
        deepstack_attn_layers (`list[int]`, *optional*):
            Decoder layer indices that receive each DeepStack tap.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        deepstack_active = deepstack_visual_embeds is not None and deepstack_attn_layers is not None
        ds_layer_to_slot: dict[int, int] = (
            {layer_idx: slot for slot, layer_idx in enumerate(deepstack_attn_layers)}
            if deepstack_active
            else {}
        )

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if deepstack_active and visual_pos_masks is not None and i in ds_layer_to_slot:
                slot = ds_layer_to_slot[i]
                img_feats = deepstack_visual_embeds[slot].to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                pos_masks = visual_pos_masks.to(hidden_states.device)
                hidden_states = hidden_states.clone()
                hidden_states[pos_masks, :] = hidden_states[pos_masks, :] + img_feats

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GraniteForDoclingPreTrainedModel(Idefics3PreTrainedModel):
    config_class = GraniteForDoclingConfig

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, GraniteForDoclingConnector) and module.pooling_mode == "pixel_shuffle_mlp_v2":
            pos_embed = _build_2d_sincos_pos_embed(
                module.pos_embed_2d.shape[-1], int(module.pos_embed_2d.shape[0] ** 0.5)
            )
            module.pos_embed_2d.data.copy_(pos_embed.to(module.pos_embed_2d.device))


class GraniteForDoclingDeepStackMerger(nn.Module):
    """Projects an intermediate ViT tap into LM hidden size for DeepStack injection."""

    def __init__(self, vision_hidden_size: int, scale_factor: int, text_hidden_size: int):
        super().__init__()
        self.scale_factor = scale_factor
        merged_dim = vision_hidden_size * (scale_factor**2)
        self.norm = nn.LayerNorm(merged_dim)
        self.fc1 = nn.Linear(merged_dim, text_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(text_hidden_size, text_hidden_size)

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.reshape(bsz, height, width, embed_dim)
        x = x.reshape(bsz, height, int(width / self.scale_factor), embed_dim * self.scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / self.scale_factor),
            int(height / self.scale_factor),
            embed_dim * (self.scale_factor**2),
        )
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (self.scale_factor**2)), embed_dim * (self.scale_factor**2))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_shuffle(x.contiguous())
        x = self.norm(x)
        return self.fc2(self.act(self.fc1(x)))


def _build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """2D sin-cos positional embedding of shape ``[grid_size**2, embed_dim]``."""
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sincos positional embeddings.")

    def _1d_sincos(dim: int, pos: np.ndarray) -> np.ndarray:
        omega = np.arange(dim // 2, dtype=np.float32)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    emb_h = _1d_sincos(embed_dim // 2, grid[0])
    emb_w = _1d_sincos(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return torch.from_numpy(emb).float()


class GraniteForDoclingConnector(Idefics3Connector):
    """Modality projector supporting ``pixel_shuffle`` and ``pixel_shuffle_mlp_v2``."""

    def __init__(self, config):
        super().__init__(config)
        self.pooling_mode = getattr(config, "mp_pooling_mode", "pixel_shuffle")

        text_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size

        self.mlp_fc2: nn.Linear | None = None
        self.ln_in: nn.LayerNorm | None = None
        self.ln_mid: nn.LayerNorm | None = None
        self.ln_out: nn.LayerNorm | None = None
        if self.pooling_mode == "pixel_shuffle_mlp_v2":
            self.mlp_fc2 = nn.Linear(text_hidden_size, text_hidden_size, bias=False)
            self.ln_in = nn.LayerNorm(vision_hidden_size)
            self.ln_mid = nn.LayerNorm(text_hidden_size)
            self.ln_out = nn.LayerNorm(text_hidden_size)

            tokens_per_tile = (
                (config.vision_config.image_size // config.vision_config.patch_size) ** 2
                // (config.scale_factor**2)
            )
            grid_size = int(tokens_per_tile**0.5)
            if grid_size * grid_size != tokens_per_tile:
                raise ValueError(
                    "image_seq_len must be a perfect square for pixel_shuffle_mlp_v2; "
                    f"got {tokens_per_tile} (image_size={config.vision_config.image_size}, "
                    f"patch_size={config.vision_config.patch_size}, scale_factor={config.scale_factor})."
                )
            pos_embed = _build_2d_sincos_pos_embed(text_hidden_size, grid_size)
            self.register_buffer("pos_embed_2d", pos_embed, persistent=False)

        self.deepstack_mergers: nn.ModuleList | None = None
        if getattr(config, "use_deepstack", False):
            self.deepstack_mergers = nn.ModuleList(
                [
                    GraniteForDoclingDeepStackMerger(
                        config.vision_config.hidden_size,
                        config.scale_factor,
                        config.text_config.hidden_size,
                    )
                    for _ in config.deepstack_visual_indexes
                ]
            )

    def forward(self, image_hidden_states):
        if self.pooling_mode == "pixel_shuffle":
            x = self.pixel_shuffle(image_hidden_states, self.scale_factor)
            return self.modality_projection(x)

        x = self.ln_in(image_hidden_states)
        x = self.pixel_shuffle(x, self.scale_factor)
        x = self.modality_projection(x)
        x = x + self.pos_embed_2d.to(dtype=x.dtype)
        x = self.ln_mid(x)
        x = nn.functional.gelu(x)
        x = self.mlp_fc2(x)
        x = self.ln_out(x)
        return x


class GraniteForDoclingModel(Idefics3Model):
    config_class = GraniteForDoclingConfig

    def __init__(self, config: GraniteForDoclingConfig):
        super().__init__(config)
        self.text_model = GraniteForDoclingTextModel._from_config(config.text_config)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask=None,
        **kwargs,
    ):
        r"""
        pixel_attention_mask (`torch.BoolTensor`, *optional*):
            Unused. GotOcr2ImageProcessor crops each tile to exactly the ViT input
            resolution, so every pixel is real. Kept for API compatibility only.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
        if pixel_attention_mask is not None:
            real_images_inds = pixel_attention_mask.view(-1).bool()
        else:
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        if pixel_values.dtype == torch.uint8:
            pixel_values = pixel_values.to(dtype=self.dtype) / 255.0
            mean = torch.tensor(
                self.config.vision_config.image_mean, dtype=self.dtype, device=pixel_values.device
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                self.config.vision_config.image_std, dtype=self.dtype, device=pixel_values.device
            ).view(1, 3, 1, 1)
            pixel_values = (pixel_values - mean) / std
        else:
            pixel_values = pixel_values.to(dtype=self.dtype)

        deepstack_enabled = (
            getattr(self.config, "use_deepstack", False)
            and self.connector.deepstack_mergers is not None
        )
        if deepstack_enabled:
            kwargs.setdefault("output_hidden_states", True)
        image_outputs = self.vision_model(
            pixel_values=pixel_values, patch_attention_mask=None, return_dict=True, **kwargs
        )
        image_hidden_states = image_outputs.last_hidden_state
        image_features = self.connector(image_hidden_states)
        image_outputs.pooler_output = image_features

        if deepstack_enabled and image_outputs.hidden_states is not None:
            ds_indexes = self.config.deepstack_visual_indexes
            image_outputs.deepstack_features = [
                self.connector.deepstack_mergers[slot](image_outputs.hidden_states[layer_idx])
                for slot, layer_idx in enumerate(ds_indexes)
            ]
        else:
            image_outputs.deepstack_features = None

        return image_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | GraniteForDoclingBaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        deepstack_visual_embeds: list[torch.Tensor] | None = None
        visual_pos_masks: torch.Tensor | None = None
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, pixel_attention_mask)
            image_hidden_states = image_outputs.pooler_output
            ds_features = getattr(image_outputs, "deepstack_features", None)
            if ds_features is not None and input_ids is not None:
                lm_dim = self.config.text_config.hidden_size
                deepstack_visual_embeds = [feat.reshape(-1, lm_dim) for feat in ds_features]
                visual_pos_masks = input_ids == self.config.image_token_id
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        deepstack_kwargs = {}
        if deepstack_visual_embeds is not None:
            deepstack_kwargs = {
                "visual_pos_masks": visual_pos_masks,
                "deepstack_visual_embeds": deepstack_visual_embeds,
                "deepstack_attn_layers": list(self.config.deepstack_attn_layers),
            }

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **deepstack_kwargs,
            **kwargs,
        )

        return GraniteForDoclingBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The GraniteForDocling model with a language modeling head: SigLIP vision encoder and dense Granite-style decoder.
    """
)
class GraniteForDoclingForConditionalGeneration(Idefics3ForConditionalGeneration):
    config_class = GraniteForDoclingConfig

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteForDoclingCausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits_scaling = getattr(self.config.text_config, "logits_scaling", 1)
        if logits_scaling != 1:
            logits = logits / logits_scaling

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return GraniteForDoclingCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "GraniteForDoclingConfig",
    "GraniteForDoclingForConditionalGeneration",
    "GraniteForDoclingModel",
    "GraniteForDoclingPreTrainedModel",
    "GraniteForDoclingProcessor",
    "GraniteForDoclingTextConfig",
    "GraniteForDoclingVisionConfig",
]
