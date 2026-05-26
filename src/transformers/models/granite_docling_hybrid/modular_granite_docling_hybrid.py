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
"""PyTorch GraniteDoclingHybrid model."""

from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_utils import ImageInput
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import BatchEncoding, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING
from ..got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
from ..granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig
from ..granitemoehybrid.modeling_granitemoehybrid import (
    GraniteFlashAttentionKwargs,
    GraniteMoeHybridModel,
    HybridMambaAttentionDynamicCache,
)
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
from .configuration_granite_docling_hybrid import GraniteDoclingHybridConfig


def _prompt_split_image(image_seq_len, image_rows, image_cols, fake_token_around_image, image_token, global_img_token):
    """Prompt with expanded image tokens for when the image is split into patches.

    Only includes the global thumbnail token when there are multiple tiles (image_rows * image_cols > 1),
    matching the nanoVLM training-time behavior. GotOcr2ImageProcessor likewise only appends the
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
    """Compact version of _prompt_split_image without repeated <image> tokens.

    Image tokens are inserted post-tokenization via _expand_image_tokens_in_ids,
    avoiding the cost of tokenizing hundreds of identical <image> placeholder strings
    (recipe ⑦ from Huang et al., 2026 — a primary TTFT bottleneck for compact VLMs).
    """
    text = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text += f"{fake_token_around_image}<row_{n_h + 1}_col_{n_w + 1}>"
        text += "\n"
    if image_rows * image_cols > 1:
        text += f"\n{fake_token_around_image}{global_img_token}{fake_token_around_image}"
    return text


def _compact_prompt_single_image(fake_token_around_image, global_img_token):
    """Compact version of _prompt_single_image without repeated <image> tokens."""
    return f"{fake_token_around_image}{global_img_token}{fake_token_around_image}"


def get_compact_image_prompt_string(image_rows, image_cols, fake_token_around_image, global_img_token):
    """Returns a compact prompt string without repeated <image> tokens.

    Used for efficient tokenization: the tokenizer only processes structural tokens
    (fake/row_col/global), and image tokens are spliced into input_ids afterwards.
    """
    if image_rows == 0 and image_cols == 0:
        return _compact_prompt_single_image(fake_token_around_image, global_img_token)
    return _compact_prompt_split_image(image_rows, image_cols, fake_token_around_image, global_img_token)


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput


logger = logging.get_logger(__name__)


## Configuration ##


class GraniteDoclingHybridVisionConfig(Idefics3VisionConfig):
    pass


class GraniteDoclingHybridTextConfig(GraniteMoeHybridConfig):
    pass


class GraniteDoclingHybridConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GraniteDoclingHybridModel`]. It is used to instantiate a
    GraniteDoclingHybrid model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Idefics3 model architecture,
    but with a GraniteMoeHybrid text model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_token_id (`int`, *optional*, defaults to 128257):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`GraniteDoclingHybridVisionConfig` or `dict`, *optional*, defaults to `GraniteDoclingHybridVisionConfig`):
            Custom vision config or dict for the vision tower
        text_config (`PretrainedConfig` or `dict`, *optional*, defaults to `GraniteMoeHybridConfig`):
            Custom text config or dict for the text model
        scale_factor (`int`, *optional*, defaults to 2):
            The scale factor for the image encoder.
        mp_pooling_mode (`str`, *optional*, defaults to `"pixel_shuffle"`):
            Modality-projector variant. Supported: `"pixel_shuffle"`, `"pixel_shuffle_mlp"`,
            `"pixel_shuffle_mlp_v2"`. The two MLP variants add a post-projection GELU + Linear
            channel mix in LM space; `v2` additionally inserts pre-shuffle / mid / post LayerNorms
            and a within-tile 2D sincos positional embedding (matching MiniCPM-style resamplers).
            The structure mirrors nanoVLM's `mp_pooling_mode`.
        use_deepstack (`bool`, *optional*, defaults to `False`):
            Whether to enable Qwen3-VL-style DeepStack: intermediate ViT block outputs are projected
            through dedicated mergers and added (residually) to the LM hidden states at image-token
            positions after specific decoder layers. See https://arxiv.org/abs/2406.04334.
        deepstack_visual_indexes (`list[int]`, *optional*, defaults to `[3, 6, 9]`):
            Indices of vision encoder blocks whose outputs (pre-final-LayerNorm) feed DeepStack.
            Slot `i` of this list is paired with slot `i` of `deepstack_attn_layers`.
        deepstack_attn_layers (`list[int]`, *optional*, defaults to `[10, 13, 17]`):
            Indices of language model decoder layers that receive the residual visual injection
            from the corresponding ViT tap. Same length as `deepstack_visual_indexes`.

    Example:
    ```python
    >>> from transformers import GraniteDoclingHybridModel, GraniteDoclingHybridConfig
    >>> # Initializing configuration
    >>> configuration = GraniteDoclingHybridConfig()
    >>> # Initializing a model from the configuration
    >>> model = GraniteDoclingHybridModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_docling_hybrid"
    sub_configs = {"text_config": CONFIG_MAPPING, "vision_config": GraniteDoclingHybridVisionConfig}

    def __init__(
        self,
        image_token_id=128257,
        tie_word_embeddings=False,
        vision_config=None,
        text_config=None,
        scale_factor=2,
        mp_pooling_mode="pixel_shuffle",
        use_deepstack=False,
        deepstack_visual_indexes=None,
        deepstack_attn_layers=None,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings

        if vision_config is None:
            self.vision_config = GraniteDoclingHybridVisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = GraniteDoclingHybridVisionConfig(**vision_config)
        elif isinstance(vision_config, GraniteDoclingHybridVisionConfig):
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "granitemoehybrid")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("text_config is None, using default GraniteMoeHybrid text config")
            text_config = CONFIG_MAPPING["granitemoehybrid"]()

        self.text_config = text_config
        self.scale_factor = scale_factor

        if mp_pooling_mode not in ("pixel_shuffle", "pixel_shuffle_mlp", "pixel_shuffle_mlp_v2"):
            raise ValueError(
                f"Unknown mp_pooling_mode={mp_pooling_mode!r}. Supported: "
                "'pixel_shuffle', 'pixel_shuffle_mlp', 'pixel_shuffle_mlp_v2'."
            )
        self.mp_pooling_mode = mp_pooling_mode

        self.use_deepstack = use_deepstack
        self.deepstack_visual_indexes = (
            list(deepstack_visual_indexes) if deepstack_visual_indexes is not None else [3, 6, 9]
        )
        self.deepstack_attn_layers = (
            list(deepstack_attn_layers) if deepstack_attn_layers is not None else [10, 13, 17]
        )
        if len(self.deepstack_visual_indexes) != len(self.deepstack_attn_layers):
            raise ValueError(
                "deepstack_visual_indexes and deepstack_attn_layers must have the same length "
                f"(got {len(self.deepstack_visual_indexes)} vs {len(self.deepstack_attn_layers)})."
            )

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


## Processing ##


class GraniteDoclingHybridProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "is_split_into_words": False,
            "return_mm_token_type_ids": False,
        },
    }


@auto_docstring
class GraniteDoclingHybridProcessor(Idefics3Processor):

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 169, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer=tokenizer, image_seq_len=image_seq_len, chat_template=chat_template, **kwargs)
        # Cover up to 16×16 grids (max_patches=256); stored as a set for O(1) lookup
        # during post-tokenization image-token expansion. Filters out the tokenizer's UNK id
        # so unknown row/col tokens don't collapse to a single entry. `unk_token_id` is
        # inlined into the comprehension because the modular converter drops bare local
        # variables when it merges the parent class's __init__ body.
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
        """Insert image_seq_len <image> token IDs after each row/col and global-img marker.

        This is the counterpart to compact tokenization (recipe ⑦, Huang et al. 2026):
        instead of tokenizing a prompt that contains hundreds of repeated '<image>' strings,
        we tokenize a compact version and expand here with a fast Python list splice.

        The resulting input_ids are identical to what full-prompt tokenization produces,
        so the model forward pass is unaffected.
        """
        image_token_id = self.image_token_id
        row_col_ids = self.row_col_ids  # set[int]
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
        **kwargs: Unpack[GraniteDoclingHybridProcessorKwargs],
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchEncoding.

        This method extends the Idefics3Processor to handle GotOcr2ImageProcessor specifics.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `list[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            text (`Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<row_x_col_y>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
            return_tensors (`Union[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            GraniteDoclingHybridProcessorKwargs,
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

            # Recipe ⑧ (Huang et al. 2026 §3.3): skip CPU-side rescale + normalize so
            # pixel_values are transferred as uint8 (1 byte/pixel) instead of float32
            # (4 bytes) or bfloat16 (2 bytes). The model normalizes on GPU instead.
            uint8_kwargs = dict(output_kwargs["images_kwargs"])
            uint8_kwargs["do_rescale"] = False
            uint8_kwargs["do_normalize"] = False
            image_inputs = self.image_processor(images, **uint8_kwargs)
            # Convert float32 [0, 255] → uint8 [0, 255]: lossless for integer-valued pixels
            pv = image_inputs.get("pixel_values")
            if pv is not None:
                image_inputs["pixel_values"] = pv.to(dtype=torch.uint8)
            inputs.update(image_inputs)

            if text is not None:
                if n_images_in_images != n_images_in_text:
                    raise ValueError(
                        f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                    )

                # GotOcr2ImageProcessor doesn't return rows/cols, compute them
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

                # Post-process inputs for GotOcr2ImageProcessor
                num_patches_arr = inputs.pop("num_patches", None)
                pixel_values = inputs.get("pixel_values")
                if pixel_values is not None and len(pixel_values.shape) == 4:
                    # TODO: change
                    # Make 5D to match Idefics3 expected format: (batch, num_images, C, H, W)
                    # pixel_values is (total_sub_images, C, H, W) — a flat concat of all patches
                    # from all images. We need to group them per sample and pad to the max
                    # patch count so the tensor is rectangular.  The Idefics3 model's
                    # get_image_features will filter out zero-padded (all-zero) sub-images.
                    batch_size = len(images)

                    # Compute number of sub-images per sample (sum patches of each image in the sample)
                    if num_patches_arr is not None:
                        img_offset = 0
                        patches_per_sample = []
                        for sample_imgs in images:
                            n_imgs = len(sample_imgs)
                            patches_per_sample.append(int(sum(num_patches_arr[img_offset:img_offset + n_imgs])))
                            img_offset += n_imgs
                    else:
                        # Fallback: assume equal distribution
                        total = pixel_values.shape[0]
                        patches_per_sample = [total // batch_size] * batch_size

                    max_patches = max(patches_per_sample)
                    if all(p == max_patches for p in patches_per_sample):
                        # All samples have the same number of sub-images — simple reshape
                        inputs["pixel_values"] = pixel_values.reshape(batch_size, max_patches, *pixel_values.shape[1:])
                    else:
                        # Variable sub-image counts — pad shorter samples with zeros
                        # (the model filters out all-zero sub-images in get_image_features)
                        padded = torch.zeros(
                            batch_size, max_patches, *pixel_values.shape[1:],
                            dtype=pixel_values.dtype, device=pixel_values.device,
                        )
                        pv_offset = 0
                        for i, count in enumerate(patches_per_sample):
                            padded[i, :count] = pixel_values[pv_offset:pv_offset + count]
                            pv_offset += count
                        inputs["pixel_values"] = padded

                    # TODO: change
                    # Explicit mask so get_image_features can distinguish real
                    # sub-images from zero-padding without relying on pixel values
                    # (a genuinely all-black tile would otherwise be dropped).
                    pam = torch.zeros(batch_size, max_patches, dtype=torch.bool)
                    for i, count in enumerate(patches_per_sample):
                        pam[i, :count] = True
                    inputs["pixel_attention_mask"] = pam

                fake_image_token = self.fake_image_token
                image_token = self.image_token
                global_img_token = self.global_image_tag

                prompt_strings = []         # full expanded — only used for _check_special_mm_tokens
                compact_prompt_strings = [] # compact — used for tokenization (no repeated <image> tokens)
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
                        # Add +2 and +3 for special BOI/EOI/fake_image_wrapper tokens
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

                # Tokenize compact prompts — avoids tokenizing N*image_seq_len repeated
                # '<image>' strings (recipe ⑦, Huang et al. 2026 §3.3).
                text_inputs = self.tokenizer(compact_prompt_strings, **output_kwargs["text_kwargs"])

                # Expand: splice image_seq_len <image> token IDs after each row/col and
                # global-img marker. Result is identical to full-prompt tokenization.
                input_ids = text_inputs["input_ids"]
                attention_mask = text_inputs.get(
                    "attention_mask", [[1] * len(ids) for ids in input_ids]
                )
                expanded_ids, expanded_mask = self._expand_image_tokens_in_ids(
                    input_ids, attention_mask
                )
                # TODO: change
                # Pad expanded sequences to uniform length so they can be stacked
                # into a tensor. Different images produce different tile counts,
                # so post-expansion lengths vary across the batch.
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

class GraniteDoclingHybridBaseModelOutputWithPast(Idefics3BaseModelOutputWithPast):
    pass


class GraniteDoclingHybridCausalLMOutputWithPast(Idefics3CausalLMOutputWithPast):
    pass


class HybridMambaAttentionDynamicCache(HybridMambaAttentionDynamicCache):
    pass


class GraniteDoclingHybridPreTrainedModel(Idefics3PreTrainedModel):
    config_class = GraniteDoclingHybridConfig

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        # ``pos_embed_2d`` is a non-persistent buffer computed deterministically in
        # ``GraniteDoclingHybridConnector.__init__``. ``from_pretrained`` runs the
        # model through ``to_empty()`` before loading state_dict, which replaces
        # buffer storage with uninitialized memory. Non-persistent buffers aren't
        # in the checkpoint, so they keep that garbage unless we refill them here.
        # Without this, every launch produces a different (often NaN-laden) pos_embed,
        # making generation non-deterministic and frequently catastrophic.
        if isinstance(module, GraniteDoclingHybridConnector) and module.pooling_mode == "pixel_shuffle_mlp_v2":
            pos_embed = _build_2d_sincos_pos_embed(
                module.pos_embed_2d.shape[-1], int(module.pos_embed_2d.shape[0] ** 0.5)
            )
            module.pos_embed_2d.data.copy_(pos_embed.to(module.pos_embed_2d.device))


class GraniteDoclingHybridDeepStackMerger(nn.Module):
    """Per-tap projector for Qwen3-VL-style DeepStack (Huang et al., arXiv:2406.04334).

    Projects intermediate ViT features into LM embedding space via
    ``pixel_shuffle -> LayerNorm -> Linear -> GELU -> Linear``. One instance is
    created per entry of ``config.deepstack_visual_indexes``; each is independent
    from the main modality projection used for the first-layer visual tokens.
    """

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
    """2D sin-cos positional embedding of shape ``[grid_size**2, embed_dim]``.

    half the channels encode the row index, the other half the column index, each
    with a 1D sin-cos basis. ``np.meshgrid(grid_w, grid_h)`` gives the same
    column-major axis order nanoVLM uses, so the resulting tensor lines up
    byte-for-byte with a checkpoint trained under ``pixel_shuffle_mlp_v2``.
    """
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
    emb = np.concatenate([emb_h, emb_w], axis=1)  # [grid_size**2, embed_dim]
    return torch.from_numpy(emb).float()


class GraniteDoclingHybridConnector(Idefics3Connector):
    """Modality projector matching nanoVLM's three ``mp_pooling_mode`` variants.

    Modes (all share the same ``pixel_shuffle`` reshape rule and same ``proj`` Linear):

    1. ``pixel_shuffle`` — ``pixel_shuffle -> proj``. Matches the stock Idefics3
       connector behaviour and the original `Idefics3Connector` weight layout.
    2. ``pixel_shuffle_mlp`` — adds a post-projection channel mix
       ``-> GELU -> mlp_fc2`` in LM space (no LayerNorms, no positional embedding).
    3. ``pixel_shuffle_mlp_v2`` — full MiniCPM-style resampler:
       ``ln_in(vit_dim) -> pixel_shuffle -> proj -> + pos_embed_2d ->
       ln_mid(lm_dim) -> GELU -> mlp_fc2 -> ln_out(lm_dim)``. The ``pos_embed_2d``
       buffer is the same MAE-style 2D sincos table nanoVLM builds — registered
       non-persistent so it doesn't bloat checkpoints; rebuilt at construction.

    When ``config.use_deepstack`` is True, ``deepstack_mergers`` is an
    ``nn.ModuleList`` of [`GraniteDoclingHybridDeepStackMerger`] — one per ViT
    tap in ``config.deepstack_visual_indexes``. DeepStack is orthogonal to the
    pooling mode; the merger has its own pre-projection LN.

    Note on the seg-rate side channel: nanoVLM's v2 mode also accepts an
    optional ``mp_use_seg_rate_embed`` toggle that adds a small ``Embedding(4,
    lm_dim)`` lookup driven by a predicted compression map. That path requires
    plumbing a compression map through ``forward`` and is intentionally not
    mirrored here — bring it across if/when you need it.
    """

    def __init__(self, config):
        super().__init__(config)
        self.pooling_mode = getattr(config, "mp_pooling_mode", "pixel_shuffle")

        text_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size

        # ---- pixel_shuffle_mlp / pixel_shuffle_mlp_v2 channel-mix layer ----
        self.mlp_fc2: nn.Linear | None = None
        if self.pooling_mode in ("pixel_shuffle_mlp", "pixel_shuffle_mlp_v2"):
            self.mlp_fc2 = nn.Linear(text_hidden_size, text_hidden_size, bias=False)

        # ---- pixel_shuffle_mlp_v2 extras ----
        self.ln_in: nn.LayerNorm | None = None
        self.ln_mid: nn.LayerNorm | None = None
        self.ln_out: nn.LayerNorm | None = None
        if self.pooling_mode == "pixel_shuffle_mlp_v2":
            self.ln_in = nn.LayerNorm(vision_hidden_size)
            self.ln_mid = nn.LayerNorm(text_hidden_size)
            self.ln_out = nn.LayerNorm(text_hidden_size)

            # Within-tile 2D sincos. Grid side = sqrt(image_seq_len) where
            # image_seq_len = (image_size // patch_size)^2 / scale_factor^2.
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
                    GraniteDoclingHybridDeepStackMerger(
                        config.vision_config.hidden_size,
                        config.scale_factor,
                        config.text_config.hidden_size,
                    )
                    for _ in config.deepstack_visual_indexes
                ]
            )

    def forward(self, image_hidden_states):
        # The modular converter flattens `Idefics3Connector` inheritance into a direct
        # nn.Module subclass, so we can't call `super().forward(...)` for the plain
        # pixel_shuffle path — inline it instead.
        if self.pooling_mode == "pixel_shuffle":
            x = self.pixel_shuffle(image_hidden_states, self.scale_factor)
            return self.modality_projection(x)

        if self.pooling_mode == "pixel_shuffle_mlp":
            x = self.pixel_shuffle(image_hidden_states, self.scale_factor)
            x = self.modality_projection(x)
            x = nn.functional.gelu(x)
            x = self.mlp_fc2(x)
            return x

        # pixel_shuffle_mlp_v2
        x = self.ln_in(image_hidden_states)
        x = self.pixel_shuffle(x, self.scale_factor)
        x = self.modality_projection(x)
        x = x + self.pos_embed_2d.to(dtype=x.dtype)
        x = self.ln_mid(x)
        x = nn.functional.gelu(x)
        x = self.mlp_fc2(x)
        x = self.ln_out(x)
        return x


class GraniteDoclingHybridTextModel(GraniteMoeHybridModel):
    """GraniteMoeHybrid text model with optional DeepStack residual injection.

    Equivalent to [`GraniteMoeHybridModel`] when DeepStack kwargs are not provided.
    During prefill, accepts ``deepstack_visual_embeds`` (one per ViT tap) plus
    ``visual_pos_masks`` and ``deepstack_attn_layers`` and, after each decoder
    layer whose index appears in ``deepstack_attn_layers``, adds the corresponding
    projected visual features to the hidden states at image-token positions only.
    Text-token hidden states are untouched. See https://arxiv.org/abs/2406.04334.
    """

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
        **kwargs: Unpack[GraniteFlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
            Boolean mask marking image-token positions in the LM sequence. Only set during prefill;
            None during the autoregressive decode steps (where no image tokens appear).
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            One tensor per ViT tap, shape `(num_image_tokens, hidden_size)`. Slot `i` corresponds
            to tap `deepstack_visual_indexes[i]` and is added at LM layer `deepstack_attn_layers[i]`.
        deepstack_attn_layers (`list[int]`, *optional*):
            LM decoder layer indices that receive each DeepStack tap. Same length as
            `deepstack_visual_embeds`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        mamba_mask = self._update_mamba_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Index → slot map for O(1) lookup inside the layer loop.
        deepstack_active = deepstack_visual_embeds is not None and deepstack_attn_layers is not None
        ds_layer_to_slot: dict[int, int] = (
            {layer_idx: slot for slot, layer_idx in enumerate(deepstack_attn_layers)}
            if deepstack_active
            else {}
        )

        for i, decoder_layer in enumerate(self.layers):
            # Depending on the layer type we opt for 2D base attention mask (Mamba) or 4D causal mask (Attention)
            layer_mask = mamba_mask if self.config.layers_block_type[i] == "mamba" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # DeepStack residual injection at image-token positions. Restricted to prefill —
            # `visual_pos_masks` is None during autoregressive decoding so we skip naturally.
            if deepstack_active and visual_pos_masks is not None and i in ds_layer_to_slot:
                slot = ds_layer_to_slot[i]
                img_feats = deepstack_visual_embeds[slot].to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                pos_masks = visual_pos_masks.to(hidden_states.device)
                hidden_states = hidden_states.clone()
                hidden_states[pos_masks, :] = hidden_states[pos_masks, :] + img_feats

        hidden_states = self.norm(hidden_states)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GraniteDoclingHybridModel(Idefics3Model):
    config_class = GraniteDoclingHybridConfig

    def __init__(self, config: GraniteDoclingHybridConfig):
        super().__init__(config)
        # Replace the AutoModel-instantiated GraniteMoeHybridModel with our subclass that
        # knows how to consume DeepStack kwargs and inject residuals between decoder layers.
        # Re-running `_from_config` here is fine: the previous text_model is dropped before
        # any forward pass and our `post_init` initialises weights identically.
        self.text_model = GraniteDoclingHybridTextModel._from_config(config.text_config)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask=None,
        **kwargs,
    ):
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        pixel_attention_mask:
            Unused. GotOcr2ImageProcessor crops each tile to exactly the ViT input
            resolution, so every pixel is real — the mask is always all-ones.
            Kept in the signature for API compatibility only.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
        # TODO: change
        # Determine which sub-images are real vs zero-padding.
        # Prefer the explicit mask from the processor (handles all-black tiles
        # correctly); fall back to the pixel-value heuristic for backward compat.
        if pixel_attention_mask is not None:
            real_images_inds = pixel_attention_mask.view(-1).bool()
        else:
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        if pixel_values.dtype == torch.uint8:
            # Recipe ⑧ (Huang et al. 2026 §3.3): uint8 path — images were transferred as
            # 1 byte/pixel; rescale and normalize here on GPU to avoid CPU-side float ops
            # and reduce PCIe bandwidth 4× vs float32.
            # TODO: change
            # nb_values_per_image = pixel_values.shape[1:].numel()
            # real_images_inds = pixel_values.sum(dim=(-1, -2, -3)) != 0
            # if not real_images_inds.all():
            #     real_images_inds = (pixel_values == 0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            # pixel_values = pixel_values[real_images_inds].contiguous()
            pixel_values = pixel_values.to(dtype=self.dtype) / 255.0
            mean = torch.tensor(
                self.config.vision_config.image_mean, dtype=self.dtype, device=pixel_values.device
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                self.config.vision_config.image_std, dtype=self.dtype, device=pixel_values.device
            ).view(1, 3, 1, 1)
            pixel_values = (pixel_values - mean) / std
        else:
            # Float path (backward-compatible): image was pre-normalized by the processor.
            pixel_values = pixel_values.to(dtype=self.dtype)
            # TODO: change
            # nb_values_per_image = pixel_values.shape[1:].numel()
            # real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            # pixel_values = pixel_values[real_images_inds].contiguous()

        # GotOcr2ImageProcessor crops tiles to exactly (patch_size * n_patches) so every
        # pixel is valid. Skip the pixel_attention_mask unfold+sum and pass None so the
        # vision encoder allocates a full-ones mask internally (recipe ⑩, Huang et al. 2026).
        # When DeepStack is enabled, ask the vision encoder for all per-layer hidden states
        # so we can tap intermediate ViT block outputs (pre-final-LayerNorm) at the configured
        # ``deepstack_visual_indexes``. The connector's ``Idefics3VisionTransformer.forward`` is
        # decorated with ``capture_outputs(tie_last_hidden_states=False)``, so ``hidden_states[i]``
        # is the output of encoder layer ``i`` *before* the post-LayerNorm — exactly the tap point
        # used in nanoVLM.
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
            # One projected tensor per ViT tap, shape [N_real_tiles, num_image_tokens, lm_dim].
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
    ) -> tuple | GraniteDoclingHybridBaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
            the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
            max_num_images is the maximum number of images among the batch_size samples in the batch.
            Padding images are not needed beyond padding the pixel_values at the entrance of the model.
            For efficiency, we only pass through the vision_model's forward the real images by
            discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
            image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        # START VISUAL INPUTS INTEGRATION
        deepstack_visual_embeds: list[torch.Tensor] | None = None
        visual_pos_masks: torch.Tensor | None = None
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, pixel_attention_mask)
            image_hidden_states = image_outputs.pooler_output
            # DeepStack: project the intermediate ViT features to LM dim and flatten across
            # the tile dim. The mergers' output is [N_tiles, num_image_tokens, lm_dim]; we
            # reshape to [N_tiles * num_image_tokens, lm_dim] so it lines up token-for-token
            # with ``visual_pos_masks`` (one position per <image> token in the LM sequence).
            ds_features = getattr(image_outputs, "deepstack_features", None)
            if ds_features is not None and input_ids is not None:
                lm_dim = self.config.text_config.hidden_size
                deepstack_visual_embeds = [feat.reshape(-1, lm_dim) for feat in ds_features]
                visual_pos_masks = input_ids == self.config.image_token_id
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        # DeepStack is restricted to prefill — see nanoVLM's ``_build_deepstack_embeds`` /
        # generate paths and the paper. During autoregressive decoding ``pixel_values`` is
        # None so we naturally pass None for the deepstack kwargs.
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

        return GraniteDoclingHybridBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The GraniteDoclingHybrid Model with a language modeling head. It is made up of a SigLIP vision encoder,
    with a GraniteMoeHybrid language model on top.
    """
)
class GraniteDoclingHybridForConditionalGeneration(Idefics3ForConditionalGeneration):
    config_class = GraniteDoclingHybridConfig

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
    ) -> tuple | GraniteDoclingHybridCausalLMOutputWithPast:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `GraniteDoclingHybridForConditionalGeneration`).
            Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
            computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
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

        return GraniteDoclingHybridCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        logits_to_keep=None,
        is_first_iteration=False,
        use_cache=False,
        **kwargs,
    ):
        # Overwritten to handle HybridMambaAttentionDynamicCache initialization

        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )

        # Initialize HybridMambaAttentionDynamicCache if needed
        if model_inputs.get("use_cache", True) and not isinstance(
            model_inputs.get("past_key_values"), HybridMambaAttentionDynamicCache
        ):
            cache_source = model_inputs.get("inputs_embeds")
            if cache_source is None:
                cache_source = model_inputs.get("decoder_inputs_embeds")
            if cache_source is not None:
                batch_size = cache_source.shape[0]
                dtype = cache_source.dtype
                device = cache_source.device
            else:
                input_tensor = model_inputs.get("input_ids")
                if input_tensor is None:
                    input_tensor = model_inputs.get("decoder_input_ids")
                if input_tensor is None:
                    input_tensor = input_ids
                if input_tensor is None:
                    raise ValueError("Unable to determine batch size for GraniteMoeHybrid cache initialization.")
                batch_size = input_tensor.shape[0]
                dtype = self.model.text_model.get_input_embeddings().weight.dtype
                device = input_tensor.device

            model_inputs["past_key_values"] = HybridMambaAttentionDynamicCache(
                self.model.text_model.config,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
            )

        if image_hidden_states is not None or (use_cache and not is_first_iteration):
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs


__all__ = [
    "GraniteDoclingHybridConfig",
    "GraniteDoclingHybridForConditionalGeneration",
    "GraniteDoclingHybridModel",
    "GraniteDoclingHybridPreTrainedModel",
]
