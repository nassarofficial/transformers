<!--Copyright 2026 IBM and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-09-08.*

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# GraniteForDocling

[GraniteForDocling](https://huggingface.co/docling-project) is a vision-language model for document conversion. Given a page image, it generates DocLang (`<doclang>`): the layout elements in reading order with their bounding boxes, the text, tables, formulas and code they contain.

The page is split into tiles of 512x512 pixels laid out on the grid that best matches its aspect ratio, plus a thumbnail of the whole page when there is more than one tile. A vision encoder embeds every tile, and a pixel-shuffle connector maps vision patches to image tokens. Intermediate vision encoder states are projected as well and added to the image tokens after the first decoder layers. A dense Granite-style text decoder then generates the DocLang.

The same modeling code loads different GraniteForDocling sizes and configurations through `text_config` and `vision_config`. Checkpoints also select a projector: a high-resolution connector that emits more image tokens per tile, or a density router (`density_router_hidden_size`) that predicts from the vision encoder features whether a page needs that fine path (`model.predict_fine_route(pixel_values)`). The fine path can also be selected per request with `fine_route=True` in the processor call. Multi-token prediction heads (`num_mtp_layers`) add an auxiliary loss during training and serve as draft heads for speculative decoding.

> [!TIP]
> This model was contributed by the [Docling team](https://huggingface.co/docling-project).

## Usage tips

- Prompt the model with `<doclang>` through the chat template to convert a full page.
- Set `padding_side="left"` during batched generation for more accurate results.

```py
processor.tokenizer.padding_side = "left"
```

- Pass `fine_route=True` to the processor for dense pages (small print, long tables, multi-column text). The high-resolution projector then uses more image tokens per tile, so the prompt gets longer and inference slower.

The example below demonstrates how to convert a page with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">

<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(
    task="image-text-to-text",
    model="docling-project/granite-for-docling-500m",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/one_column.png"},
            {"type": "text", "text": "<doclang>"},
        ],
    }
]
pipe(text=messages, max_new_tokens=1024, return_full_text=False)
```

</hfoption>

<hfoption id="AutoModel">

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "docling-project/granite-for-docling-500m"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/one_column.png"},
            {"type": "text", "text": "<doclang>"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

</hfoption>

</hfoptions>

## GraniteForDoclingConfig

[[autodoc]] GraniteForDoclingConfig

## GraniteForDoclingTextConfig

[[autodoc]] GraniteForDoclingTextConfig

## GraniteForDoclingVisionConfig

[[autodoc]] GraniteForDoclingVisionConfig

## GraniteForDoclingImageProcessor

[[autodoc]] GraniteForDoclingImageProcessor
    - preprocess

## GraniteForDoclingProcessor

[[autodoc]] GraniteForDoclingProcessor
    - __call__

## GraniteForDoclingModel

[[autodoc]] GraniteForDoclingModel
    - forward
    - get_image_features

## GraniteForDoclingTextModel

[[autodoc]] GraniteForDoclingTextModel
    - forward

## GraniteForDoclingForConditionalGeneration

[[autodoc]] GraniteForDoclingForConditionalGeneration
    - forward
    - get_image_features
