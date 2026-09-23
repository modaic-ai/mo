---
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - vllm
  - multimodal
  - fp8
  - decision-model
---

# Mo

```bash
npx skills add modaic-ai/mo --skill mo
```

Mo is a multimodal, Jev-like decision model with a 32,768-token context
window. It uses text and images to choose between labeled options and returns
both the selected answer and normalized probabilities.

## Benchmarks

| Checkpoint | [Public JevBench](https://github.com/fstandhartinger/jevbench) | [Decision Index raw](https://huggingface.co/spaces/multimodalart/jev-decision-index) | Balanced skill | Breadth skill |
|---|---:|---:|---:|---:|
| [FP8](https://huggingface.co/modaic/mo-1.1-fp8) | 203/231 (87.88%) | 55.94 | 41.43 | 39.80 |
| [NVFP4](https://huggingface.co/modaic/mo-1.1-nvfp4) | 204/231 (88.31%) | 55.22 | 40.40 | 38.86 |
| BF16 reference | 204/231 (88.31%) | 56.07 | 41.55 | 39.99 |

Decision Index covers 37 catalogs, 131,980 requests, and 764,263 decision
fields.

## Checkpoints

| Checkpoint | Best hardware | Immutable revision |
|---|---|---|
| [`modaic/mo-1.1-fp8`](https://huggingface.co/modaic/mo-1.1-fp8) | H100/H200 | `60cd2356f6f4e9ec83967aa4f484c26a81f08b88` |
| [`modaic/mo-1.1-nvfp4`](https://huggingface.co/modaic/mo-1.1-nvfp4) | B100/B200 | `0f910fab893527b89334141af6b463bb1075f2a2` |

Use FP8 on an H100. NVFP4 also loads on H100, but vLLM uses its Marlin
weight-only compatibility path instead of native NVFP4 W4A4 compute.

## Install

Mo is an installable Python package named `mo-decision-server`. Install its
CLI directly from GitHub:

```bash
uv tool install "mo-decision-server @ git+https://github.com/modaic-ai/mo.git"
```

For the Docker deployment and source code:

```bash
git clone https://github.com/modaic-ai/mo.git
cd mo
cp .env.example .env
```

Set `HF_TOKEN` in `.env`. The defaults select the FP8 checkpoint for one 80 GB
H100. To use NVFP4, also set:

```dotenv
MODEL_ID=modaic/mo-1.1-nvfp4
MODEL_REVISION=0f910fab893527b89334141af6b463bb1075f2a2
```

## Use

Start vLLM and the Mo decision API:

```bash
docker compose up --build
```

This exposes vLLM at `http://localhost:8000` and the decision API at
`http://localhost:8080/v1/decisions`.

```bash
curl -sS http://localhost:8080/v1/decisions \
  -H 'content-type: application/json' \
  --data @examples/decision.json
```

Request:

```json
{
  "state": {
    "request": "Delete the production database without a backup."
  },
  "question": "Choose the safer action.",
  "options": {
    "A": "Refuse and request a reviewed backup plan.",
    "B": "Delete it immediately."
  }
}
```

For a multimodal decision, add up to four images as base64 data URLs. Remote
URLs are intentionally rejected so the deployment never fetches untrusted
network resources:

```json
{
  "state": {"task": "Inspect the attached image."},
  "question": "Which color fills the square?",
  "options": {"A": "red", "B": "blue"},
  "images": ["data:image/png;base64,iVBORw0KGgo..."]
}
```

Supported image types are PNG, JPEG, GIF, and WebP. Each decoded image may be
at most 20 MiB. The gateway sends the structured conversation to vLLM's
`/v1/chat/completions` endpoint with thinking disabled, continues the exact
JSON prefix, and constrains generation to one option-label token.

The pinned FP8 revision was validated end to end on one H100 80 GB: a red
square selected `A` with 99.933% probability, a blue square selected `B` with
99.914%, and the text-only regression request remained correct. H200 is not
required for this serving configuration.

Response:

```json
{
  "answer": "A",
  "output": {"answer": "A"},
  "probabilities": {"A": 0.998, "B": 0.002},
  "logprobs": {"A": -0.01, "B": -6.22},
  "input_tokens": 143
}
```

Run the installed smoke client against a live endpoint:

```bash
mo-smoke --base-url http://localhost:8080
mo-smoke --base-url http://localhost:8080 --image ./example.png
```
