---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
library_name: transformers
pipeline_tag: text-generation
tags:
  - vllm
  - fp8
  - compressed-tensors
  - decision-model
  - qwen
---

# Mo

Mo is a merged, specialized decision model derived from
`Qwen/Qwen3.6-35B-A3B`. It accepts structured state, a question, and labeled
choices through a thinking-disabled JSON chat template, then uses normalized
next-label probabilities to select an answer.

This is a standalone compressed-tensors FP8 checkpoint. It does not require a
PEFT adapter at load time.

## Serving

The tested runtime is vLLM 0.28.0 on one H100 80 GB:

```bash
vllm serve modaic/mo \
  --served-model-name mo \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --generation-config vllm \
  --language-model-only \
  --max-logprobs 255 \
  --logprobs-mode raw_logprobs
```

The exact client renderer must:

1. Use the tokenizer's native chat template with `enable_thinking=false`.
2. Use system message `You are a bot that answers questions in JSON.`
3. Format the user message as `State`, `Instructions`, and `Choices`.
4. Continue an assistant message ending exactly at `{"answer" : "`.
5. Restrict the next token to the supplied atomic choice labels.
6. Normalize raw next-token log probabilities only over those labels.

The companion serving implementation is intended to live in the GitHub
repository named `mo`.

## Evaluation

| Benchmark | FP8 | BF16 | Delta |
|---|---:|---:|---:|
| Public JevBench accuracy | 87.88% | 88.31% | -0.43 pp |
| JevBench correctness AUROC | 0.9206 | 0.9128 | +0.0078 |
| Decision Index balanced raw | 55.94 | 56.07 | -0.13 |
| Decision Index balanced skill | 41.43 | 41.55 | -0.12 |
| Decision Index breadth skill | 39.80 | 39.99 | -0.19 |

Decision Index coverage was 37 catalogs / 131,980 requests / 764,263 fields,
with a maximum complete prompt length of 22,718 tokens and no truncation.

## Provenance

- Base revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Adapter checkpoint: 3,450
- Adapter SHA-256: `19947c06e73761754d80aeebd5bd87a455806c34dbe653359aa3105cd02d95b2`
- Renderer contract SHA-256: `4a584a39e17e10f063f904398d71631c0d1615d9a5b0b45df6e2887a335588e6`
- FP8 artifact bytes: 37,702,513,480
- Indexed weight bytes: 37,667,035,872
- Shards: 38
- Quantization: compressed-tensors FP8, dynamic per-token activations and
  per-channel weights

See `provenance.json` in this repository for machine-readable identities.

## Limitations

Mo is designed for categorical decisions, not unconstrained chat. Quality
depends on clearly specified state, instructions, and choices. Choice labels
must each be one atomic token under the bundled tokenizer. FP8 changes some
decisions: full Decision Index field-level agreement with BF16 was 97.91%.
