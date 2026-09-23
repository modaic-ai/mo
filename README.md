# Mo

Mo is a specialized decision model built from `Qwen/Qwen3.6-35B-A3B`. This
repository contains the exact vLLM 0.28 serving recipe used to validate the
merged, compressed-tensors FP8 checkpoint on one 80 GB H100.

Mo does not answer with free-form prose. It receives structured state, one
question, and labeled choices; it scores the next-token distribution at the
JSON answer slot and returns both the selected label and normalized
probabilities.

## Validated release

- Model repository: `modaic/mo-1.1-fp8`
- Base revision: `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`
- Adapter checkpoint: step 3,450
- Adapter SHA-256: `19947c06e73761754d80aeebd5bd87a455806c34dbe653359aa3105cd02d95b2`
- Artifact: merged compressed-tensors FP8, 38 safetensors shards, 35.11 GiB
- vLLM: `0.28.0`
- Transformers: `5.14.1`
- Context: 32,768 tokens, with no truncation
- Hardware: one H100 80 GB (`tensor_parallel_size=1`)

The model is already merged. Its index contains 35,138,874,736 parameters and
zero LoRA/adapter keys. The merge gate measured zero logit drift between the
live adapter and merged checkpoint before FP8 compression.

## Start the endpoint

Requirements: Docker with the NVIDIA Container Toolkit, one Hopper GPU, and a
Hugging Face token that can read the model repository.

```bash
cp .env.example .env
# Set HF_TOKEN and pin MODEL_REVISION to the immutable HF commit after upload.
docker compose up --build
```

This starts:

- `http://localhost:8000`: the raw OpenAI-compatible vLLM server.
- `http://localhost:8080/v1/decide`: the decision gateway.

The vLLM process uses the measured production configuration:

```text
vllm serve modaic/mo-1.1-fp8
  --served-model-name mo
  --tensor-parallel-size 1
  --max-model-len 32768
  --max-num-batched-tokens 65536
  --max-num-seqs 128
  --gpu-memory-utilization 0.90
  --enable-prefix-caching
  --generation-config vllm
  --language-model-only
  --max-logprobs 255
  --logprobs-mode raw_logprobs
```

Do not add a reasoning parser. The gateway renders the native Qwen chat
template with `enable_thinking=false`, then continues the assistant message at
the exact JSON boundary `{"answer" : "`. This is the interface used during
training and evaluation.

## Make a decision

```bash
curl -sS http://localhost:8080/v1/decide \
  -H 'content-type: application/json' \
  --data @examples/decision.json
```

Example response:

```json
{
  "answer": "A",
  "output": {"answer": "A"},
  "probabilities": {"A": 0.998, "B": 0.002},
  "logprobs": {"A": -0.01, "B": -6.22},
  "input_tokens": 143,
  "normal_prompt_sha256": "...",
  "partial_prompt_sha256": "...",
  "usage": {
    "prompt_tokens": 143,
    "completion_tokens": 1,
    "total_tokens": 144,
    "engine_requests": 1
  }
}
```

Option labels must be distinct, atomic tokenizer tokens. Requests with
non-atomic labels, more than 255 choices, non-finite JSON values, or prompts
above 32,768 tokens fail closed. The gateway never truncates a decision. vLLM
accepts at most 128 explicit `logprob_token_ids` per request, so option sets of
129–255 labels are replayed in chunks while preserving the full allowed-token
constraint. The response reports the actual number of engine calls in
`usage.engine_requests`.

For an integration check after startup:

```bash
uv run mo-smoke --base-url http://localhost:8080
```

## Accuracy and throughput

All measurements use the exact JSON decision renderer above.

| Benchmark | FP8/H100 | BF16 checkpoint | Delta |
|---|---:|---:|---:|
| Public JevBench accuracy | 87.88% | 88.31% | -0.43 pp |
| JevBench correctness AUROC | 0.9206 | 0.9128 | +0.0078 |
| Decision Index balanced raw | 55.94 | 56.07 | -0.13 |
| Decision Index balanced skill | 41.43 | 41.55 | -0.12 |
| Decision Index breadth skill | 39.80 | 39.99 | -0.19 |

The full Decision Index covered 37 catalogs, 131,980 requests, and 764,263
fields. FP8 retained 97.91% field-level prediction agreement with BF16.

| Runtime | Input tok/GPU-s | Decisions/GPU-s | GPU $/M input tok |
|---|---:|---:|---:|
| FP8 on H100 | 51,769.7 | 120.50 | $0.02119 |
| Historical BF16 on H200 | 18,339.3 | 42.69 | $0.06876 |

Costs use the public GPU rates at evaluation time and exclude cold starts,
CPU, and memory. They describe batched one-token decisions, not free-form
generation.

## Publish the model

The standalone artifact currently lives in Modal volume
`qwen36-35b-json-decider-fp8` at
`/qwen36-35b-json-decider-ckpt3450-fp8-dynamic`. The uploader revalidates the
sealed metadata before transferring any weights.

Create a private HF repository and upload:

```bash
uv run --extra publish modal run publish_hf.py \
  --repo-id modaic/mo-1.1-fp8 \
  --create \
  --private
```

If the repository already exists, omit `--create`. Deliberately pass
`--no-private` when creating a public repository. After upload, pin
`MODEL_REVISION` in `.env` to the returned immutable commit rather than `main`.

## Development

```bash
uv sync --extra dev
make check
```
