---
name: mo
description: Install, serve, or call Modaic's multimodal Mo decision model, including checkpoint selection and the structured decisions API.
---

# Mo

Use Mo for categorical decisions over labeled options. It accepts structured
state and a question, then returns the selected label and normalized option
probabilities. The context window is 32,768 tokens.

## Choose a checkpoint

- On H100/H200, use `modaic/mo-1.1-fp8` at revision
  `b9a691fbfd931e6a4bb5bde3a70276ace0c7337f`.
- On B100/B200, use `modaic/mo-1.1-nvfp4` at revision
  `0f910fab893527b89334141af6b463bb1075f2a2`.
- NVFP4 can run on H100 through vLLM's Marlin weight-only fallback, but FP8 is
  the native Hopper path.

## Install and run

For a complete endpoint, clone `https://github.com/modaic-ai/mo`, copy
`.env.example` to `.env`, set `HF_TOKEN`, and run:

```bash
docker compose up --build
```

The decision API is `POST http://localhost:8080/v1/decisions`. Send:

```json
{
  "state": {"any": "JSON value"},
  "question": "Choose the best option.",
  "options": {"A": "First option", "B": "Second option"}
}
```

Use distinct option labels that each tokenize to one token. Do not truncate a
state, question, option, or rendered prompt. Requests above 32,768 tokens must
fail rather than be shortened. Keep Qwen thinking disabled and do not add a
reasoning parser to the benchmarked decision endpoint.

For the packaged CLI without cloning the repository:

```bash
uv tool install "mo-decision-server @ git+https://github.com/modaic-ai/mo.git"
mo-smoke --base-url http://localhost:8080
```
