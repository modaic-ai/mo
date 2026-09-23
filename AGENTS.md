# Mo repository notes

- Preserve the exact JSON decision renderer in `src/mo/renderer.py`.
- Never enable Qwen thinking for decision scoring.
- Never truncate a state, question, choices, or rendered prompt.
- Choice labels must be unique atomic tokens and preserve the continued-message boundary.
- Keep vLLM and Transformers pinned to the benchmarked versions unless a full accuracy and throughput regression run passes.
- Do not publish weights unless `publish_hf.py` passes every artifact identity check.
- Pin deployments to an immutable Hugging Face commit after publication.
