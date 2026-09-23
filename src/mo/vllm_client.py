from __future__ import annotations

from typing import Any

import httpx

from mo.renderer import RenderedDecision, normalize_logprobs

MAX_LOGPROB_TOKEN_IDS = 128


def _token_id_chunks(token_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        token_ids[start : start + MAX_LOGPROB_TOKEN_IDS]
        for start in range(0, len(token_ids), MAX_LOGPROB_TOKEN_IDS)
    )


def _label_logprobs(
    top_logprobs: dict[str, float],
    rendered: RenderedDecision,
    requested_token_ids: tuple[int, ...],
) -> dict[str, float]:
    requested = set(requested_token_ids)
    result: dict[str, float] = {}
    for label, token_id in zip(rendered.labels, rendered.label_token_ids, strict=True):
        if token_id not in requested:
            continue
        candidates = (label, f"token_id:{token_id}")
        matches = [key for key in candidates if key in top_logprobs]
        if len(matches) != 1:
            raise RuntimeError(
                f"vLLM did not return exactly one raw logprob for label {label!r} "
                f"(token {token_id})"
            )
        result[label] = float(top_logprobs[matches[0]])
    return result


async def score_decision(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str | None,
    served_model_name: str,
    rendered: RenderedDecision,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    chunks = _token_id_chunks(rendered.label_token_ids)
    raw: dict[str, float] = {}
    sampled_token_ids: set[int] = set()
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for requested_token_ids in chunks:
        response = await client.post(
            f"{base_url.rstrip('/')}/v1/completions",
            headers=headers,
            json={
                "model": served_model_name,
                "prompt": rendered.prompt,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 0,
                "logprob_token_ids": list(requested_token_ids),
                "allowed_token_ids": list(rendered.label_token_ids),
                "return_tokens_as_token_ids": True,
            },
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("vLLM returned an unexpected completion count")
        logprobs = choices[0].get("logprobs")
        if not isinstance(logprobs, dict):
            raise RuntimeError("vLLM omitted completion logprobs")
        rows = logprobs.get("top_logprobs")
        tokens = logprobs.get("tokens")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise RuntimeError("vLLM returned an unexpected top_logprobs shape")
        if not isinstance(tokens, list) or len(tokens) != 1:
            raise RuntimeError("vLLM returned an unexpected sampled-token shape")
        token_key = tokens[0]
        if not isinstance(token_key, str) or not token_key.startswith("token_id:"):
            raise RuntimeError("vLLM did not return the sampled token as a token ID")
        sampled_token_ids.add(int(token_key.removeprefix("token_id:")))
        raw.update(_label_logprobs(rows[0], rendered, requested_token_ids))

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("vLLM omitted completion usage")
        for field in usage_totals:
            value = usage.get(field)
            if not isinstance(value, int):
                raise RuntimeError(f"vLLM returned invalid {field}")
            usage_totals[field] += value

    if len(sampled_token_ids) != 1:
        raise RuntimeError("chunked vLLM requests sampled different answers")
    sampled_token_id = next(iter(sampled_token_ids))
    if sampled_token_id not in rendered.label_token_ids:
        raise RuntimeError("vLLM sampled a token outside the allowed labels")
    if set(raw) != set(rendered.labels):
        raise RuntimeError("vLLM did not return every requested label logprob")
    probabilities = normalize_logprobs(rendered.labels, raw)
    answer = max(rendered.labels, key=probabilities.__getitem__)
    expected_token_id = rendered.label_token_ids[rendered.labels.index(answer)]
    if sampled_token_id != expected_token_id:
        raise RuntimeError("sampled answer disagrees with the maximum raw logprob")
    return {
        "answer": answer,
        "probabilities": probabilities,
        "logprobs": raw,
        "usage": {**usage_totals, "engine_requests": len(chunks)},
    }
