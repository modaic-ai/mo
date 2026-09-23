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


def _chat_logprob_row(logprobs: Any) -> tuple[str, dict[str, float]]:
    if not isinstance(logprobs, dict):
        raise RuntimeError("vLLM omitted chat completion logprobs")
    content = logprobs.get("content")
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
        raise RuntimeError("vLLM returned an unexpected chat logprobs shape")
    row = content[0]
    sampled = row.get("token")
    entries = row.get("top_logprobs")
    if not isinstance(sampled, str) or not isinstance(entries, list):
        raise RuntimeError("vLLM returned malformed chat token logprobs")
    result: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("vLLM returned a malformed top-logprob entry")
        token = entry.get("token")
        logprob = entry.get("logprob")
        if not isinstance(token, str) or not isinstance(logprob, (int, float)):
            raise RuntimeError("vLLM returned a malformed top-logprob value")
        if token in result:
            raise RuntimeError(f"vLLM returned duplicate logprob token {token!r}")
        result[token] = float(logprob)
    return sampled, result


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
    prompt_token_counts: set[int] = set()
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for requested_token_ids in chunks:
        response = await client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json={
                "model": served_model_name,
                "messages": list(rendered.messages),
                "temperature": 0,
                "max_completion_tokens": 1,
                "logprobs": True,
                "top_logprobs": 0,
                "logprob_token_ids": list(requested_token_ids),
                "allowed_token_ids": list(rendered.label_token_ids),
                "return_tokens_as_token_ids": True,
                "add_generation_prompt": False,
                "continue_final_message": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("vLLM returned an unexpected completion count")
        token_key, top_logprobs = _chat_logprob_row(choices[0].get("logprobs"))
        if not token_key.startswith("token_id:"):
            raise RuntimeError("vLLM did not return the sampled token as a token ID")
        sampled_token_ids.add(int(token_key.removeprefix("token_id:")))
        raw.update(_label_logprobs(top_logprobs, rendered, requested_token_ids))

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("vLLM omitted completion usage")
        for field in usage_totals:
            value = usage.get(field)
            if not isinstance(value, int):
                raise RuntimeError(f"vLLM returned invalid {field}")
            usage_totals[field] += value
        prompt_token_counts.add(usage["prompt_tokens"])

    if len(sampled_token_ids) != 1:
        raise RuntimeError("chunked vLLM requests sampled different answers")
    sampled_token_id = next(iter(sampled_token_ids))
    if sampled_token_id not in rendered.label_token_ids:
        raise RuntimeError("vLLM sampled a token outside the allowed labels")
    if set(raw) != set(rendered.labels):
        raise RuntimeError("vLLM did not return every requested label logprob")
    if len(prompt_token_counts) != 1:
        raise RuntimeError("chunked vLLM requests reported different prompt lengths")
    probabilities = normalize_logprobs(rendered.labels, raw)
    answer = max(rendered.labels, key=probabilities.__getitem__)
    expected_token_id = rendered.label_token_ids[rendered.labels.index(answer)]
    if sampled_token_id != expected_token_id:
        raise RuntimeError("sampled answer disagrees with the maximum raw logprob")
    return {
        "answer": answer,
        "probabilities": probabilities,
        "logprobs": raw,
        "input_tokens": next(iter(prompt_token_counts)),
        "usage": {**usage_totals, "engine_requests": len(chunks)},
    }
