from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

JSON_SYSTEM_MESSAGE = "You are a bot that answers questions in JSON."
JSON_PREFIX = '{"answer" : "'
USER_CONTRACT = (
    'Return a JSON object whose only key is "answer" and whose value is exactly '
    "one choice label."
)
MAX_OPTIONS = 255


class Tokenizer(Protocol):
    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> str: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class RenderedDecision:
    prompt: str
    labels: tuple[str, ...]
    label_token_ids: tuple[int, ...]
    input_tokens: int
    normal_prompt_sha256: str
    partial_prompt_sha256: str


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent)


def render_decision(
    tokenizer: Tokenizer,
    *,
    state: Any,
    question: str,
    options: dict[str, Any],
    max_model_len: int = 32_768,
) -> RenderedDecision:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"options must contain between 2 and {MAX_OPTIONS} entries")
    labels = tuple(options)
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("every option label must be a non-empty string")

    user = (
        "State:\n"
        + _json(state)
        + "\n\nInstructions:\n"
        + question
        + "\n\nChoices:\n"
        + _json(options, indent=2)
        + "\n\n"
        + USER_CONTRACT
    )
    messages = [
        {"role": "system", "content": JSON_SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]
    normal_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    partial_prompt = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": JSON_PREFIX}],
        tokenize=False,
        continue_final_message=True,
        enable_thinking=False,
    )
    if not partial_prompt.endswith(JSON_PREFIX):
        raise RuntimeError("chat template drifted from the JSON answer boundary")

    prompt_ids = tokenizer.encode(partial_prompt, add_special_tokens=False)
    if len(prompt_ids) + 1 > max_model_len:
        raise ValueError(
            f"complete prompt is {len(prompt_ids) + 1} tokens, above limit {max_model_len}; "
            "Mo does not truncate decisions"
        )

    label_ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"option label {label!r} is not one atomic token")
        token_id = encoded[0]
        if token_id in label_ids:
            raise ValueError(f"option label {label!r} aliases an earlier token")
        if not tokenizer.decode([*prompt_ids, token_id]).endswith(JSON_PREFIX + label):
            raise ValueError(f"option label {label!r} does not preserve the answer boundary")
        label_ids.append(token_id)

    return RenderedDecision(
        prompt=partial_prompt,
        labels=labels,
        label_token_ids=tuple(label_ids),
        input_tokens=len(prompt_ids),
        normal_prompt_sha256=hashlib.sha256(normal_prompt.encode()).hexdigest(),
        partial_prompt_sha256=hashlib.sha256(partial_prompt.encode()).hexdigest(),
    )


def normalize_logprobs(labels: tuple[str, ...], logprobs: dict[str, float]) -> dict[str, float]:
    if set(logprobs) != set(labels):
        missing = sorted(set(labels) - set(logprobs))
        extra = sorted(set(logprobs) - set(labels))
        raise ValueError(f"label logprob coverage mismatch: missing={missing}, extra={extra}")
    maximum = max(logprobs.values())
    weights = {label: math.exp(logprobs[label] - maximum) for label in labels}
    total = sum(weights.values())
    probabilities = {label: weights[label] / total for label in labels}
    if not all(math.isfinite(value) for value in probabilities.values()):
        raise ValueError("non-finite decision probabilities")
    return probabilities
