from __future__ import annotations

import base64
import binascii
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
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {"image/gif", "image/jpeg", "image/png", "image/webp"}
)


class Tokenizer(Protocol):
    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> str: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class RenderedDecision:
    prompt: str
    messages: tuple[dict[str, Any], ...]
    labels: tuple[str, ...]
    label_token_ids: tuple[int, ...]
    input_tokens: int
    normal_prompt_sha256: str
    partial_prompt_sha256: str
    image_sha256: tuple[str, ...]
    request_sha256: str


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent)


def _image_bytes(data_url: str, *, max_image_bytes: int) -> tuple[bytes, str]:
    if not isinstance(data_url, str):
        raise ValueError("every image must be a base64 data URL string")
    header, separator, payload = data_url.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise ValueError("images must use base64 data URLs")
    mime_type = header.removeprefix("data:").removesuffix(";base64").lower()
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ValueError(f"unsupported image MIME type {mime_type!r}")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("image data URL contains invalid base64") from error
    if not decoded:
        raise ValueError("image data URL is empty")
    if len(decoded) > max_image_bytes:
        raise ValueError(
            f"decoded image is {len(decoded)} bytes, above limit {max_image_bytes}"
        )

    signatures = {
        "image/gif": decoded.startswith((b"GIF87a", b"GIF89a")),
        "image/jpeg": decoded.startswith(b"\xff\xd8\xff"),
        "image/png": decoded.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/webp": decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP",
    }
    if not signatures[mime_type]:
        raise ValueError(f"image bytes do not match declared MIME type {mime_type!r}")
    return decoded, mime_type


def render_decision(
    tokenizer: Tokenizer,
    *,
    state: Any,
    question: str,
    options: dict[str, Any],
    image_data_urls: tuple[str, ...] = (),
    max_model_len: int = 32_768,
    max_images: int = MAX_IMAGES,
    max_image_bytes: int = MAX_IMAGE_BYTES,
) -> RenderedDecision:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"options must contain between 2 and {MAX_OPTIONS} entries")
    labels = tuple(options)
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("every option label must be a non-empty string")
    if len(image_data_urls) > max_images:
        raise ValueError(f"images contains {len(image_data_urls)} items, above limit {max_images}")

    image_sha256: list[str] = []
    for data_url in image_data_urls:
        decoded, _ = _image_bytes(data_url, max_image_bytes=max_image_bytes)
        image_sha256.append(hashlib.sha256(decoded).hexdigest())

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
    text_messages = [
        {"role": "system", "content": JSON_SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]
    if image_data_urls:
        user_content: str | list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_url}}
            for data_url in image_data_urls
        ]
        user_content.append({"type": "text", "text": user})
    else:
        user_content = user
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": JSON_SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": JSON_PREFIX},
    ]
    normal_prompt = tokenizer.apply_chat_template(
        text_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    partial_prompt = tokenizer.apply_chat_template(
        [*text_messages, {"role": "assistant", "content": JSON_PREFIX}],
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

    request_identity = {
        "messages": messages,
        "image_sha256": image_sha256,
        "enable_thinking": False,
        "continue_final_message": True,
    }
    return RenderedDecision(
        prompt=partial_prompt,
        messages=tuple(messages),
        labels=labels,
        label_token_ids=tuple(label_ids),
        input_tokens=len(prompt_ids),
        normal_prompt_sha256=hashlib.sha256(normal_prompt.encode()).hexdigest(),
        partial_prompt_sha256=hashlib.sha256(partial_prompt.encode()).hexdigest(),
        image_sha256=tuple(image_sha256),
        request_sha256=hashlib.sha256(
            _json(request_identity).encode()
        ).hexdigest(),
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
