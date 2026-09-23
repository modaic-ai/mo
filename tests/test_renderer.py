from __future__ import annotations

import asyncio
import base64
import hashlib
import math

import pytest

from mo.renderer import JSON_PREFIX, normalize_logprobs, render_decision
from mo.vllm_client import (
    MAX_LOGPROB_TOKEN_IDS,
    _chat_logprob_row,
    _label_logprobs,
    _token_id_chunks,
    score_decision,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\nunit-test"
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


class FakeTokenizer:
    labels = {"A": 10, "B": 11, "wide": 12}

    def apply_chat_template(self, conversation, **kwargs):
        def content_text(content):
            if isinstance(content, str):
                return content
            return "".join(
                part.get("text", "<image>") if isinstance(part, dict) else str(part)
                for part in content
            )

        text = "<chat>" + "".join(content_text(item["content"]) for item in conversation)
        if kwargs.get("continue_final_message"):
            return text
        return text + "<assistant>"

    def encode(self, text, *, add_special_tokens):
        if text in self.labels:
            return [self.labels[text]]
        return list(range(len(text.split()) + 3))

    def decode(self, token_ids):
        label = next((key for key, value in self.labels.items() if value == token_ids[-1]), "?")
        return JSON_PREFIX + label


def test_exact_json_boundary_and_atomic_labels():
    rendered = render_decision(
        FakeTokenizer(),
        state={"risk": "high"},
        question="Choose.",
        options={"A": "stop", "B": "continue"},
    )
    assert rendered.prompt.endswith(JSON_PREFIX)
    assert rendered.labels == ("A", "B")
    assert rendered.label_token_ids == (10, 11)


def test_multimodal_messages_and_image_identity():
    rendered = render_decision(
        FakeTokenizer(),
        state={"instruction": "inspect the image"},
        question="Choose.",
        options={"A": "red", "B": "blue"},
        image_data_urls=(PNG_DATA_URL,),
    )
    user_content = rendered.messages[1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0] == {"type": "image_url", "image_url": {"url": PNG_DATA_URL}}
    assert user_content[-1]["type"] == "text"
    assert rendered.messages[-1] == {"role": "assistant", "content": JSON_PREFIX}
    assert rendered.image_sha256 == (hashlib.sha256(PNG_BYTES).hexdigest(),)
    assert len(rendered.request_sha256) == 64


def test_refuses_remote_or_spoofed_image():
    kwargs = {
        "state": {},
        "question": "Choose.",
        "options": {"A": "red", "B": "blue"},
    }
    with pytest.raises(ValueError, match="base64 data URLs"):
        render_decision(
            FakeTokenizer(),
            image_data_urls=("https://example.com/image.png",),
            **kwargs,
        )
    spoofed = "data:image/jpeg;base64," + base64.b64encode(PNG_BYTES).decode()
    with pytest.raises(ValueError, match="do not match"):
        render_decision(FakeTokenizer(), image_data_urls=(spoofed,), **kwargs)


def test_refuses_non_atomic_label():
    with pytest.raises(ValueError, match="not one atomic token"):
        render_decision(
            FakeTokenizer(),
            state={},
            question="Choose.",
            options={"A": "stop", "not atomic": "continue"},
        )


def test_refuses_truncation():
    with pytest.raises(ValueError, match="does not truncate"):
        render_decision(
            FakeTokenizer(),
            state={"large": "state"},
            question="Choose.",
            options={"A": "stop", "B": "continue"},
            max_model_len=2,
        )


def test_raw_logprobs_are_normalized_over_choices():
    probabilities = normalize_logprobs(("A", "B"), {"A": -1.0, "B": -2.0})
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert probabilities["A"] > probabilities["B"]


def test_token_id_logprob_keys_are_supported():
    rendered = render_decision(
        FakeTokenizer(),
        state={},
        question="Choose.",
        options={"A": "stop", "B": "continue"},
    )
    assert _label_logprobs(
        {"token_id:10": -1.0, "token_id:11": -2.0}, rendered, (10, 11)
    ) == {
        "A": -1.0,
        "B": -2.0,
    }


def test_logprob_token_ids_are_chunked_at_vllm_limit():
    token_ids = tuple(range(MAX_LOGPROB_TOKEN_IDS * 2 + 1))
    chunks = _token_id_chunks(token_ids)
    assert tuple(map(len, chunks)) == (128, 128, 1)
    assert tuple(token_id for chunk in chunks for token_id in chunk) == token_ids


def test_chat_logprob_shape_and_multimodal_request():
    sampled, raw = _chat_logprob_row(
        {
            "content": [
                {
                    "token": "token_id:10",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "token_id:10", "logprob": -0.1},
                        {"token": "token_id:11", "logprob": -2.0},
                    ],
                }
            ]
        }
    )
    assert sampled == "token_id:10"
    assert raw == {"token_id:10": -0.1, "token_id:11": -2.0}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": "token_id:10",
                                    "logprob": -0.1,
                                    "top_logprobs": [
                                        {"token": "token_id:10", "logprob": -0.1},
                                        {"token": "token_id:11", "logprob": -2.0},
                                    ],
                                }
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 512, "completion_tokens": 1, "total_tokens": 513},
            }

    class FakeClient:
        def __init__(self):
            self.url = None
            self.body = None

        async def post(self, url, *, headers, json):
            self.url = url
            self.body = json
            return FakeResponse()

    rendered = render_decision(
        FakeTokenizer(),
        state={},
        question="Choose.",
        options={"A": "red", "B": "blue"},
        image_data_urls=(PNG_DATA_URL,),
    )
    client = FakeClient()
    result = asyncio.run(
        score_decision(
            client,
            base_url="http://vllm:8000",
            api_key=None,
            served_model_name="mo",
            rendered=rendered,
        )
    )
    assert client.url == "http://vllm:8000/v1/chat/completions"
    assert client.body["messages"][1]["content"][0]["type"] == "image_url"
    assert client.body["continue_final_message"] is True
    assert client.body["add_generation_prompt"] is False
    assert client.body["chat_template_kwargs"] == {"enable_thinking": False}
    assert client.body["allowed_token_ids"] == [10, 11]
    assert result["answer"] == "A"
    assert result["input_tokens"] == 512
