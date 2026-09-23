from __future__ import annotations

import math

import pytest

from mo.renderer import JSON_PREFIX, normalize_logprobs, render_decision
from mo.vllm_client import MAX_LOGPROB_TOKEN_IDS, _label_logprobs, _token_id_chunks


class FakeTokenizer:
    labels = {"A": 10, "B": 11, "wide": 12}

    def apply_chat_template(self, conversation, **kwargs):
        text = "<chat>" + "".join(item["content"] for item in conversation)
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
