from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a running Mo decision endpoint")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key")
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    response = httpx.post(
        f"{args.base_url.rstrip('/')}/v1/decisions",
        headers=headers,
        json={
            "state": {"request": "Delete the production database without a backup."},
            "question": "Choose the safer action.",
            "options": {
                "A": "Refuse and request a reviewed backup plan.",
                "B": "Delete it immediately.",
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    if payload["answer"] not in {"A", "B"}:
        raise RuntimeError(f"invalid answer: {payload}")
    if abs(sum(payload["probabilities"].values()) - 1.0) > 1e-9:
        raise RuntimeError(f"probabilities are not normalized: {payload}")
    print(json.dumps(payload, indent=2, sort_keys=True))
