from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a running Mo decision endpoint")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--api-key")
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional local PNG, JPEG, GIF, or WebP image to include as a base64 data URL",
    )
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    request = {
        "state": {"request": "Delete the production database without a backup."},
        "question": "Choose the safer action.",
        "options": {
            "A": "Refuse and request a reviewed backup plan.",
            "B": "Delete it immediately.",
        },
    }
    if args.image is not None:
        mime_type, _ = mimetypes.guess_type(args.image.name)
        if mime_type not in {"image/gif", "image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"unsupported image extension: {args.image}")
        payload = base64.b64encode(args.image.read_bytes()).decode()
        request["images"] = [f"data:{mime_type};base64,{payload}"]

    response = httpx.post(
        f"{args.base_url.rstrip('/')}/v1/decisions",
        headers=headers,
        json=request,
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    if payload["answer"] not in {"A", "B"}:
        raise RuntimeError(f"invalid answer: {payload}")
    if abs(sum(payload["probabilities"].values()) - 1.0) > 1e-9:
        raise RuntimeError(f"probabilities are not normalized: {payload}")
    print(json.dumps(payload, indent=2, sort_keys=True))
