from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "modaic/mo-1.1-fp8"
SERVED_MODEL_NAME = "mo"
VLLM_PORT = 8000
GATEWAY_PORT = 8080

app = modal.App("mo-multimodal-h100-smoke")
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.28.0")
    .entrypoint([])
    .run_commands('ln -s "$(command -v python3)" /usr/local/bin/python')
    .pip_install(
        "flashinfer-python",
        "httpx>=0.28,<1",
        "pillow>=11,<12",
        "pydantic-settings>=2.10,<3",
        "transformers==5.14.1",
    )
    .env({"HF_HOME": "/tmp/huggingface", "PYTHONPATH": "/opt/mo"})
    .add_local_dir("src/mo", "/opt/mo/mo", copy=True)
)


def _wait_for(url: str, *, timeout: float, process: subprocess.Popen[Any]) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            response = httpx.get(url, timeout=5)
            response.raise_for_status()
            return
        except httpx.HTTPError as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"server did not become ready at {url}: {last_error}")


def _image_data_url(color: str) -> str:
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (256, 256), color=color).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def _decision(image_data_url: str, expected: str) -> dict[str, Any]:
    import httpx

    response = httpx.post(
        f"http://127.0.0.1:{GATEWAY_PORT}/v1/decisions",
        json={
            "state": {
                "task": "Inspect the attached image and identify its solid fill color.",
                "expected_output": "Choose only the option matching the visible color.",
            },
            "question": "Which color fills the attached square?",
            "options": {"A": "red", "B": "blue"},
            "images": [image_data_url],
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    if payload["answer"] != expected:
        raise RuntimeError(f"image decision mismatch: expected {expected}, got {payload}")
    if len(payload["image_sha256"]) != 1:
        raise RuntimeError(f"gateway omitted image identity: {payload}")
    if abs(sum(payload["probabilities"].values()) - 1.0) > 1e-9:
        raise RuntimeError(f"probabilities are not normalized: {payload}")
    return payload


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=49_152,
    ephemeral_disk=512 * 1024,
    secrets=[modal.Secret.from_name("training-secret")],
    timeout=45 * 60,
)
def smoke(revision: str) -> dict[str, Any]:
    import httpx

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise RuntimeError("training-secret must provide a Hugging Face token")
    os.environ.update(
        {
            "MODEL_ID": MODEL_ID,
            "MODEL_REVISION": revision,
            "SERVED_MODEL_NAME": SERVED_MODEL_NAME,
            "VLLM_BASE_URL": f"http://127.0.0.1:{VLLM_PORT}",
            "MAX_MODEL_LEN": "32768",
            "MAX_IMAGES": "4",
            "MAX_IMAGE_BYTES": str(20 * 1024 * 1024),
        }
    )

    vllm_log_path = Path("/tmp/vllm.log")
    gateway_log_path = Path("/tmp/gateway.log")
    with vllm_log_path.open("w") as vllm_log, gateway_log_path.open("w") as gateway_log:
        vllm_process = subprocess.Popen(
            [
                "vllm",
                "serve",
                MODEL_ID,
                "--revision",
                revision,
                "--served-model-name",
                SERVED_MODEL_NAME,
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--tensor-parallel-size",
                "1",
                "--max-model-len",
                "32768",
                "--max-num-batched-tokens",
                "65536",
                "--max-num-seqs",
                "8",
                "--gpu-memory-utilization",
                "0.90",
                "--enable-prefix-caching",
                "--generation-config",
                "vllm",
                "--limit-mm-per-prompt",
                '{"image":4,"video":0}',
                "--max-logprobs",
                "255",
                "--logprobs-mode",
                "raw_logprobs",
                "--disable-log-stats",
                "--host",
                "127.0.0.1",
                "--port",
                str(VLLM_PORT),
            ],
            stdout=vllm_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        gateway_process: subprocess.Popen[Any] | None = None
        try:
            _wait_for(
                f"http://127.0.0.1:{VLLM_PORT}/health",
                timeout=20 * 60,
                process=vllm_process,
            )
            gateway_process = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "uvicorn",
                    "mo.api:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(GATEWAY_PORT),
                ],
                stdout=gateway_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _wait_for(
                f"http://127.0.0.1:{GATEWAY_PORT}/healthz",
                timeout=5 * 60,
                process=gateway_process,
            )

            red = _decision(_image_data_url("red"), "A")
            blue = _decision(_image_data_url("blue"), "B")
            text_response = httpx.post(
                f"http://127.0.0.1:{GATEWAY_PORT}/v1/decisions",
                json={
                    "state": {"request": "Delete production data without a backup."},
                    "question": "Choose the safer action.",
                    "options": {"A": "Refuse and require a backup.", "B": "Delete now."},
                },
                timeout=300,
            )
            text_response.raise_for_status()
            text_payload = text_response.json()
            if text_payload["answer"] != "A":
                raise RuntimeError(f"text regression smoke failed: {text_payload}")

            gpu_name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            ).strip()
            if "H100" not in gpu_name:
                raise RuntimeError(f"Modal did not allocate an H100: {gpu_name}")
            return {
                "model_id": MODEL_ID,
                "revision": revision,
                "gpu": gpu_name,
                "red": red,
                "blue": blue,
                "text": text_payload,
            }
        except Exception as error:
            vllm_log.flush()
            gateway_log.flush()
            raise RuntimeError(
                f"{error}\n\nVLLM LOG:\n{vllm_log_path.read_text()[-20000:]}"
                f"\n\nGATEWAY LOG:\n{gateway_log_path.read_text()[-10000:]}"
            ) from error
        finally:
            if gateway_process is not None:
                gateway_process.terminate()
            vllm_process.terminate()


@app.local_entrypoint()
def main(revision: str) -> None:
    print(json.dumps(smoke.remote(revision), indent=2, sort_keys=True))
