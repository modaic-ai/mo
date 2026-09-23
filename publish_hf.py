from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal

MODEL_VOLUME = "qwen36-35b-json-decider-fp8"
SOURCE = Path("/models/qwen36-35b-json-decider-ckpt3450-fp8-dynamic")
EXPECTED = {
    "config.json": "05c1952302c5cf4c843e128be1918116abc501bf1fdc0615e695462a4099cb80",
    "model.safetensors.index.json": (
        "4e6396b31910b520540132f781ebe97b0c8ce82956d2f7e76dd5a5cd47964ad7"
    ),
    "SUCCESS.json": "f96cb5aa3f001443503d7218acee4c682823be4efd03a102be6e1df7e507bedb",
    "chat_template.jinja": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
}
EXPECTED_SHARDS = 38
EXPECTED_PARAMETERS = 35_138_874_736
EXPECTED_WEIGHT_BYTES = 37_667_035_872
PACKAGED_ASSETS = {
    "preprocessor_config.json": "5102cc0567b75a34738c5af8e547c00b1faca4ca63c60b1b79acd0423af0ef43",
    "video_preprocessor_config.json": (
        "00bd47a5eaaf8760744a12658cd99ba168b818edc4c0a983b90157289c8e546a"
    ),
}

app = modal.App("mo-publish-hugging-face")
volume = modal.Volume.from_name(MODEL_VOLUME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub>=1.1,<2")
    .add_local_file("hf/README.md", "/opt/mo/README.md")
    .add_local_file("hf/provenance.json", "/opt/mo/provenance.json")
    .add_local_file("hf/preprocessor_config.json", "/opt/mo/preprocessor_config.json")
    .add_local_file(
        "hf/video_preprocessor_config.json", "/opt/mo/video_preprocessor_config.json"
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source() -> dict[str, object]:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    for filename, expected in EXPECTED.items():
        actual = _sha256(SOURCE / filename)
        if actual != expected:
            raise RuntimeError(f"artifact hash drift for {filename}: {actual}")
    forbidden = sorted(
        path.name
        for path in SOURCE.iterdir()
        if "adapter" in path.name.lower() or "lora" in path.name.lower()
    )
    if forbidden:
        raise RuntimeError(f"merged artifact contains PEFT files: {forbidden}")

    config = json.loads((SOURCE / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    if (
        quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("quantization_status") != "compressed"
        or quantization.get("format") != "float-quantized"
    ):
        raise RuntimeError("artifact is not the accepted compressed-tensors FP8 release")

    index = json.loads((SOURCE / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("invalid weight index")
    if any("adapter" in key.lower() or "lora" in key.lower() for key in weight_map):
        raise RuntimeError("weight index still contains adapter parameters")
    shards = set(weight_map.values())
    metadata = index.get("metadata", {})
    if (
        len(shards) != EXPECTED_SHARDS
        or metadata.get("total_parameters") != EXPECTED_PARAMETERS
        or metadata.get("total_size") != EXPECTED_WEIGHT_BYTES
    ):
        raise RuntimeError("weight index identity drift")
    missing = sorted(shard for shard in shards if not (SOURCE / shard).is_file())
    if missing:
        raise RuntimeError(f"missing weight shards: {missing}")
    return {
        "parameters": EXPECTED_PARAMETERS,
        "weight_bytes": EXPECTED_WEIGHT_BYTES,
        "shards": EXPECTED_SHARDS,
    }


def _verify_packaged_assets() -> None:
    for filename, expected in PACKAGED_ASSETS.items():
        actual = _sha256(Path("/opt/mo") / filename)
        if actual != expected:
            raise RuntimeError(f"packaged processor hash drift for {filename}: {actual}")


@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("training-secret")],
    cpu=8,
    memory=16_384,
    timeout=24 * 60 * 60,
)
def publish(
    repo_id: str,
    create: bool,
    private: bool,
    revision: str,
    upload_weights: bool,
) -> dict[str, object]:
    from huggingface_hub import HfApi

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise RuntimeError("training-secret must provide HF_TOKEN or HUGGING_FACE_HUB_TOKEN")
    identity = _verify_source()
    _verify_packaged_assets()
    api = HfApi()
    if create:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=False)
    else:
        info = api.model_info(repo_id=repo_id, revision=revision)
        if bool(info.private) != private:
            raise RuntimeError(
                f"repository privacy mismatch: existing private={info.private}, requested={private}"
            )
    if upload_weights:
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            folder_path=SOURCE,
            num_workers=8,
        )
    card_commit = api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        path_or_fileobj="/opt/mo/README.md",
        path_in_repo="README.md",
        commit_message="Add Mo model card",
    )
    provenance_commit = api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        path_or_fileobj="/opt/mo/provenance.json",
        path_in_repo="provenance.json",
        commit_message="Add sealed Mo provenance",
    )
    processor_commits = {
        filename: api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            path_or_fileobj=f"/opt/mo/{filename}",
            path_in_repo=filename,
            commit_message=f"Add pinned {filename}",
        ).oid
        for filename in PACKAGED_ASSETS
    }
    final = api.model_info(repo_id=repo_id, revision=revision)
    return {
        **identity,
        "repo_id": repo_id,
        "private": bool(final.private),
        "revision": revision,
        "commit_sha": final.sha,
        "card_commit": card_commit.oid,
        "provenance_commit": provenance_commit.oid,
        "processor_commits": processor_commits,
        "uploaded_weights": upload_weights,
    }


@app.local_entrypoint()
def main(
    repo_id: str,
    create: bool = False,
    private: bool = True,
    revision: str = "main",
    upload_weights: bool = True,
) -> None:
    print(
        json.dumps(
            publish.remote(repo_id, create, private, revision, upload_weights),
            indent=2,
            sort_keys=True,
        )
    )
