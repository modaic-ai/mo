from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from mo.renderer import render_decision
from mo.settings import Settings
from mo.vllm_client import score_decision


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Any
    question: str = Field(min_length=1)
    options: dict[str, Any]


class DecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    output: dict[str, str]
    probabilities: dict[str, float]
    logprobs: dict[str, float]
    input_tokens: int
    normal_prompt_sha256: str
    partial_prompt_sha256: str
    usage: dict[str, Any] | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            trust_remote_code=True,
        )
        app.state.tokenizer = tokenizer
        app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        yield
        await app.state.client.aclose()

    app = FastAPI(title="Mo Decision API", version="0.1.0", lifespan=lifespan)

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if config.mo_api_key is None:
            return
        if authorization != f"Bearer {config.mo_api_key}":
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        response = await app.state.client.get(f"{config.vllm_base_url.rstrip('/')}/health")
        response.raise_for_status()
        return {"status": "ok"}

    @app.post(
        "/v1/decide",
        response_model=DecisionResponse,
        dependencies=[Depends(authorize)],
    )
    async def decide(request: DecisionRequest) -> DecisionResponse:
        try:
            rendered = render_decision(
                app.state.tokenizer,
                state=request.state,
                question=request.question,
                options=request.options,
                max_model_len=config.max_model_len,
            )
            result = await score_decision(
                app.state.client,
                base_url=config.vllm_base_url,
                api_key=config.vllm_api_key,
                served_model_name=config.served_model_name,
                rendered=rendered,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return DecisionResponse(
            answer=result["answer"],
            output={"answer": result["answer"]},
            probabilities=result["probabilities"],
            logprobs=result["logprobs"],
            input_tokens=rendered.input_tokens,
            normal_prompt_sha256=rendered.normal_prompt_sha256,
            partial_prompt_sha256=rendered.partial_prompt_sha256,
            usage=result["usage"],
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("mo.api:app", host="0.0.0.0", port=8080)
