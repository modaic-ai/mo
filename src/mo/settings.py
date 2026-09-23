from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_id: str = "modaic/mo"
    model_revision: str = "main"
    served_model_name: str = "mo"
    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str | None = None
    mo_api_key: str | None = None
    max_model_len: int = 32_768
