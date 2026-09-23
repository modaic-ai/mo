from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_id: str = "modaic/mo-1.1-fp8"
    model_revision: str = "b9a691fbfd931e6a4bb5bde3a70276ace0c7337f"
    served_model_name: str = "mo"
    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str | None = None
    mo_api_key: str | None = None
    max_model_len: int = 32_768
