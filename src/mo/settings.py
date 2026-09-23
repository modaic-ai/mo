from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_id: str = "modaic/mo-1.1-fp8"
    model_revision: str = "60cd2356f6f4e9ec83967aa4f484c26a81f08b88"
    served_model_name: str = "mo"
    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str | None = None
    mo_api_key: str | None = None
    max_model_len: int = 32_768
    max_images: int = 4
    max_image_bytes: int = 20 * 1024 * 1024
