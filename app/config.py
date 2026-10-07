"""Runtime settings. Every field can be overridden by an environment variable of the same name."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    chunk_size: int = 800
    chunk_overlap: int = 150
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: str = "data/indexes"


def get_settings() -> Settings:
    """Read settings fresh from the environment (no caching, so env overrides apply immediately)."""
    return Settings()
