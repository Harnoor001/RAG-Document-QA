"""Runtime settings. Every field can be overridden by an environment variable of the same name."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    chunk_size: int = 800
    chunk_overlap: int = 150
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: str = "data/indexes"

    top_k: int = 5
    confidence_threshold: float = 0.35
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_timeout_s: float = 60.0
    ollama_keep_alive: str = "30m"  # how long Ollama keeps the model in memory after a call
    max_context_chars: int = 6000


def get_settings() -> Settings:
    """Read settings fresh from the environment (no caching, so env overrides apply immediately)."""
    return Settings()
