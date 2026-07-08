from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration for the OmniVoice API.
    These settings can be overridden using environment variables or a .env file.
    """
    # API Settings
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Concurrency & Protection
    MAX_CONCURRENT_GENERATIONS: int = 4
    

    # Database Settings
    DB_PATH: str = "voices.db"

    # Chunking Settings (from Qwen3 architecture)
    AUTOCHUNK: bool = True
    MIN_CHUNK_CHARS: int = 20
    MAX_CHUNK_CHARS: int = 70
    CHUNK_GAP_MS: int = 120
    DEFAULT_SAMPLE_RATE: int = 24000
    
    # Model configuration
    MODEL_NAME: str = "omnivoice"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
