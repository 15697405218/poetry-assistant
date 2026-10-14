from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# Windows + rust tokenizers: parallel encode can 0xC0000005 the process.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    access_token_max_minutes: int = 1440
    ark_api_key: str
    base_url: str
    chat_model: str
    poetry_data_dir: str
    chroma_persist_dir: str
    tavily_api_key: str = ""
    reranker_model: str = "BAAI/bge-reranker-large"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    redis_url: str = "redis://localhost:6379/0"
    chat_rate_limit: int = 20
    chat_rate_window: int = 60
    login_fail_max: int = 5
    login_lock_seconds: int = 900
    cache_ttl_seconds: int = 21600
    retrieve_top_k: int = 5
    recall_k: int = 15
    poem_distance_max: float = 1.2
    raw_token_limit: int = 100000
    keep_low: int = 15
    index_version: str = "v2"
    hnsw_search_ef: int = 64
    gpu_max_inflight: int = 1
    job_ttl_seconds: int = 3600
    job_wait_seconds: int = 180


def _overlay_yaml(data: Settings) -> Settings:
    raw = os.getenv("APP_CONFIG", "config/dev.yaml")
    path = Path(raw)
    if not path.is_file():
        return data
    try:
        import yaml
    except ImportError:
        return data
    extra = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(extra, dict):
        return data
    keep = {k: v for k, v in extra.items() if hasattr(data, k) and v is not None}
    if not keep:
        return data
    return data.model_copy(update=keep)


settings = _overlay_yaml(Settings())