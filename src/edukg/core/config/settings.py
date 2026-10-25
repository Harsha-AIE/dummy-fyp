from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUKG_PATH_")

    project_root: Path = Path(__file__).resolve().parents[4]
    data_dir: Path | None = Field(default=None)
    raw_dir: Path | None = Field(default=None)
    processed_dir: Path | None = Field(default=None)
    cache_dir: Path | None = Field(default=None)
    ontology_dir: Path | None = Field(default=None)
    log_dir: Path | None = Field(default=None)

    def model_post_init(self, __context) -> None:
        object.__setattr__(self, "data_dir", self.data_dir or self.project_root / "data")
        object.__setattr__(self, "raw_dir", self.raw_dir or self.data_dir / "raw")
        object.__setattr__(self, "processed_dir", self.processed_dir or self.data_dir / "processed")
        object.__setattr__(self, "cache_dir", self.cache_dir or self.data_dir / "cache")
        object.__setattr__(self, "ontology_dir", self.ontology_dir or self.data_dir / "ontologies")
        object.__setattr__(self, "log_dir", self.log_dir or self.project_root / "logs")
        for p in (self.data_dir, self.raw_dir, self.processed_dir, self.cache_dir, self.ontology_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUKG_OPENAI_", env_file=".env", extra="ignore")

    api_key: str = Field(default="")
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, gt=0)
    request_timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI deployment used for both chat (ontology extension proposals,
    stage 4) and embeddings (entity linking, stage 7). Deliberately uses the
    AZURE_OPENAI_ env prefix (not EDUKG_) to match the deployment's own
    conventional env var names rather than requiring them to be renamed."""

    model_config = SettingsConfigDict(env_prefix="AZURE_OPENAI_", env_file=".env", extra="ignore")

    endpoint: str = Field(default="")
    api_key: str = Field(default="")
    api_version: str = Field(default="2024-12-01-preview")
    embedding_api_version: str = Field(default="2024-02-01")
    chat_deployment: str = Field(default="gpt-4o")
    embedding_deployment: str = Field(default="text-embedding-3-large")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, gt=0)
    request_timeout_seconds: int = Field(default=60, gt=0)


class PostgresSettings(BaseSettings):
    """Neon-hosted Postgres — durable persistence-of-record for uploaded
    source documents. The pipeline still processes from the local
    data/raw copy (PDFProcessor/DOCXProcessor/PaddleOCR all expect a real
    filesystem path), so Postgres isn't in that hot path; it exists so an
    upload survives a filesystem wipe and isn't tied to one server's disk.
    Reads the single DATABASE_URL env var — Neon's/most hosts' own
    convention — rather than an EDUKG_-prefixed set of parts."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    dsn: str = Field(default="", validation_alias="DATABASE_URL")
    connect_timeout_seconds: int = Field(
        default=5,
        gt=0,
        description=(
            "psycopg2's connect_timeout — without an explicit value it falls back to the OS/driver "
            "default, which can take 10-20+ seconds to give up on an unreachable host. Found live: "
            "a transient Neon outage made every /api/agents/ask request (which touches Postgres up "
            "to twice — Knowledge Tracing's event write and the mastery-gap re-ranking read) take "
            "20+ seconds before its own try/except finally caught the failure and degraded "
            "gracefully. A short timeout makes that degradation fast instead of slow."
        ),
    )


class Neo4jSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUKG_NEO4J_", env_file=".env", extra="ignore")

    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="")
    database: str = Field(default="neo4j")
    max_connection_pool_size: int = Field(default=50, gt=0)
    batch_size: int = Field(default=500, gt=0)


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUKG_EMBEDDING_", protected_namespaces=())

    model_name: str = Field(default="sentence-transformers/all-mpnet-base-v2")
    dimension: int = Field(default=768, gt=0)
    batch_size: int = Field(default=32, gt=0)
    device: str = Field(default="cpu")


class OCRSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUKG_OCR_")

    lang: str = Field(default="en")
    use_gpu: bool = Field(default=False)
    layout_aware: bool = Field(default=True)


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUKG_CHUNK_")

    chunk_size_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def overlap_lt_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size_tokens", 512)
        if v >= size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return v


class ThresholdSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUKG_THRESHOLD_")

    entity_linking_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    entity_linking_query_similarity: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Separate, deliberately lower similarity floor used only for read-only "
            "query-time concept tagging (ConceptTagger/create_if_missing=False), not "
            "ingestion-time entity creation. Ingestion's 0.85 is calibrated to avoid "
            "wrongly merging two distinct concepts into one permanent KG entity — a "
            "real, hard-to-undo mistake. Query-time tagging never mutates the KG "
            "either way, so the cost of a looser match is just a slightly-imprecise "
            "concept badge/re-rank signal, not data corruption, and a student asking "
            "'genetic material' should still surface the DNA concept. Calibrated "
            "empirically: real paraphrase pairs (e.g. 'genetic material'/'DNA', "
            "'drugs stop working against infections'/'antibiotic resistance') score "
            "0.41-0.57 cosine similarity on this embedding model; genuinely unrelated "
            "pairs score 0.08-0.26 — 0.4 sits with a wide margin above the unrelated "
            "cluster."
        ),
    )
    relation_confidence_min: float = Field(default=0.55, ge=0.0, le=1.0)
    triple_confidence_min: float = Field(default=0.5, ge=0.0, le=1.0)
    ontology_extension_confidence_min: float = Field(default=0.6, ge=0.0, le=1.0)
    intent_confidence_min: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Below this, PedagogicalGuideAgent escalates tier-1 to the LLM tier"
    )


class CacheSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUKG_CACHE_")

    enabled: bool = Field(default=True)
    backend: str = Field(default="filesystem")  # filesystem | redis (future)


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUKG_LOG_")

    level: str = Field(default="INFO")
    json_logs: bool = Field(default=True)
    log_to_file: bool = Field(default=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="development")
    api_cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:3001",
        description=(
            "Comma-separated allowed CORS origins for the FastAPI app, read from API_CORS_ORIGINS. "
            "Previously hardcoded in main.py — that caused a real outage when the frontend dev "
            "server had to move to a different port (its usual one was taken by another project) "
            "and every request started failing CORS with no clear error."
        ),
    )
    paths: PathSettings = Field(default_factory=PathSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
