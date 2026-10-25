from __future__ import annotations

from edukg.core.config import AzureOpenAISettings, get_settings
from edukg.core.exceptions import ExternalServiceError
from edukg.core.interfaces import EmbeddingProvider
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)

# Native output size of text-embedding-3-large. Azure's embeddings endpoint uses
# its own api-version, distinct from the chat deployment's, hence a separate
# client instance from AzureOpenAILLMProvider rather than a shared one.
_TEXT_EMBEDDING_3_LARGE_DIMENSION = 3072


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """
    EmbeddingProvider implementation backed by an Azure OpenAI embedding
    deployment (text-embedding-3-large). Used by EntityLinkingService (stage 7)
    and sized to whatever this provider reports via dimension() — callers must
    not assume the local-model default in EmbeddingSettings.
    """

    def __init__(self, settings: AzureOpenAISettings | None = None) -> None:
        self._settings = settings or get_settings().azure_openai
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError as exc:
                raise ExternalServiceError("The 'openai' package is not installed") from exc
            if not self._settings.endpoint or not self._settings.api_key:
                raise ExternalServiceError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY are not configured")
            self._client = AzureOpenAI(
                azure_endpoint=self._settings.endpoint,
                api_key=self._settings.api_key,
                api_version=self._settings.embedding_api_version,
                timeout=self._settings.request_timeout_seconds,
            )
        return self._client

    def dimension(self) -> int:
        return _TEXT_EMBEDDING_3_LARGE_DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        logger.info("azure_embedding_request_started", deployment=self._settings.embedding_deployment, count=len(texts))
        try:
            response = client.embeddings.create(model=self._settings.embedding_deployment, input=texts)
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure becomes a typed error
            logger.error("azure_embedding_request_failed", error=str(exc))
            raise ExternalServiceError(f"Azure OpenAI embedding request failed: {exc}") from exc

        vectors = [item.embedding for item in response.data]
        logger.info("azure_embedding_request_completed", count=len(vectors))
        return vectors
