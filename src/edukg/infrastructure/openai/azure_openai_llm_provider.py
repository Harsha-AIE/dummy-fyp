from __future__ import annotations

from edukg.core.config import AzureOpenAISettings, get_settings
from edukg.core.exceptions import ExternalServiceError
from edukg.core.interfaces import LLMProvider
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)


class AzureOpenAILLMProvider(LLMProvider):
    """
    LLMProvider implementation backed by an Azure OpenAI chat deployment.

    Same contract as OpenAILLMProvider (given a system/user prompt pair, return a
    raw JSON string) so it is a drop-in replacement wherever LLMProvider is
    injected (e.g. OntologyExtensionService) — no caller changes required.
    Azure's API takes the *deployment name* in the `model` field, not a model
    name, which is the only real difference from the plain-OpenAI provider.
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
                api_version=self._settings.api_version,
                timeout=self._settings.request_timeout_seconds,
            )
        return self._client

    def model_name(self) -> str:
        return self._settings.chat_deployment

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> str:
        client = self._get_client()
        logger.info("azure_llm_request_started", deployment=self._settings.chat_deployment, schema_name=schema_name)
        try:
            response = client.chat.completions.create(
                model=self._settings.chat_deployment,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_output_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure becomes a typed error
            logger.error("azure_llm_request_failed", schema_name=schema_name, error=str(exc))
            raise ExternalServiceError(f"Azure OpenAI request failed for schema '{schema_name}': {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise ExternalServiceError(f"Azure OpenAI returned an empty response for schema '{schema_name}'")

        logger.info("azure_llm_request_completed", schema_name=schema_name, response_chars=len(content))
        return content
