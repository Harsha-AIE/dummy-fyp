from __future__ import annotations

from edukg.core.config import OpenAISettings, get_settings
from edukg.core.exceptions import ExternalServiceError
from edukg.core.interfaces import LLMProvider
from edukg.core.logging_setup import get_logger

logger = get_logger(__name__)


class OpenAILLMProvider(LLMProvider):
    """
    LLMProvider implementation backed by the OpenAI Chat Completions API.

    This class has exactly one job: given a system/user prompt pair, return a raw
    JSON string. It knows nothing about ontologies, entities, or the pipeline —
    callers are responsible for prompt content and for validating the response.
    Swapping providers later means writing a new LLMProvider implementation; no
    other code changes.
    """

    def __init__(self, settings: OpenAISettings | None = None) -> None:
        self._settings = settings or get_settings().openai
        self._client = None  # lazy-initialized: keeps import optional until actually used

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ExternalServiceError("The 'openai' package is not installed") from exc
            if not self._settings.api_key:
                raise ExternalServiceError("EDUKG_OPENAI_API_KEY is not configured")
            self._client = OpenAI(api_key=self._settings.api_key, timeout=self._settings.request_timeout_seconds)
        return self._client

    def model_name(self) -> str:
        return self._settings.model

    def complete_json(self, *, system_prompt: str, user_prompt: str, schema_name: str) -> str:
        client = self._get_client()
        logger.info("llm_request_started", model=self._settings.model, schema_name=schema_name)
        try:
            response = client.chat.completions.create(
                model=self._settings.model,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_output_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure becomes a typed error
            logger.error("llm_request_failed", schema_name=schema_name, error=str(exc))
            raise ExternalServiceError(f"OpenAI request failed for schema '{schema_name}': {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise ExternalServiceError(f"OpenAI returned an empty response for schema '{schema_name}'")

        logger.info("llm_request_completed", schema_name=schema_name, response_chars=len(content))
        return content
