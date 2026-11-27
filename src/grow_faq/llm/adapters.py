"""One ``generate(system, user)`` contract for every configured provider.

Provider packages are imported lazily so asking a question only needs the
client for the selected provider. Adapters never add facts of their own.
"""

from __future__ import annotations

from typing import Any, Protocol

from grow_faq.config import Settings

MAX_TOKENS = 512


class LLMError(RuntimeError):
    """A provider call failed or returned no usable text."""


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str:
        """Return the provider's draft answer."""


def _text(value: Any) -> str:
    if value is None:
        raise LLMError("the provider returned an empty response")
    if isinstance(value, str):
        if not value.strip():
            raise LLMError("the provider returned an empty response")
        return value
    if isinstance(value, list):
        parts = [
            part.text if hasattr(part, "text") else str(part)
            for part in value
            if part is not None
        ]
        result = "".join(parts).strip()
        if not result:
            raise LLMError("the provider returned an empty response")
        return result
    result = str(value).strip()
    if not result:
        raise LLMError("the provider returned an empty response")
    return result


class GroqLLM:
    def __init__(self, settings: Settings, api_key: str) -> None:
        from groq import Groq

        self._model = settings.llm_model
        self._client = Groq(api_key=api_key, timeout=settings.llm_timeout_seconds)

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        return _text(response.choices[0].message.content)


class GeminiLLM:
    def __init__(self, settings: Settings, api_key: str) -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(settings.llm_model)

    def generate(self, system: str, user: str) -> str:
        response = self._model.generate_content(f"{system}\n\n{user}")
        return _text(getattr(response, "text", None))


class OpenAILLM:
    def __init__(self, settings: Settings, api_key: str) -> None:
        from openai import OpenAI

        self._model = settings.llm_model
        self._client = OpenAI(api_key=api_key, timeout=settings.llm_timeout_seconds)

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        return _text(response.choices[0].message.content)


class AnthropicLLM:
    def __init__(self, settings: Settings, api_key: str) -> None:
        from anthropic import Anthropic

        self._model = settings.llm_model
        self._client = Anthropic(api_key=api_key, timeout=settings.llm_timeout_seconds)

    def generate(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _text(getattr(response, "content", None))


class MistralLLM:
    def __init__(self, settings: Settings, api_key: str) -> None:
        from mistralai import Mistral

        self._model = settings.llm_model
        self._client = Mistral(api_key=api_key)

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.complete(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=MAX_TOKENS,
        )
        return _text(response.choices[0].message.content)


ADAPTERS = {
    "groq": GroqLLM,
    "gemini": GeminiLLM,
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "mistral": MistralLLM,
}


def build_llm(settings: Settings) -> LLM:
    """Instantiate the provider selected in config."""
    adapter = ADAPTERS.get(settings.llm_provider)
    if adapter is None:
        raise LLMError(f"unsupported LLM provider: {settings.llm_provider}")
    try:
        return adapter(settings, settings.require_api_key())
    except ImportError as exc:
        raise LLMError(
            f"the client package for LLM provider {settings.llm_provider!r} is "
            f"not installed. Run: pip install -e .[{settings.llm_provider}]"
        ) from exc
