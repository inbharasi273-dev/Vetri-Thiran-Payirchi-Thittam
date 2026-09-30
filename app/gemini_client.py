from __future__ import annotations

import json
import re
from typing import Any

from google import genai
from google.genai import types

from .config import get_settings


class GeminiService:

    def __init__(self) -> None:
        settings = get_settings()

        self.model = settings.gemini_model

        if settings.gemini_api_key:
            self._client = genai.Client(
                api_key=settings.gemini_api_key
            )
        else:
            self._client = None

    @property
    def configured(self) -> bool:
        return self._client is not None

    def _require_client(self):
        if self._client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Add your Gemini API key to the .env file."
            )

        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 2048,
        response_mime_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:

        client = self._require_client()

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )

        if response_mime_type:
            config.response_mime_type = response_mime_type

        if response_schema:
            config.response_schema = response_schema

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        text = getattr(response, "text", None)

        if not text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return text.strip()


def clean_json_block(raw: str) -> str:

    value = raw.strip()

    value = re.sub(
        r"^```(?:json)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\s*```$",
        "",
        value,
    )

    return value.strip()


def parse_json_response(raw: str) -> Any:

    cleaned = clean_json_block(raw)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        candidates = []

        first_object = cleaned.find("{")
        last_object = cleaned.rfind("}")

        if first_object >= 0 and last_object > first_object:
            candidates.append(
                cleaned[first_object:last_object + 1]
            )

        first_array = cleaned.find("[")
        last_array = cleaned.rfind("]")

        if first_array >= 0 and last_array > first_array:
            candidates.append(
                cleaned[first_array:last_array + 1]
            )

        for candidate in candidates:

            try:
                return json.loads(candidate)

            except json.JSONDecodeError:
                continue

        raise ValueError(
            "Gemini returned invalid JSON."
        ) from None