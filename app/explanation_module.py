from __future__ import annotations

from functools import lru_cache

from .config import get_settings
from .gemini_client import GeminiService


try:
    import torch

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

except Exception:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


@lru_cache(maxsize=1)
def _load_local_model():

    settings = get_settings()

    if (
        AutoTokenizer is None
        or AutoModelForSeq2SeqLM is None
    ):
        raise RuntimeError(
            "Transformers or PyTorch is not available."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        settings.explanation_model
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.explanation_model
    )

    model.eval()

    return tokenizer, model


def local_model_loaded() -> bool:

    return _load_local_model.cache_info().currsize > 0


def _local_explain(topic: str) -> str:

    tokenizer, model = _load_local_model()

    prompt = (
        "Explain the following educational concept "
        "in simple language for a beginner. "
        "Use a short definition, 2-4 key points, "
        "and one simple example. "
        f"Concept: {topic}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=220,
            num_beams=4,
            early_stopping=True,
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    return result.strip()


def explain_concept(topic: str) -> str:

    settings = get_settings()

    try:

        result = _local_explain(topic)

        if result:
            return result

        raise RuntimeError(
            "Local model returned an empty explanation."
        )

    except Exception as local_error:

        if not settings.explanation_allow_gemini_fallback:

            raise RuntimeError(
                f"Local explanation model failed: "
                f"{local_error}"
            ) from local_error

        service = GeminiService()

        return service.generate(
            f"Explain this concept for a beginner: {topic}",
            system_instruction=(
                "Give a simple educational explanation. "
                "Include a definition, key points, "
                "and one easy example."
            ),
            temperature=0.25,
            max_output_tokens=900,
        )