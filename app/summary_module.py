from .gemini_client import GeminiService


SUMMARY_SYSTEM = """
You are EduGenie's educational summarizer.

Summarize the supplied passage for quick revision.

Preserve the important facts and relationships.

Remove repetition and irrelevant wording.

Use simple language and concise bullet points
where useful.

Do not introduce information that is not supported
by the supplied passage.
""".strip()


def summarize_text(text: str) -> str:

    service = GeminiService()

    return service.generate(
        text,
        system_instruction=SUMMARY_SYSTEM,
        temperature=0.2,
        max_output_tokens=1800,
    )