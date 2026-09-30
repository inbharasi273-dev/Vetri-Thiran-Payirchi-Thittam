from .gemini_client import GeminiService


SYSTEM_PROMPT = """
You are EduGenie, a reliable educational assistant.

Answer the student's question clearly and concisely.

Use simple language appropriate for a learner.

If the question is ambiguous, briefly explain the ambiguity
and answer the most likely interpretation.

Do not invent facts, sources, citations, or calculations.

For academic questions, provide a short explanation rather
than only giving the final answer.
""".strip()


def answer_question(question: str) -> str:

    service = GeminiService()

    return service.generate(
        question,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=1200,
    )