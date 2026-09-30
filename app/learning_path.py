from .gemini_client import GeminiService


LEARNING_SYSTEM = """
You are EduGenie's learning-path planner.

Create a practical roadmap from beginner to advanced.

For each stage include:

1. What to learn
2. Important concepts
3. A small practice activity

Adapt the roadmap to the learner's requested topic.

Do not assume prior knowledge unless the learner
explicitly provides it.
""".strip()


def recommend_learning_path(
    topic: str,
) -> str:

    service = GeminiService()

    return service.generate(
        f"Create a beginner-to-advanced learning path for: {topic}",

        system_instruction=LEARNING_SYSTEM,

        temperature=0.35,

        max_output_tokens=1800,
    )