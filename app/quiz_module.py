from .gemini_client import (
    GeminiService,
    parse_json_response,
)

from .schemas import (
    QuizQuestion,
    QuizResponse,
)


QUIZ_SCHEMA = {

    "type": "OBJECT",

    "properties": {

        "questions": {

            "type": "ARRAY",

            "items": {

                "type": "OBJECT",

                "properties": {

                    "question": {
                        "type": "STRING"
                    },

                    "options": {

                        "type": "ARRAY",

                        "items": {
                            "type": "STRING"
                        }
                    },

                    "correct_answer": {
                        "type": "STRING"
                    },
                },

                "required": [
                    "question",
                    "options",
                    "correct_answer",
                ],
            },
        }
    },

    "required": [
        "questions"
    ],
}


def generate_quiz(
    topic: str,
    num_questions: int,
) -> QuizResponse:

    prompt = f"""
Create an educational quiz about:

{topic}

Requirements:

- Generate exactly {num_questions} questions.
- Every question must contain exactly four options.
- Every option must be different.
- correct_answer must exactly match one of the options.
- Mix recall and understanding questions.
- Questions should be suitable for students.
- Return only valid JSON.
""".strip()

    service = GeminiService()

    raw = service.generate(
        prompt,

        system_instruction=(
            "You generate structured educational "
            "quiz data."
        ),

        temperature=0.5,

        max_output_tokens=3000,

        response_mime_type="application/json",

        response_schema=QUIZ_SCHEMA,
    )

    data = parse_json_response(raw)

    if isinstance(data, dict):

        questions = data.get(
            "questions"
        )

    else:

        questions = data

    if not isinstance(questions, list):

        raise ValueError(
            "Quiz response did not contain "
            "a questions list."
        )

    normalized = []

    for item in questions[:num_questions]:

        question = QuizQuestion.model_validate(
            item
        )

        if len(question.options) != 4:

            raise ValueError(
                "Each quiz question must contain "
                "exactly four options."
            )

        if question.correct_answer not in question.options:

            raise ValueError(
                "correct_answer must match one "
                "of the available options."
            )

        normalized.append(question)

    if len(normalized) != num_questions:

        raise ValueError(
            f"Expected {num_questions} questions, "
            f"received {len(normalized)}."
        )

    return QuizResponse(
        topic=topic,
        questions=normalized,
    )