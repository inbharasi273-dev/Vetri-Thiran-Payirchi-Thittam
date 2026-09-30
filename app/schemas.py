from pydantic import BaseModel, Field, field_validator


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        max_length=50000,
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()

        if len(value) < 2:
            raise ValueError(
                "Text must contain at least 2 non-space characters."
            )

        return value


class QuizRequest(TextRequest):
    num_questions: int = Field(
        default=5,
        ge=1,
        le=15,
    )


class AnswerResponse(BaseModel):
    result: str


class QuizQuestion(BaseModel):
    question: str

    options: list[str] = Field(
        min_length=4,
        max_length=4,
    )

    correct_answer: str

    @field_validator("correct_answer")
    @classmethod
    def clean_answer(cls, value: str) -> str:
        return value.strip()


class QuizResponse(BaseModel):
    topic: str
    questions: list[QuizQuestion]


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
    gemini_model: str
    explanation_model: str
    local_explanation_loaded: bool