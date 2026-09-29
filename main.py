from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.responses import HTMLResponse

from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates


from .config import get_settings

from .explanation_module import (
    explain_concept,
    local_model_loaded,
)

from .gemini_client import GeminiService

from .learning_path import (
    recommend_learning_path,
)

from .qna import answer_question

from .quiz_module import generate_quiz

from .schemas import (
    AnswerResponse,
    HealthResponse,
    QuizRequest,
    QuizResponse,
    TextRequest,
)

from .summary_module import summarize_text


BASE_DIR = Path(__file__).resolve().parent


app = FastAPI(
    title="EduGenie API",

    description=(
        "Google Gemini powered learning assistant."
    ),

    version="1.0.0",
)


app.mount(
    "/static",

    StaticFiles(
        directory=BASE_DIR / "static"
    ),

    name="static",
)


templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)


@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,

        name="index.html",

        context={},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health():

    settings = get_settings()

    gemini = GeminiService()

    return HealthResponse(

        status="ok",

        gemini_configured=gemini.configured,

        gemini_model=settings.gemini_model,

        explanation_model=settings.explanation_model,

        local_explanation_loaded=local_model_loaded(),
    )


@app.post(
    "/qa",
    response_model=AnswerResponse,
)
async def qa(
    payload: TextRequest,
):

    try:

        answer = answer_question(
            payload.text
        )

        return AnswerResponse(
            result=answer
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@app.post(
    "/explain",
    response_model=AnswerResponse,
)
async def explain(
    payload: TextRequest,
):

    try:

        answer = explain_concept(
            payload.text
        )

        return AnswerResponse(
            result=answer
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@app.post(
    "/quiz",
    response_model=QuizResponse,
)
async def quiz(
    payload: QuizRequest,
):

    try:

        return generate_quiz(
            payload.text,
            payload.num_questions,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@app.post(
    "/summarize",
    response_model=AnswerResponse,
)
async def summarize(
    payload: TextRequest,
):

    try:

        answer = summarize_text(
            payload.text
        )

        return AnswerResponse(
            result=answer
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@app.post(
    "/learn/recommendations",
    response_model=AnswerResponse,
)
async def learning_recommendations(
    payload: TextRequest,
):

    try:

        answer = recommend_learning_path(
            payload.text
        )

        return AnswerResponse(
            result=answer
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc