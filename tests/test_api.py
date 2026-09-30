from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_page():

    response = client.get("/")

    assert response.status_code == 200

    assert "EduGenie" in response.text


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    assert "gemini_model" in data


def test_qa_validation():

    response = client.post(
        "/qa",
        json={
            "text": "x"
        },
    )

    assert response.status_code == 422


def test_quiz_validation():

    response = client.post(
        "/quiz",
        json={
            "text": "Pythagoras theorem",
            "num_questions": 20,
        },
    )

    assert response.status_code == 422