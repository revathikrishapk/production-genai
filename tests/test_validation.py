from fastapi import FastAPI

from app.api.schemas import QueryRequest


def create_test_app():

    app = FastAPI()

    @app.post("/query")
    def query(request: QueryRequest):

        return {
            "question": request.question
        }

    return app


def test_empty_question():

    app = create_test_app()

    from fastapi.testclient import TestClient

    with TestClient(app) as client:

        response = client.post(
            "/query",
            json={
                "question": ""
            },
        )

        assert response.status_code == 422


def test_placeholder_question():

    app = create_test_app()

    from fastapi.testclient import TestClient

    with TestClient(app) as client:

        response = client.post(
            "/query",
            json={
                "question": "string"
            },
        )

        assert response.status_code == 422


def test_valid_question():

    app = create_test_app()

    from fastapi.testclient import TestClient

    with TestClient(app) as client:

        response = client.post(
            "/query",
            json={
                "question": "What is PageRank?"
            },
        )

        assert response.status_code == 200