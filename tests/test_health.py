from fastapi import FastAPI
from fastapi.testclient import TestClient


def create_test_app():

    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "healthy"
        }

    return app


def test_health():

    app = create_test_app()

    with TestClient(app) as client:

        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        assert response.json() == {
            "status": "healthy"
        }