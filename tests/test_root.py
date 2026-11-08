from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_returns_service_name():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "supplypilot"
