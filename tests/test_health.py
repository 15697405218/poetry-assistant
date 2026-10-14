from fastapi.testclient import TestClient

from app.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "index_version" in r.json()
    assert "fts_poems" in r.json()
    assert "queue" in r.json()
