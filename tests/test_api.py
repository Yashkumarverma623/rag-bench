from fastapi.testclient import TestClient

from rag_bench.serve import app

client = TestClient(app)


def test_get_root_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "rag-bench" in response.text


def test_get_results_endpoint():
    response = client.get("/results")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert isinstance(data["runs"], list)
