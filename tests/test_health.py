"""Tests for the health check endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_reports_healthy_status() -> None:
    response = client.get("/api/v1/health")
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "arxiv-graph-rag-api"
