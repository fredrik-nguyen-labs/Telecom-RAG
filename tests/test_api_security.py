from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.requests import Request

from backend.app import main

client = TestClient(main.app)


def test_refuses_non_json_chat_requests() -> None:
    response = client.post(
        "/api/chat",
        content="hello",
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 415


def test_refuses_oversized_chat_requests() -> None:
    response = client.post(
        "/api/chat",
        json={"question": "x" * (main.MAX_REQUEST_BYTES + 5)},
    )
    assert response.status_code == 413


def test_security_headers_and_no_cache() -> None:
    response = client.get("/not-found")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"


def test_rate_limit_identity_does_not_trust_caller_forwarded_for() -> None:
    request = Request(
        {
            "type": "http",
            "client": ("203.0.113.10", 443),
            "headers": [(b"x-forwarded-for", b"198.51.100.4")],
        }
    )
    assert main._client_key(request) == "203.0.113.10"


def test_health_does_not_expose_internal_errors(monkeypatch) -> None:
    def fail():
        raise RuntimeError("private provider configuration and token")

    monkeypatch.setattr(main, "get_runtime", fail)
    response = client.get("/api/health")
    assert response.status_code == 503
    assert "private" not in response.text
    assert "token" not in response.text


def test_early_rejection_exposes_error_only_to_allowed_origins(monkeypatch) -> None:
    # Production uses this allowlisted origin; local tests default to localhost.
    monkeypatch.setenv("CORS_ORIGINS", "https://telecom.fnsystems.dev")
    response = client.post(
        "/api/chat",
        content="invalid",
        headers={
            "Content-Type": "text/plain",
            "Origin": "https://telecom.fnsystems.dev",
        },
    )
    assert response.status_code == 415
    assert response.headers["access-control-allow-origin"] == "https://telecom.fnsystems.dev"
    assert response.headers["cache-control"] == "no-store"

    rejected = client.post(
        "/api/chat",
        content="invalid",
        headers={
            "Content-Type": "text/plain",
            "Origin": "https://untrusted.example",
        },
    )
    assert "access-control-allow-origin" not in rejected.headers
