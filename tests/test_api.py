import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeEngine, result, tone


def read_until(ws, wanted):
    seen = []
    while True:
        message = ws.receive_json()
        seen.append(message)
        if message["type"] == wanted:
            return seen


def test_health_ok_when_model_loaded(settings):
    with TestClient(create_app(FakeEngine(), settings)) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_health_503_when_model_missing(settings):
    engine = FakeEngine()
    engine.loaded = False
    with TestClient(create_app(engine, settings)) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "error"


def test_websocket_full_session(settings):
    engine = FakeEngine([
        result((0.0, 1.5, "hello from the test"), (2.0, 3.0, "and more")),
        result((0.0, 1.0, "and more")),
    ])
    with TestClient(create_app(engine, settings)) as client:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "config", "language": "en"})
            assert ws.receive_json()["type"] == "ready"

            ws.send_bytes(tone(3.0))
            first = read_until(ws, "partial")
            assert any(m["type"] == "final" and m["text"] == "hello from the test" for m in first)

            ws.send_json({"type": "stop"})
            tail = read_until(ws, "done")
            assert tail[-1]["fullText"] == "hello from the test and more"


def test_websocket_rejects_unsupported_language(settings):
    with TestClient(create_app(FakeEngine(), settings)) as client:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "config", "language": "xx"})
            message = ws.receive_json()
            assert message["type"] == "error" and message["fatal"] is True


def test_websocket_requires_config_first(settings):
    with TestClient(create_app(FakeEngine(), settings)) as client:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_json({"type": "stop"})
            assert ws.receive_json()["type"] == "error"


def test_websocket_reports_unloaded_model(settings):
    engine = FakeEngine()
    engine.loaded = False
    with TestClient(create_app(engine, settings)) as client:
        with client.websocket_connect("/ws/transcribe") as ws:
            message = ws.receive_json()
            assert message["type"] == "error" and message["fatal"] is True
