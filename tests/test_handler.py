import base64
import json

from app import config
from app.handler import handler
from tests.conftest import make_b64_image, requires_model


def test_handler_bad_input():
    resp = handler({"images": [{"id": "1"}]}, None)  # missing b64 or url
    assert resp["statusCode"] == 400


def test_handler_deeply_nested_json():
    resp = handler({"body": "[" * 100000}, None)
    assert resp["statusCode"] == 400


def test_handler_invalid_json_body():
    resp = handler({"body": "{not json"}, None)
    assert resp["statusCode"] == 400
    assert json.loads(resp["body"])["error"] == "bad_input"


@requires_model
def test_handler_decode_failed():
    resp = handler({"images": [{"id": "bad-1", "b64": "not-an-image"}]}, None)
    emb = resp["embeddings"][0]
    assert emb["id"] == "bad-1"
    assert emb["ok"] is False
    assert emb["error"] == "decode_failed"


@requires_model
def test_handler_rejects_non_http_urls():
    resp = handler({"images": [{"id": "f", "url": "file:///etc/passwd"}]}, None)
    assert resp["embeddings"][0]["error"] == "bad_input"


@requires_model
def test_handler_too_large(monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", 10)
    resp = handler({"images": [{"id": "big", "b64": make_b64_image(size=(64, 64))}]}, None)
    assert resp["embeddings"][0]["error"] == "too_large"


@requires_model
def test_handler_success(b64_image):
    event = {
        "images": [
            {"id": "good-1", "b64": b64_image},
            {"id": "good-2", "b64": "data:image/jpeg;base64," + b64_image},
            {"id": "bad-1", "b64": "invalid-b64"},
        ]
    }
    resp = handler(event, None)

    assert resp["dim"] == len(resp["embeddings"][0]["vector"])
    assert resp["normalized"] is True
    good, data_uri, bad = resp["embeddings"]
    assert good["ok"] is True and good["error"] is None
    assert data_uri["ok"] is True
    assert good["vector"] == data_uri["vector"]
    assert bad["ok"] is False and bad["error"] == "decode_failed"


@requires_model
def test_handler_api_gateway_event(b64_image):
    payload = json.dumps({"images": [{"id": "gw", "b64": b64_image}]})
    for event in (
        {"body": payload},
        {"body": base64.b64encode(payload.encode()).decode(), "isBase64Encoded": True},
    ):
        resp = handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["embeddings"][0]["ok"] is True


@requires_model
def test_handler_decompression_bomb(monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 100 * 100)
    resp = handler({"images": [
        {"id": "huge", "b64": make_b64_image(size=(200, 200), fmt="PNG")},
        {"id": "ok", "b64": make_b64_image(size=(50, 50), fmt="PNG")},
    ]}, None)
    huge, ok = resp["embeddings"]
    assert huge["error"] == "too_large"
    assert ok["ok"] is True


@requires_model
def test_handler_malformed_url():
    resp = handler({"images": [{"id": "u", "url": "http://[::1"}]}, None)
    assert resp["embeddings"][0]["error"] == "bad_input"
