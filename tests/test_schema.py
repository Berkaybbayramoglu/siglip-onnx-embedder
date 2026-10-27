import pytest

from app.schema import validate_request


def test_valid_request():
    out = validate_request({"images": [{"id": "a", "b64": "x"}, {"id": 2, "url": "http://x"}]}, 16)
    assert [i["id"] for i in out] == ["a", 2]


@pytest.mark.parametrize("payload", [
    None,
    [],
    {},
    {"images": []},
    {"images": "nope"},
    {"images": ["nope"]},
    {"images": [{"b64": "x"}]},
    {"images": [{"id": "1"}]},
])
def test_invalid_requests(payload):
    with pytest.raises(ValueError):
        validate_request(payload, 16)


def test_batch_limit():
    with pytest.raises(ValueError, match="maximum"):
        validate_request({"images": [{"id": i, "b64": "x"} for i in range(3)]}, 2)
