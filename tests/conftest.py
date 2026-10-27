import base64
from io import BytesIO

import pytest
from PIL import Image

from app import config

requires_model = pytest.mark.skipif(
    not config.MODEL_PATH.exists() or not config.PREPROCESSOR_CONFIG_PATH.exists(),
    reason="ONNX model not found. Run 'make export' first.",
)


def make_b64_image(color=(255, 0, 0), size=(10, 10), fmt="JPEG"):
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def b64_image():
    return make_b64_image()
