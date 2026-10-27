import numpy as np
import pytest
from PIL import Image

from app import config
from app.model import SigLIPModel
from app.preprocess import Preprocessor
from tests.conftest import requires_model

FP32_PATH = config.MODEL_DIR / "siglip-vision-fp32.onnx"


def _images():
    rng = np.random.default_rng(0)
    return [
        Image.new("RGB", (256, 256), color=(128, 128, 128)),
        Image.new("RGB", (300, 200), color=(0, 0, 255)),
        Image.fromarray(rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)),
    ]


@requires_model
def test_embedding_shape_norm_and_determinism():
    preprocessor = Preprocessor(config.PREPROCESSOR_CONFIG_PATH)
    model = SigLIPModel(config.MODEL_PATH)

    pixel_values = preprocessor.preprocess_batch(_images())
    assert pixel_values.shape[:2] == (3, 3)

    vectors = model.encode(pixel_values)
    assert vectors.shape == (3, model.dim)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(vectors, model.encode(pixel_values))

    # Dynamic INT8 quantization computes activation scales per batch, so batching may shift
    # a vector slightly, but it must stay essentially the same embedding.
    for i in range(3):
        single = model.encode(pixel_values[i:i + 1])[0]
        assert float(single @ vectors[i]) > 0.97


@pytest.mark.skipif(not FP32_PATH.exists(), reason="FP32 model not found. Run 'make export' first.")
def test_fp32_supports_dynamic_batch():
    preprocessor = Preprocessor(config.PREPROCESSOR_CONFIG_PATH)
    model = SigLIPModel(FP32_PATH)

    pixel_values = preprocessor.preprocess_batch(_images())
    batch = model.encode(pixel_values)
    singles = np.stack([model.encode(pixel_values[i:i + 1])[0] for i in range(3)])
    np.testing.assert_allclose(batch, singles, atol=1e-4)
