import json

import numpy as np
import pytest
from PIL import Image

from app import config
from app.preprocess import Preprocessor

transformers = pytest.importorskip("transformers")


@pytest.fixture
def random_image():
    # Create a random RGB image of size 300x400
    img_array = np.random.default_rng(0).integers(0, 256, (400, 300, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


def test_preprocessing_parity(random_image):
    if not config.PREPROCESSOR_CONFIG_PATH.exists():
        pytest.skip("preprocessor_config.json not found. Run 'make export' first.")

    # Our Preprocessor
    our_preprocessor = Preprocessor(config_path=config.PREPROCESSOR_CONFIG_PATH)
    our_output = our_preprocessor.preprocess(random_image)

    # HF AutoImageProcessor built from the same exported config (no network needed)
    with open(config.PREPROCESSOR_CONFIG_PATH) as f:
        cfg = json.load(f)
    hf_processor = transformers.SiglipImageProcessor(**{k: v for k, v in cfg.items() if k != "image_processor_type"})
    hf_output = hf_processor(images=random_image, return_tensors="np")["pixel_values"][0]

    # Calculate max absolute difference
    max_abs_diff = np.max(np.abs(our_output - hf_output))

    print(f"Max absolute difference: {max_abs_diff}")

    assert max_abs_diff < 1e-4, f"Preprocessing parity failed! Max diff: {max_abs_diff}"
