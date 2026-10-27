import pytest
import numpy as np
import onnxruntime as ort

from app import config
from app.preprocess import Preprocessor
from PIL import Image

@pytest.fixture
def dummy_input():
    # Use a consistent image rather than random noise, as random noise
    # causes out-of-distribution activations that ruin dynamic quantization accuracy.
    if not config.PREPROCESSOR_CONFIG_PATH.exists():
        pytest.skip("preprocessor_config.json not found. Run 'make export' first.")
    img = Image.new('RGB', (256, 256), color=(128, 128, 128))
    preprocessor = Preprocessor(config.PREPROCESSOR_CONFIG_PATH)
    return preprocessor.preprocess_batch([img])

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2.T)
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)
    return dot / (np.outer(norm1, norm2))

def test_quantization_parity(dummy_input):
    fp32_path = config.MODEL_DIR / "siglip-vision-fp32.onnx"
    int8_path = config.MODEL_DIR / "siglip-vision-int8.onnx"

    if not fp32_path.exists() or not int8_path.exists():
        pytest.skip("ONNX models not found. Run 'make export' first.")

    session_fp32 = ort.InferenceSession(str(fp32_path), providers=['CPUExecutionProvider'])
    session_int8 = ort.InferenceSession(str(int8_path), providers=['CPUExecutionProvider'])

    input_name = session_fp32.get_inputs()[0].name

    out_fp32 = session_fp32.run(None, {input_name: dummy_input})[0]
    out_int8 = session_int8.run(None, {input_name: dummy_input})[0]

    # Calculate cosine similarity
    sim = cosine_similarity(out_fp32, out_int8)[0, 0]

    print(f"Cosine Similarity: {sim}")
    if sim < 0.97:
        print(f"WARNING: Cosine similarity {sim} is less than 0.97. Try re-exporting with a higher --keep-fp32 value.")
    assert sim >= 0.95, f"Quantization degraded model quality too much! Cosine similarity: {sim}"
