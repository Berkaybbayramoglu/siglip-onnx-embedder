import json
import os
from pathlib import Path

# Repository root (parent of the `app/` package). Relative paths are resolved
# against it so the service works regardless of the current working directory.
ROOT_DIR = Path(__file__).resolve().parent.parent


def _path(env_name: str, default: str) -> Path:
    p = Path(os.environ.get(env_name, default))
    return p if p.is_absolute() else ROOT_DIR / p


MODEL_DIR = _path("MODEL_DIR", "models")
MODEL_PATH = _path("MODEL_PATH", str(MODEL_DIR / "siglip-vision-int8.onnx"))
PREPROCESSOR_CONFIG_PATH = _path("PREPROCESSOR_CONFIG_PATH", str(MODEL_DIR / "preprocessor_config.json"))
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

MAX_BATCH = int(os.environ.get("MAX_BATCH", 16))
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 10 * 1024 * 1024))
# Guards against decompression bombs: a tiny PNG can decode to hundreds of megapixels
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 40_000_000))
DOWNLOAD_TIMEOUT = float(os.environ.get("DOWNLOAD_TIMEOUT", 10))  # total seconds per URL
OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
ALLOW_URLS = os.environ.get("ALLOW_URLS", "true").lower() in ("1", "true", "yes")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def _model_info() -> dict:
    # Written by scripts/export_model.py so the response reports the model that was actually exported.
    try:
        with open(MODEL_INFO_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_info = _model_info()
MODEL_NAME = os.environ.get("MODEL_NAME", _info.get("model_id", "google/siglip2-base-patch16-256"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", str(_info.get("model_version", "1")))
