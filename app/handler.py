import base64
import binascii
import json
import logging
import time
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from app import config
from app.schema import validate_request
from app.preprocess import Preprocessor
from app.model import SigLIPModel

# Configure logging
logger = logging.getLogger()
logger.setLevel(config.LOG_LEVEL)


def json_log(level: int, event: str, **kwargs):
    log_record = {"event": event}
    log_record.update(kwargs)
    logger.log(level, json.dumps(log_record))


# Header check in process_image() is the real limit; this is a backstop for formats PIL sizes lazily
Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

# Initialize globally for cold start efficiency
preprocessor = None
model = None
GLOBAL_INIT_ERROR = None
try:
    preprocessor = Preprocessor(config.PREPROCESSOR_CONFIG_PATH)
    model = SigLIPModel(config.MODEL_PATH, num_threads=config.OMP_NUM_THREADS)
except Exception as e:
    json_log(logging.ERROR, "global_init_failed", error=str(e))
    GLOBAL_INIT_ERROR = str(e)


def _decode_b64(b64: str) -> bytes:
    # Accept data URIs such as "data:image/png;base64,...."
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _download(url: str) -> tuple[bytes | None, str | None]:
    try:
        if urlparse(url).scheme not in ("http", "https"):
            return None, "bad_input"
    except ValueError:
        return None, "bad_input"

    deadline = time.monotonic() + config.DOWNLOAD_TIMEOUT
    with requests.get(url, timeout=config.DOWNLOAD_TIMEOUT, stream=True) as resp:
        if resp.status_code != 200:
            return None, "download_failed"

        # Check content length if available
        cl = resp.headers.get("Content-Length")
        if cl and cl.isdigit() and int(cl) > config.MAX_IMAGE_BYTES:
            return None, "too_large"

        # Read at most MAX_IMAGE_BYTES + 1 so oversized bodies are never fully buffered
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            buf.extend(chunk)
            if len(buf) > config.MAX_IMAGE_BYTES:
                return None, "too_large"
            # requests' timeout applies per socket read; also enforce a total deadline
            if time.monotonic() > deadline:
                return None, "download_failed"
        return bytes(buf), None


def process_image(img_req: dict) -> tuple[Image.Image | None, str | None]:
    """
    Returns (PIL.Image, None) on success, or (None, error_enum) on failure.
    Error enums: download_failed | decode_failed | too_large | bad_input | internal_error
    """
    b64 = img_req.get("b64")
    url = img_req.get("url")

    img_bytes = None
    try:
        if b64:
            if not isinstance(b64, str):
                return None, "bad_input"
            try:
                img_bytes = _decode_b64(b64)
            except (binascii.Error, ValueError) as e:
                json_log(logging.ERROR, "decode_failed", id=img_req.get("id"), error=str(e))
                return None, "decode_failed"
            if len(img_bytes) > config.MAX_IMAGE_BYTES:
                return None, "too_large"
        elif url:
            if not isinstance(url, str):
                return None, "bad_input"
            if not config.ALLOW_URLS:
                return None, "bad_input"
            img_bytes, err = _download(url)
            if err:
                return None, err
        else:
            return None, "bad_input"
    except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema):
        return None, "bad_input"
    except requests.RequestException as e:
        json_log(logging.ERROR, "download_error", id=img_req.get("id"), error=str(e))
        return None, "download_failed"
    except Exception as e:
        json_log(logging.ERROR, "internal_error", id=img_req.get("id"), error=str(e))
        return None, "internal_error"

    try:
        image = Image.open(BytesIO(img_bytes))
        # Check dimensions from the header before decoding pixel data
        if image.width * image.height > config.MAX_IMAGE_PIXELS:
            return None, "too_large"
        image.load()  # Verify it's actually an image
        return image, None
    except Image.DecompressionBombError:
        return None, "too_large"
    except UnidentifiedImageError:
        json_log(logging.ERROR, "decode_failed", id=img_req.get("id"), reason="UnidentifiedImageError")
        return None, "decode_failed"
    except Exception as e:
        json_log(logging.ERROR, "decode_failed", id=img_req.get("id"), error=str(e))
        return None, "decode_failed"


def _is_http_event(event) -> bool:
    # API Gateway (REST/HTTP API), Lambda Function URLs and ALB put the JSON payload in "body"
    return isinstance(event, dict) and "body" in event and "images" not in event


def _parse_body(event):
    if _is_http_event(event):
        body = event.get("body") or ""
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        return json.loads(body) if isinstance(body, str) else body
    if isinstance(event, (str, bytes)):
        return json.loads(event)
    return event


def _error(status: int, error: str, details: str) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "isBase64Encoded": False,
        "body": json.dumps({"error": error, "details": details}),
    }


def handler(event, context):
    start_time = time.time()

    # Parse event body
    try:
        body = _parse_body(event)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return _error(400, "bad_input", "Invalid JSON")

    # Validate
    try:
        requests_list = validate_request(body, config.MAX_BATCH)
    except ValueError as e:
        return _error(400, "bad_input", str(e))

    if GLOBAL_INIT_ERROR:
        json_log(logging.ERROR, "invocation_failed", reason="global_init_error")
        return _error(500, "internal_error", GLOBAL_INIT_ERROR)

    valid_images = []
    valid_indices = []
    results = [{"id": req["id"], "ok": False, "error": "internal_error"} for req in requests_list]

    # Process inputs
    for idx, req in enumerate(requests_list):
        img, err = process_image(req)
        if err:
            results[idx]["error"] = err
        else:
            valid_images.append(img)
            valid_indices.append(idx)

    # Batch inference
    if valid_images:
        try:
            pixel_values = preprocessor.preprocess_batch(valid_images)
            embeddings = model.encode(pixel_values)

            for i, idx in enumerate(valid_indices):
                results[idx] = {
                    "id": requests_list[idx]["id"],
                    "ok": True,
                    "vector": embeddings[i].tolist(),
                    "error": None
                }
        except Exception as e:
            json_log(logging.ERROR, "inference_error", error=str(e))
            for idx in valid_indices:
                results[idx]["error"] = "internal_error"
                results[idx]["ok"] = False

    duration_ms = (time.time() - start_time) * 1000

    # Log the summary for this invocation
    json_log(logging.INFO, "invocation_complete",
             batch_size=len(requests_list),
             successful=sum(1 for r in results if r["ok"]),
             duration_ms=round(duration_ms, 2))

    response_body = {
        "model": config.MODEL_NAME,
        "model_version": config.MODEL_VERSION,
        "dim": model.dim,
        "normalized": True,
        "embeddings": results
    }

    # If API Gateway proxy format, wrap in statusCode/body
    if _is_http_event(event):
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "isBase64Encoded": False,
            "body": json.dumps(response_body)
        }

    # Raw invocation format
    return response_body
