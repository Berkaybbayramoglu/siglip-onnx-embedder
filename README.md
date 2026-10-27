# SigLIP ONNX Embedder

A lightweight, stateless image-embedding service built on **[SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256)** vision features. It runs on **AWS Lambda** (container image), as a plain **HTTP server**, or as a Python library.

At runtime there is **no PyTorch and no Transformers**. Inference uses only **ONNX Runtime + NumPy + Pillow**, which keeps the image small, cold starts short and CPU inference cheap. PyTorch is needed only once, at build time, to export the model.

```
image (base64 / URL) ──► decode ──► resize + normalize (NumPy) ──► ONNX Runtime (INT8) ──► L2-normalized 768-d vector
```

## Features

- **Pure ONNX inference.** The vision tower is exported to ONNX and dynamically quantized to INT8, with the few layers that quantize badly kept in FP32 automatically (see [Accuracy](#accuracy--performance)).
- **Matches Hugging Face preprocessing.** The PIL/NumPy pipeline agrees with `SiglipImageProcessor` to within ~1e-7, and a test enforces a 1e-4 tolerance.
- **Batching with per-image error isolation.** A broken image in a batch gets an error code, and the other images are still embedded.
- **Lambda-ready.** Accepts direct invocations, API Gateway (REST/HTTP), Lambda Function URL and ALB events, including `isBase64Encoded` bodies.
- **Any fixed-resolution SigLIP / SigLIP 2 checkpoint.** Export a different model with one flag. The response reports which model it serves.
- **Configured through environment variables.** No code changes are needed to deploy.

## Quick start (local)

Requires Python 3.10+.

```bash
git clone https://github.com/Berkaybbayramoglu/siglip-onnx-embedder.git
cd siglip-onnx-embedder

python3 -m venv .venv && source .venv/bin/activate

# CPU-only PyTorch is enough for exporting (much smaller download than the CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu
make install-dev

# Download the model from Hugging Face and export it to ONNX
# (~2-3 minutes plus a one-time ~400 MB download)
make export

# Run the test suite
make test

# Start a local HTTP server on :8080
make serve
```

```bash
curl -s -XPOST localhost:8080/embed \
  -d '{"images": [{"id": "photo-1", "url": "https://picsum.photos/id/237/400/300"}]}'
```

`make export` writes these files to `models/` (all of them are git-ignored):

| File | Purpose |
|---|---|
| `siglip-vision-int8.onnx` | Quantized model used at runtime (~110 MB) |
| `siglip-vision-fp32.onnx` (+ `.data`) | Full-precision model, used by the tests and available as an option (~370 MB) |
| `preprocessor_config.json` | Resize / normalization parameters from Hugging Face |
| `model_info.json` | Model id, embedding dimension and which layers were kept in FP32 |

## API

### Request

Each item needs an `id`, which is echoed back in the response, and **either** `b64` (raw base64 or a `data:image/...;base64,` URI) **or** `url` (http/https).

```json
{
  "images": [
    { "id": "img-1", "b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=" },
    { "id": "img-2", "url": "https://example.com/image.jpg" }
  ]
}
```

### Response

```json
{
  "model": "google/siglip2-base-patch16-256",
  "model_version": "1",
  "dim": 768,
  "normalized": true,
  "embeddings": [
    { "id": "img-1", "ok": true,  "vector": [0.0123, -0.0456, "..."], "error": null },
    { "id": "img-2", "ok": false, "error": "download_failed" }
  ]
}
```

Vectors are L2-normalized, so **cosine similarity is a plain dot product**.

The `error` field takes one of these values for a single image:

| Error | Meaning |
|---|---|
| `bad_input` | `b64`/`url` of the wrong type, an invalid or non-http(s) URL, or URL input disabled |
| `decode_failed` | Invalid base64, or the bytes are not an image Pillow can read |
| `download_failed` | Network error, timeout or non-200 status |
| `too_large` | The image exceeds `MAX_IMAGE_BYTES` or `MAX_IMAGE_PIXELS` |
| `internal_error` | Unexpected failure during inference |

These request-level errors return `400 {"error": "bad_input", "details": "..."}`:

- invalid JSON
- a missing or empty `images` list
- more than `MAX_BATCH` images
- an item that is not an object, has no `id`, or has neither `b64` nor `url`

If the model failed to load, the service returns `500`.

When the event comes from API Gateway, a Function URL or an ALB, the response is wrapped as `{"statusCode", "headers", "body"}`. A direct invocation returns the JSON object above on success. Request-level errors are always returned in the wrapped form, so check for `statusCode` in the result.

## Configuration

All settings are environment variables. Relative paths resolve against the repository root.

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `models` | Directory containing the exported files |
| `MODEL_PATH` | `$MODEL_DIR/siglip-vision-int8.onnx` | ONNX model to serve. Point it at `siglip-vision-fp32.onnx` for full precision |
| `PREPROCESSOR_CONFIG_PATH` | `$MODEL_DIR/preprocessor_config.json` | Preprocessing config |
| `MODEL_NAME` | from `model_info.json` | Model name reported in responses |
| `MODEL_VERSION` | `1` | Version string reported in responses |
| `MAX_BATCH` | `16` | Maximum images per request |
| `MAX_IMAGE_BYTES` | `10485760` (10 MB) | Maximum size per image (base64-decoded or downloaded) |
| `MAX_IMAGE_PIXELS` | `40000000` | Maximum width × height per image, as protection against decompression bombs |
| `DOWNLOAD_TIMEOUT` | `10` | Total seconds allowed for fetching a URL |
| `ALLOW_URLS` | `true` | Set to `false` to accept base64 input only |
| `OMP_NUM_THREADS` | CPU count | ONNX Runtime intra-op threads |
| `LOG_LEVEL` | `INFO` | Log level (logs are single-line JSON) |
| `MAX_CONCURRENT_REQUESTS` | `2` | `scripts/serve.py` only: requests processed at the same time |

### Using a different model

Any fixed-resolution SigLIP or SigLIP 2 checkpoint on the Hugging Face Hub, or a local fine-tuned copy, can be exported. The variable-resolution `*-naflex` variants are not supported.

```bash
make export MODEL_ID=google/siglip2-large-patch16-256
# or
python scripts/export_model.py --model-id /path/to/my-finetuned-siglip --output-dir models
```

The input resolution and embedding dimension are detected automatically. Other export options:

- `--keep-fp32 N`: how many quantization-sensitive layers stay in FP32 (default `1`; `0` quantizes everything).
- `--calib-dir DIR`: a folder of your own sample images, used to choose those layers. Without it, synthetic images are used.

## Deploying to AWS Lambda

The `Dockerfile` builds an image on the official AWS Lambda Python 3.12 base image. It contains only the INT8 model; the FP32 model and dev tools are left out. Run `make export` before building.

**Try it locally with the Lambda runtime emulator:**

```bash
make run-local     # builds the image and serves it on :9000

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"images": [{"id": "test-1", "url": "https://picsum.photos/256"}]}'
```

**Push the image to ECR and create the function.** Replace the placeholders with your own values:

```bash
export AWS_REGION=<your-region>                # e.g. eu-central-1
export AWS_ACCOUNT_ID=<your-account-id>
export ECR_REPO=siglip-embedder
export IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build --platform linux/amd64 --provenance=false -t $IMAGE_URI .
docker push $IMAGE_URI

aws lambda create-function \
  --function-name siglip-embedder \
  --package-type Image \
  --code ImageUri=$IMAGE_URI \
  --role arn:aws:iam::$AWS_ACCOUNT_ID:role/<your-lambda-execution-role> \
  --memory-size 3008 \
  --timeout 60 \
  --region $AWS_REGION
```

Notes:

- Lambda allocates CPU in proportion to memory. **2–3 GB** is a good starting point for this model.
- To expose the function over HTTP, add a Function URL (`aws lambda create-function-url-config --auth-type AWS_IAM ...`) or put API Gateway in front of it. **Keep authentication enabled.** An open endpoint lets anyone run inference on your bill.
- To run outside Lambda (on ECS/Fargate, Kubernetes or a VM), use `scripts/serve.py`, or wrap `app.handler.handler` in your own HTTP framework.

## Accuracy & performance

Measured for `google/siglip2-base-patch16-256` on 56 real photos (640×480):

| Model | Size | Cosine vs FP32 (mean / min) | Same nearest neighbour as FP32 | Single-image latency (p50) |
|---|---|---|---|---|
| FP32 | 372 MB | 1.000 / 1.000 | 100 % | ~220 ms |
| INT8, all MatMuls quantized | 103 MB | 0.855 / 0.768 | 58 % | ~110 ms |
| **INT8, 1 sensitive layer in FP32 (default)** | **110 MB** | **0.985 / 0.964** | **84 %** | **~130 ms** |

Latency is for ONNX Runtime on a desktop CPU (Intel i9-14900HX). Expect different absolute numbers on Lambda.

**Why one layer in FP32?** Plain dynamic INT8 quantization badly damages SigLIP 2 embeddings. The cause is a single MLP down-projection (block 5) whose input activations contain extreme outliers. `export_model.py` finds such layers automatically by measuring the effect of each one and keeps the worst ones in FP32. This brings the embeddings close to FP32 quality for +7 MB.

If you need exact FP32 embeddings, for example because you are mixing vectors with ones computed in PyTorch, set `MODEL_PATH=models/siglip-vision-fp32.onnx`.

> Dynamic quantization computes activation scales per batch. As a result, the INT8 vector for an image can shift very slightly (cosine > 0.98) depending on which other images share its batch. The FP32 model is exactly batch-invariant.

## Tests

```bash
make test
```

| Test file | What it checks |
|---|---|
| `test_schema.py` | Request validation |
| `test_handler.py` | Error codes, data URIs, size and pixel limits, URL scheme filtering, malformed JSON, and the API Gateway / base64-encoded event formats |
| `test_embeddings.py` | Output shape, unit norm, determinism and dynamic batch support |
| `test_preprocess_parity.py` | NumPy preprocessing vs Hugging Face `SiglipImageProcessor` |
| `test_quantization_parity.py` | INT8 vs FP32 cosine similarity |

Tests that need the exported model are skipped when it is missing, so the suite also runs in a minimal environment (`pip install -r requirements.txt pytest`).

## Project structure

```
app/
  config.py        environment-based configuration
  handler.py       Lambda entry point: parsing, validation, fetching, batching
  model.py         ONNX Runtime session + L2 normalization
  preprocess.py    PIL/NumPy port of the SigLIP image processor
  schema.py        request validation
scripts/
  export_model.py  HF -> ONNX (FP32) -> INT8, with sensitive-layer search
  serve.py         dependency-free HTTP server (POST /embed, GET /health)
  benchmark.py     latency / throughput micro-benchmark
tests/
Dockerfile         AWS Lambda container image
```

## Security notes

- URL inputs make the service fetch arbitrary http(s) URLs. If it runs inside a private network, this can be abused for SSRF. Set `ALLOW_URLS=false`, or restrict egress at the network level, when that matters.
- Image size is limited by `MAX_IMAGE_BYTES` and `MAX_IMAGE_PIXELS`, and the number of images per request by `MAX_BATCH`.
- `scripts/serve.py` is a minimal server built on the standard library. For internet-facing deployments, put it behind a reverse proxy, or use Lambda.

## License

The code is released under the [MIT License](LICENSE).

The model weights are **not** included in this repository. They are downloaded from Hugging Face during `make export` and are subject to their own license (`google/siglip2-*` is released under Apache 2.0).
