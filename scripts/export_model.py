"""Export the vision tower of a SigLIP / SigLIP2 checkpoint to ONNX (FP32 + dynamic INT8)."""
import argparse
import glob
import json
import logging
import os
import sys
import tempfile
import warnings

import numpy as np
import onnx
import onnxruntime as ort

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from onnxruntime.quantization import quantize_dynamic, QuantType
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from app.preprocess import Preprocessor  # noqa: E402

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-256"

# onnxruntime's quantizer logs a "consider pre-processing" hint on every call; torch warns that
# dynamic_axes is legacy (it is kept because it works on both old and new exporters)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*dynamic_axes.*")


class SigLIPVisionWrapper(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Load the model
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values):
        # We only want to export get_image_features
        out = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        return out


def export_config(model_id, config_path):
    print("Exporting preprocessor_config.json...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    config_dict = processor.to_dict()
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {config_path}")
    return config_dict


def export_to_onnx(model_id, fp32_path, height, width):
    print(f"Loading model {model_id}...")
    wrapper = SigLIPVisionWrapper(model_id)
    wrapper.eval()

    # SigLIP expects input of shape [batch_size, 3, height, width]
    # Batch of 2: torch.export specializes size-1 dims, which would freeze the batch axis to 1
    dummy_input = torch.randn(2, 3, height, width)
    with torch.no_grad():
        dim = wrapper(dummy_input).shape[-1]

    print("Exporting to ONNX (FP32)...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        fp32_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['image_embeds'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'image_embeds': {0: 'batch_size'}
        }
    )
    print(f"FP32 model saved to {fp32_path}")
    return int(dim)


def quantize_model(fp32_path, int8_path, nodes_to_exclude=()):
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=['MatMul'],
        per_channel=True,
        nodes_to_exclude=list(nodes_to_exclude)
    )


def calibration_batch(config_path, calib_dir=None, n=8):
    """Images used to measure quantization error: files from calib_dir, or synthetic gradients."""
    images = []
    if calib_dir:
        for f in sorted(glob.glob(os.path.join(calib_dir, "*")))[:32]:
            try:
                images.append(Image.open(f).convert("RGB"))
            except Exception:
                pass
    if not images:
        rng = np.random.default_rng(0)
        ramp = np.linspace(0, 255, 256)
        for _ in range(n):
            channels = [np.add.outer(ramp * rng.random(), ramp * rng.random()) % 256 for _ in range(3)]
            images.append(Image.fromarray(np.stack(channels, -1).astype(np.uint8)))
        images.append(Image.new("RGB", (256, 256), (128, 128, 128)))
    return Preprocessor(config_path).preprocess_batch(images)


def _embed(model_path, pixel_values):
    out = ort.InferenceSession(model_path, providers=['CPUExecutionProvider']).run(None, {'pixel_values': pixel_values})[0]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def find_sensitive_nodes(fp32_path, pixel_values, keep):
    """
    Dynamic INT8 quantization of a single MLP down-projection (fc2) can wreck ViT embeddings because
    of a few huge activation outliers. Quantize the model once per fc2 MatMul with only that node
    left in FP32 and return the `keep` nodes whose exclusion recovers the most cosine similarity.
    """
    graph = onnx.load(fp32_path, load_external_data=False).graph
    shapes = {i.name: list(i.dims) for i in graph.initializer}
    fc2 = []
    for node in graph.node:
        w = [shapes[i] for i in node.input if i in shapes]
        if node.op_type == "MatMul" and w and len(w[0]) == 2 and w[0][0] > w[0][1]:
            fc2.append(node.name)

    reference = _embed(fp32_path, pixel_values)
    scores = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = os.path.join(tmp, "probe.onnx")
        for i, name in enumerate(fc2):
            quantize_model(fp32_path, tmp_path, [name])
            cos = float((_embed(tmp_path, pixel_values) * reference).sum(1).mean())
            print(f"  [{i + 1}/{len(fc2)}] keep {name} in FP32 -> mean cosine {cos:.4f}")
            scores.append((cos, name))
    return [name for _, name in sorted(scores, reverse=True)[:keep]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
                        help=f"Hugging Face model id or local path (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--output-dir", default="models", help="Where to write the ONNX files (default: models)")
    parser.add_argument("--keep-fp32", type=int, default=1,
                        help="Number of quantization-sensitive MLP layers to keep in FP32 (default: 1, 0 = quantize all)")
    parser.add_argument("--calib-dir", default=None,
                        help="Optional folder of sample images used to pick sensitive layers (default: synthetic images)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fp32_path = os.path.join(args.output_dir, "siglip-vision-fp32.onnx")
    int8_path = os.path.join(args.output_dir, "siglip-vision-int8.onnx")
    config_path = os.path.join(args.output_dir, "preprocessor_config.json")

    cfg = export_config(args.model_id, config_path)
    size = cfg.get("size") or {}
    if isinstance(size, int):
        size = {"height": size, "width": size}
    height = size.get("height") or size.get("shortest_edge") or 256
    width = size.get("width") or size.get("shortest_edge") or 256

    dim = export_to_onnx(args.model_id, fp32_path, height, width)

    pixel_values = calibration_batch(config_path, args.calib_dir)
    excluded = []
    if args.keep_fp32 > 0:
        print("Searching for quantization-sensitive layers...")
        excluded = find_sensitive_nodes(fp32_path, pixel_values, args.keep_fp32)
        print(f"Keeping in FP32: {excluded}")

    print("Quantizing to INT8...")
    quantize_model(fp32_path, int8_path, excluded)
    cos = (_embed(int8_path, pixel_values) * _embed(fp32_path, pixel_values)).sum(1)
    print(f"INT8 model saved to {int8_path} (cosine vs FP32 on calibration set: min {cos.min():.4f}, mean {cos.mean():.4f})")

    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump({"model_id": args.model_id, "model_version": "1", "dim": dim,
                   "image_size": [height, width], "fp32_nodes": excluded}, f, indent=2)
    print(f"Export complete ({args.model_id}, dim={dim}).")


if __name__ == "__main__":
    main()
