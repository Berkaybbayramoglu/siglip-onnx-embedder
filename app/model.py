import os
import numpy as np
import onnxruntime as ort


class SigLIPModel:
    def __init__(self, model_path, num_threads=None):
        model_path = str(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run 'make export' first.")

        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = num_threads

        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        # Determine model dim from output shape if possible, otherwise hardcode
        output_shape = self.session.get_outputs()[0].shape
        self.dim = output_shape[1] if len(output_shape) > 1 and isinstance(output_shape[1], int) else 768

    def encode(self, pixel_values: np.ndarray) -> np.ndarray:
        """
        Runs the ONNX model on a batch of preprocessed images.
        pixel_values shape: (B, 3, H, W)
        Returns L2 normalized embeddings of shape (B, dim)
        """
        if pixel_values.size == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self.session.run(None, {self.input_name: pixel_values})[0]

        # L2 Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-12, norms)
        return embeddings / norms
