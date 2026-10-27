import json
import numpy as np
from PIL import Image

class Preprocessor:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.image_mean = np.array(self.config.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.image_std = np.array(self.config.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        
        # Determine target size
        size = self.config.get("size", {"height": 256, "width": 256})
        if isinstance(size, int):
            self.target_size = (size, size)
        elif isinstance(size, dict) and "height" in size and "width" in size:
            # PIL resize expects (width, height)
            self.target_size = (size["width"], size["height"])
        elif isinstance(size, dict) and "shortest_edge" in size:
            # SigLIP resizes to a square without keeping the aspect ratio
            self.target_size = (size["shortest_edge"], size["shortest_edge"])
        elif isinstance(size, (list, tuple)):
            self.target_size = (size[1], size[0])  # Assuming [height, width]
        else:
            raise ValueError(f"Unsupported 'size' in preprocessor config: {size!r}")
            
        # Determine resample method
        resample_mapping = {
            1: Image.Resampling.LANCZOS,
            2: Image.Resampling.BILINEAR,
            3: Image.Resampling.BICUBIC,
            4: Image.Resampling.BOX,
            5: Image.Resampling.NEAREST,
        }
        resample_val = self.config.get("resample", 3)
        self.resample = resample_mapping.get(resample_val, Image.Resampling.BICUBIC)
        
        self.rescale_factor = self.config.get("rescale_factor", 1 / 255.0)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesses a single PIL Image to a NumPy array for ONNX inference.
        Output shape: (3, H, W)
        """
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Resize
        image = image.resize(self.target_size, resample=self.resample)
        
        # Convert to numpy array with float64 to match HF potential precision
        img_array = np.array(image, dtype=np.float64)
        
        # Rescale
        img_array *= self.rescale_factor
        
        # Normalize
        img_array = (img_array - self.image_mean) / self.image_std
        
        # Change data layout from HWC to CHW
        img_array = img_array.transpose(2, 0, 1)
        
        return img_array.astype(np.float32)

    def preprocess_batch(self, images: list[Image.Image]) -> np.ndarray:
        """
        Preprocesses a list of PIL Images to a NumPy array for ONNX inference.
        Output shape: (B, 3, H, W)
        """
        if not images:
            return np.array([], dtype=np.float32)
            
        processed_images = [self.preprocess(img) for img in images]
        return np.stack(processed_images, axis=0)
