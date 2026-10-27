import time
import base64
from io import BytesIO
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.handler import handler  # noqa: E402

def create_b64_image():
    img = Image.new('RGB', (256, 256), color='blue')
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def benchmark():
    b64_img = create_b64_image()
    
    # Warmup
    print("Warming up...")
    event_single = {"images": [{"id": "1", "b64": b64_img}]}
    handler(event_single, None)
    
    # Single image benchmark
    print("Running single image benchmark...")
    single_times = []
    for _ in range(10):
        start = time.time()
        handler(event_single, None)
        single_times.append(time.time() - start)
        
    single_times.sort()
    p50 = single_times[len(single_times)//2]
    p95 = single_times[int(len(single_times)*0.95)]
    print(f"Single image latency - p50: {p50:.3f}s, p95: {p95:.3f}s")
    
    # Batch benchmark
    print("Running batch benchmark (batch_size=16)...")
    event_batch = {"images": [{"id": str(i), "b64": b64_img} for i in range(16)]}
    
    start = time.time()
    handler(event_batch, None)
    batch_time = time.time() - start
    
    throughput = 16 / batch_time
    print(f"Batch inference time: {batch_time:.3f}s")
    print(f"Throughput: {throughput:.2f} images/sec")

if __name__ == "__main__":
    benchmark()
