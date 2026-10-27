def validate_request(payload: dict, max_batch: int) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
        
    images = payload.get("images")
    if not images or not isinstance(images, list):
        raise ValueError("'images' must be a non-empty list")
        
    if len(images) > max_batch:
        raise ValueError(f"Batch size exceeds maximum allowed ({max_batch})")
        
    validated_images = []
    for i, img in enumerate(images):
        if not isinstance(img, dict):
            raise ValueError(f"Image item at index {i} must be an object")
            
        img_id = img.get("id")
        if img_id is None:
            raise ValueError(f"Image item at index {i} is missing 'id'")
            
        b64 = img.get("b64")
        url = img.get("url")
        
        if not b64 and not url:
            raise ValueError(f"Image '{img_id}' must provide either 'b64' or 'url'")
            
        validated_images.append({
            "id": img_id,
            "b64": b64,
            "url": url
        })
        
    return validated_images
