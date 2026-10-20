"""Image generation tool.

Provides an image generation tool with multiple backends (OpenAI DALL-E, FAL.ai FLUX, local Stable Diffusion placeholder).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

IMAGE_GEN_SCHEMA = {
    "description": (
        "Generate images from a text prompt using various backends. "
        "Images are saved to the specified output directory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt to generate images from.",
            },
            "size": {
                "type": "string",
                "description": "The size of the generated image (e.g., '1024x1024').",
                "default": "1024x1024",
                "enum": ["256x256", "512x512", "1024x1024"],
            },
            "backend": {
                "type": "string",
                "description": "The image generation backend to use.",
                "default": "openai",
                "enum": ["openai", "fal", "local"],
            },
            "num_images": {
                "type": "integer",
                "description": "The number of images to generate.",
                "default": 1,
                "minimum": 1,
                "maximum": 4,
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to save generated images.",
                "default": ".mullet/images",
            },
        },
        "required": ["prompt"],
    },
}


def _save_image(image_url: str, output_dir: Path, prompt_hash: str) -> Path:
    """Helper to download and save an image."""
    response = httpx.get(image_url)
    response.raise_for_status()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"image_{prompt_hash}_{timestamp}.png"
    filepath = output_dir / filename
    filepath.write_bytes(response.content)
    return filepath


def image_gen_handler(
    prompt: str,
    size: str = "1024x1024",
    backend: str = "openai",
    num_images: int = 1,
    output_dir: str = ".mullet/images",
    **kwargs,
) -> str:
    """Tool handler for image generation."""
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    generated_images: list[str] = []
    success = False
    error_message = ""
    prompt_hash = str(abs(hash(prompt)))[:8]

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            error_message = "OPENAI_API_KEY not found in environment variables."
        else:
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "prompt": prompt,
                "n": num_images,
                "size": size,
                "response_format": "url",
            }
            try:
                response = httpx.post(
                    "https://api.openai.com/v1/images/generations",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                for img_data in data.get("data", []):
                    image_url = img_data.get("url")
                    if image_url:
                        filepath = _save_image(image_url, output_path, prompt_hash)
                        generated_images.append(str(filepath))
                success = bool(generated_images)
            except httpx.HTTPStatusError as e:
                error_message = f"OpenAI API error: {e.response.text}"
            except httpx.RequestError as e:
                error_message = f"OpenAI API request failed: {e}"

    elif backend == "fal":
        fal_key = os.environ.get("FAL_KEY")
        if not fal_key:
            error_message = "FAL_KEY not found in environment variables."
        else:
            headers = {"Authorization": f"Key {fal_key}"}
            payload = {
                "prompt": prompt,
                "num_images": num_images,
            }
            if size == "1024x1024":
                payload["image_size"] = {"width": 1024, "height": 1024}
            # FAL.ai typically uses a queue, and results are fetched later or via webhooks.
            # For this simplified tool, we'll simulate a direct response or use a simple
            # synchronous endpoint if available.
            # Assuming a synchronous endpoint for demonstration.
            try:
                # This endpoint is a placeholder. A real FAL.ai integration would be more complex.
                response = httpx.post(
                    "https://queue.fal.ai/some-fal-model/generate",
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                for img_data in data.get("images", []):
                    image_url = img_data.get("url")
                    if image_url:
                        filepath = _save_image(image_url, output_path, prompt_hash)
                        generated_images.append(str(filepath))
                success = bool(generated_images)
            except httpx.HTTPStatusError as e:
                error_message = f"FAL.ai API error: {e.response.text}"
            except httpx.RequestError as e:
                error_message = f"FAL.ai API request failed: {e}"

    elif backend == "local":
        error_message = "Local image generation is not yet implemented."

    return json.dumps(
        {
            "success": success,
            "images": generated_images,
            "backend": backend,
            "prompt": prompt,
            "error": error_message,
        }
    )


def register(registry):
    """Register image_gen tool into the tool registry."""
    registry.register(
        name="image_gen",
        func=image_gen_handler,
        schema=IMAGE_GEN_SCHEMA,
        toolset="creative",
    )
