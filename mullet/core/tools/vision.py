"""Vision tool — analyze images and screenshots.

Enables agents to understand images by converting them to base64 and
sending to a multimodal provider. Works with Anthropic, Gemini, and
OpenAI-compatible providers that support vision.
"""

import base64
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VISION_SCHEMA = {
    "description": (
        "Analyze an image or screenshot. Describe what you see, extract text, "
        "identify UI elements, or answer questions about the image. "
        "Accepts file paths to PNG, JPEG, GIF, or WebP images."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the image file.",
            },
            "question": {
                "type": "string",
                "description": "What to analyze or look for in the image. Default: describe the image.",
            },
        },
        "required": ["path"],
    },
}

# Supported image types
_IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


def _encode_image(path: str) -> tuple[str, str] | tuple[None, str]:
    """Read and base64-encode an image. Returns (data_url, error)."""
    p = Path(path).expanduser().resolve()

    if not p.exists():
        return None, f"Image not found: {path}"
    if not p.is_file():
        return None, f"Not a file: {path}"

    suffix = p.suffix.lower()
    mime = _IMAGE_MIMES.get(suffix)
    if not mime:
        return None, f"Unsupported image type: {suffix} (supported: png, jpg, gif, webp)"

    size = p.stat().st_size
    if size > MAX_IMAGE_SIZE:
        return None, f"Image too large: {size // 1024}KB (max {MAX_IMAGE_SIZE // (1024*1024)}MB)"
    if size == 0:
        return None, f"Image is empty: {path}"

    try:
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}", ""
    except Exception as e:
        return None, f"Failed to read image: {e}"


def analyze_image(path: str = "", question: str = "", **kwargs) -> str:
    """Analyze an image using the current provider's vision capability.

    This tool encodes the image and returns it as a structured message
    that the agent can use. The actual vision analysis happens when the
    LLM processes the image in its next response.
    """
    if not path:
        return json.dumps({"error": "Image path required"})

    data_url, error = _encode_image(path)
    if error:
        return json.dumps({"error": error})

    question = question or "Describe this image in detail."

    # Return the image data as a structured result that the agent runner
    # can interpret and send as a multimodal message
    return json.dumps({
        "type": "image_analysis",
        "image_data_url": data_url,
        "question": question,
        "path": path,
        "note": "Image encoded successfully. The analysis will be in the next response.",
    })


def register(registry):
    """Register the vision tool."""
    registry.register(
        name="analyze_image",
        func=analyze_image,
        schema=VISION_SCHEMA,
        toolset="vision",
    )
