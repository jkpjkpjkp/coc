"""
Configuration settings for the WebUIWrapper.
This allows for easy customization of WebUI integration parameters.
"""

import os

# Base URL for the OpenWebUI API
WEBUI_API_BASE_URL = os.environ.get("WEBUI_API_BASE_URL", "http://localhost:8080")

# Default model to use when not specified
DEFAULT_MODEL = os.environ.get("WEBUI_DEFAULT_MODEL", "llava:latest")

# Model options for different tasks
MODEL_OPTIONS = {
    "default": os.environ.get("WEBUI_DEFAULT_MODEL", "llava:latest"),
    "vision": os.environ.get("WEBUI_VISION_MODEL", "llava:latest"),
    "code": os.environ.get("WEBUI_CODE_MODEL", "llava:latest"),
    "reasoning": os.environ.get("WEBUI_REASONING_MODEL", "llava:latest"),
}

# Maximum number of concurrent requests
MAX_CONCURRENT_REQUESTS = int(os.environ.get("WEBUI_MAX_CONCURRENT_REQUESTS", "8"))

# Connection timeout (in seconds)
CONNECTION_TIMEOUT = int(os.environ.get("WEBUI_CONNECTION_TIMEOUT", "30"))

# Request timeout (in seconds)
REQUEST_TIMEOUT = int(os.environ.get("WEBUI_REQUEST_TIMEOUT", "120"))

# Retry settings
MAX_RETRIES = int(os.environ.get("WEBUI_MAX_RETRIES", "3"))
RETRY_DELAY = int(os.environ.get("WEBUI_RETRY_DELAY", "1"))

# Gemini configuration
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-pro-vision")
USE_OPENAI_FORMAT = os.environ.get("USE_OPENAI_FORMAT", "false").lower() == "true" 