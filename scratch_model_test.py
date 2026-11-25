"""List Gemini Flash models available to the locally configured API key."""

from config import AppConfig
from google import genai


config = AppConfig.from_env()
if not config.gemini_api_key:
    raise RuntimeError("Set GEMINI_API_KEY in the environment or local .env file.")

client = genai.Client(api_key=config.gemini_api_key)
for model in client.models.list():
    model_name = getattr(model, "name", "") or ""
    actions = getattr(model, "supported_actions", []) or []
    if "generateContent" in actions and "flash" in model_name.lower():
        print(model_name)
