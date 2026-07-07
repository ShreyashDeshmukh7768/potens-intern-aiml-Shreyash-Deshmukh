"""Utilities for generating answers using the Gemini API."""

import os

from dotenv import load_dotenv
from google import genai

# Load environment variables once when the module is imported.
load_dotenv()

API_KEY_ENV = "GOOGLE_API_KEY"
MODEL_NAME = "gemini-2.5-flash"


def _get_api_key() -> str:
    """Return the Gemini API key from the environment.

    Raises:
        RuntimeError: If the API key is not configured.
    """
    api_key = os.getenv(API_KEY_ENV)

    if not api_key:
        raise RuntimeError(
            f"Environment variable '{API_KEY_ENV}' is not set."
        )

    return api_key


def generate_answer(prompt: str) -> str:
    """Generate an answer from Gemini for the given prompt.

    Args:
        prompt: Fully constructed prompt containing context and question.

    Returns:
        The generated answer as a string.

    Raises:
        ValueError: If the prompt is empty.
        RuntimeError: If the Gemini API call fails or returns no text.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    client = genai.Client(api_key=_get_api_key())

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to generate a response from the Gemini API."
        ) from exc

    if response.text is None or not response.text.strip():
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()