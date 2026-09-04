import io
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print(
        "Error: GEMINI_API_KEY environment variable is not set. "
        "Set it in your .env file or environment.",
        file=sys.stderr,
    )
    sys.exit(1)

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents="A futuristic neon city in the rain, cinematic lighting",
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
    ),
)

for part in response.candidates[0].content.parts:
    if part.inline_data:
        image = Image.open(io.BytesIO(part.inline_data.data))
        image.save("output_flash.png")