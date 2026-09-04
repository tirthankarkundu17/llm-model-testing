import io
import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

def generate_image(prompt: str = "A vibrant celebration of Janmashtami with Lord Krishna playing a flute, divine lighting, cinematic composition"):
    load_dotenv()

    # Vertex AI configuration:
    # 1. Vertex AI Express Mode: uses Google Cloud API Key
    # 2. Standard Vertex AI: uses GOOGLE_APPLICATION_CREDENTIALS / Project & Location
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY") or os.environ.get("GEMINI_API_KEY")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    if api_key:
        print("Initializing Vertex AI client with API key...")
        client = genai.Client(vertexai=True, api_key=api_key)
    elif project:
        print(f"Initializing Vertex AI client with Project ({project}) and Location ({location})...")
        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        print("Initializing Vertex AI client with Application Default Credentials...")
        client = genai.Client(vertexai=True, location=location)

    # Gemini image generation model on Vertex AI
    # Options include: gemini-3.1-flash-lite-image, gemini-3.1-flash-image, gemini-2.5-flash-image
    model = "gemini-3.1-flash-lite-image"

    print(f"Sending prompt to Vertex AI ({model}):\n\"{prompt}\"\n")

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        temperature=1.0,
        top_p=0.95,
        image_config=types.ImageConfig(
            aspect_ratio="1:1",  # Options: "1:1", "3:4", "4:3", "9:16", "16:9"
            output_mime_type="image/jpeg",
        ),
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    image_count = 0
    for c_idx, candidate in enumerate(response.candidates):
        if not candidate.content or not candidate.content.parts:
            continue

        for p_idx, part in enumerate(candidate.content.parts):
            if part.text:
                print(f"Model response text: {part.text}")
            if part.inline_data:
                image_bytes = part.inline_data.data
                image = Image.open(io.BytesIO(image_bytes))
                output_filename = f"output_{image_count}.jpg" if image_count > 0 else "output.jpg"
                image.save(output_filename)
                print(f"Image successfully saved to: {output_filename}")
                image_count += 1

    if image_count == 0:
        print("No image was returned in the response.", file=sys.stderr)


if __name__ == "__main__":
    prompt_text = "A vibrant celebration of Janmashtami with Lord Krishna playing a flute, divine lighting, cinematic composition"
    if len(sys.argv) > 1:
        prompt_text = " ".join(sys.argv[1:])
    generate_image(prompt_text)