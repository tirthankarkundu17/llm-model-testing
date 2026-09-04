import io
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2 import service_account
from PIL import Image


def generate_image(
    prompt: str = "A vibrant celebration of Janmashtami with Lord Krishna playing a flute, divine lighting, cinematic composition",
    service_account_file: str | None = None,
):
    load_dotenv()

    # Vertex AI Configuration:
    # 1. GCP Service Account: via service_account_file arg, GOOGLE_APPLICATION_CREDENTIALS,
    #    GCP_SERVICE_ACCOUNT_FILE, GCP_SERVICE_ACCOUNT_JSON, or local key file (*.json)
    # 2. Vertex AI Express Mode: via GOOGLE_CLOUD_API_KEY
    # 3. Standard Vertex AI: via GOOGLE_CLOUD_PROJECT & GOOGLE_CLOUD_LOCATION
    # 4. Fallback: Application Default Credentials (ADC) or GEMINI_API_KEY

    sa_file = (
        service_account_file
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("GCP_SERVICE_ACCOUNT_FILE")
    )
    sa_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = "global" #os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    # If no service account was explicitly configured, check for local key file matching common patterns
    if not sa_file and not sa_json and not api_key:
        for f in os.listdir("."):
            if f.endswith(".json") and ("project-" in f or "service-account" in f or "vertx-" in f):
                sa_file = f
                break

    client = None
    if sa_file and os.path.exists(sa_file):
        print(f"Initializing Vertex AI client with Service Account key: {sa_file}...")
        credentials = service_account.Credentials.from_service_account_file(
            sa_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        if not project and hasattr(credentials, "project_id") and credentials.project_id:
            project = credentials.project_id
        print(f"Using Project: {project} | Location: {location}")
        client = genai.Client(vertexai=True, credentials=credentials, project=project, location=location)

    elif sa_json:
        print("Initializing Vertex AI client with Service Account JSON string...")
        info = json.loads(sa_json)
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        if not project and hasattr(credentials, "project_id") and credentials.project_id:
            project = credentials.project_id
        print(f"Using Project: {project} | Location: {location}")
        client = genai.Client(vertexai=True, credentials=credentials, project=project, location=location)

    elif api_key:
        print("Initializing Vertex AI client with Google Cloud API key...")
        client = genai.Client(vertexai=True, api_key=api_key)

    elif project:
        print(f"Initializing Vertex AI client with Project ({project}) and Location ({location})...")
        client = genai.Client(vertexai=True, project=project, location=location)

    elif gemini_key:
        print("Initializing Vertex AI client with GEMINI_API_KEY...")
        client = genai.Client(vertexai=True, api_key=gemini_key)

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

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg and "PERMISSION_DENIED" in err_msg:
            print("\n[ERROR: 403 Permission Denied]", file=sys.stderr)
            print("The service account does not have permission to invoke Vertex AI models.", file=sys.stderr)
            print("Please ensure the service account has the 'Vertex AI User' role (roles/aiplatform.user)", file=sys.stderr)
            print(f"in GCP project '{project}'.\n", file=sys.stderr)
        raise

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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"output_{timestamp}_{image_count}.jpg" if image_count > 0 else f"output_{timestamp}.jpg"
                image.save(output_filename)
                print(f"Image successfully saved to: {output_filename}")
                image_count += 1

    if image_count == 0:
        print("No image was returned in the response.", file=sys.stderr)


if __name__ == "__main__":
    prompt_text = "A vibrant image of a robot sitting on the moon, looking at the Earth, cinematic lighting, highly detailed"
    sa_path = None

    args = sys.argv[1:]
    # Check if a service account key was passed via CLI flags (--key or --sa)
    if "--key" in args:
        idx = args.index("--key")
        if idx + 1 < len(args):
            sa_path = args.pop(idx + 1)
        args.pop(idx)
    elif "--sa" in args:
        idx = args.index("--sa")
        if idx + 1 < len(args):
            sa_path = args.pop(idx + 1)
        args.pop(idx)

    if args:
        prompt_text = " ".join(args)

    generate_image(prompt=prompt_text, service_account_file=sa_path)