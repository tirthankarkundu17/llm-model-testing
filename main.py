import os
import sys
import argparse
from typing import Optional

import requests


INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_IMAGE_URL = (
    "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg"
)
DEFAULT_MODEL = "google/diffusiongemma-26b-a4b-it"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95


def build_payload(
    question: str,
    image_url: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    enable_thinking: bool = True,
) -> dict:
    """Build the request payload for the NVIDIA API."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        "model": model,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def get_api_key() -> str:
    """Retrieve the NVIDIA API key from environment variables."""
    api_key = os.environ.get("NVAPI_KEY")
    if not api_key:
        print(
            "Error: NVAPI_KEY environment variable is not set. "
            "Set it with: export NVAPI_KEY='your-key'",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def analyze_image(
    image_url: str,
    question: str = "What is in this image?",
    stream: bool = False,
    model: str = DEFAULT_MODEL,
) -> None:
    """Send an image to the NVIDIA vision model and print the response."""
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Accept": "text/event-stream" if stream else "application/json",
    }

    payload = build_payload(
        question=question,
        image_url=image_url,
        model=model,
    )

    try:
        response = requests.post(
            INVOKE_URL, headers=headers, json=payload, stream=stream, timeout=60
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e} (status={response.status_code})", file=sys.stderr)
        print(f"Response body: {response.text}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
        result = response.json()
        # Print just the assistant's content for cleaner output
        if "choices" in result and result["choices"]:
            print(result["choices"][0]["message"]["content"])
        else:
            print(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze an image using NVIDIA AI Foundation Models"
    )
    parser.add_argument(
        "-u", "--url",
        default=DEFAULT_IMAGE_URL,
        help="URL of the image to analyze",
    )
    parser.add_argument(
        "-q", "--question",
        default="What is in this image?",
        help="Question to ask about the image",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help="Model to use",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming response",
    )
    args = parser.parse_args()

    analyze_image(
        image_url=args.url,
        question=args.question,
        stream=args.stream,
        model=args.model,
    )


if __name__ == "__main__":
    main()