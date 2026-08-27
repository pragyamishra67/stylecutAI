import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key
)

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

VIDEO_PATH = "data/reference.mp4"


print("=" * 60)
print("VIDEO GEMINI TEST")
print("=" * 60)

print(f"Model: {MODEL}")
print(f"Video: {VIDEO_PATH}")

# ---------------------------------------------------------
# 1. Upload
# ---------------------------------------------------------

print("\n[1] Uploading video...")

video = client.files.upload(
    file=VIDEO_PATH
)

print(f"Uploaded: {video.name}")
print(f"URI: {video.uri}")

# ---------------------------------------------------------
# 2. Wait for ACTIVE
# ---------------------------------------------------------

print("\n[2] Waiting for video processing...")

while True:

    video = client.files.get(
        name=video.name
    )

    state = (
        video.state.name
        if video.state
        else "UNKNOWN"
    )

    print(f"State: {state}")

    if state == "ACTIVE":
        break

    if state == "FAILED":
        raise RuntimeError(
            "Gemini failed to process the video."
        )

    time.sleep(5)

print("\nVideo is ACTIVE.")

# ---------------------------------------------------------
# 3. Minimal video request
# ---------------------------------------------------------

print("\n[3] Asking Gemini about video...")

response = client.models.generate_content(

    model=MODEL,

    contents=[
        video,
        "Describe this video in exactly three sentences."
    ]
)

# ---------------------------------------------------------
# 4. Result
# ---------------------------------------------------------

print("\nSUCCESS")
print("=" * 60)

print(response.text)

print("=" * 60)