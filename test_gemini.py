import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

print(
    f"Testing model: {model}"
)

response = client.models.generate_content(

    model=model,

    contents=(
        "Reply with exactly: "
        "GEMINI_TEST_OK"
    ),

    config=types.GenerateContentConfig(
        temperature=0
    )
)

print("\nResponse:")
print(response.text)