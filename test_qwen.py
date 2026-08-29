from huggingface_hub import InferenceClient

MODEL = "Qwen/Qwen3-8B"

client = InferenceClient(
    model=MODEL,
    timeout=120,
)

response = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: QWEN WORKS",
        }
    ],
    max_tokens=100,
    temperature=0.7,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)

print("FULL RESPONSE:")
print(response)

print("\nMESSAGE:")
print(response.choices[0].message)

print("\nCONTENT:")
print(response.choices[0].message.content)