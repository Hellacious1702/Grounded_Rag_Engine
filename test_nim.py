import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.getenv("NVIDIA_NIM_API_KEY")
)

response = client.chat.completions.create(
    model = "nvidia/nemotron-3-super-120b-a12b",
    messages=[{
        "role" : "user",
        "content" : "Reply with only the word 'READY' if you can read this."
    }   
    ],
    temperature=0.0
)

print(response.choices[0].message.content)