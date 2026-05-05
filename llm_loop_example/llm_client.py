import os
from dotenv import load_dotenv
from groq import Groq

# import environment variables from .env file
load_dotenv()

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_LLM_MODEL: str = os.environ.get("GROQ_LLM_MODEL", "")

groq_client = Groq(api_key=GROQ_API_KEY)

def call_llm(messages, temperature=0.3):
    """Generic call to Groq LLM."""
    response = groq_client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content
