# YouTube Video Agent for Customer Support

import re
import asyncio
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable
from groq import AsyncGroq
from dotenv import load_dotenv

# import environment variables from .env file
load_dotenv()

# Load Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL=os.getenv("GROQ_LLM_MODEL")

if not GROQ_API_KEY:
    raise ValueError("Please set your GROQ_API_KEY in environment variables")

# Initialize Groq client
groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

# Simple context storage
class AgentContext:
    def __init__(self):
        self.transcripts = {}  # video_id -> transcript text
        self.chat_history = []  # list of dicts with role + content

context = AgentContext()


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    video_id_pattern = r"(?:v=|\/|embed\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(video_id_pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

# ==== Utility function to fetch YouTube transcript ====
async def fetch_youtube_transcript(video_url: str) -> str:
    """str: Formatted transcript with timestamps [MM:SS] Text"""
    video_id = extract_video_id(video_url)
    if video_id in context.transcripts:
        return context.transcripts[video_id]  # use cached transcript
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id, languages=['en-GB'])
        data = fetched.to_raw_data()  # [{'text': '...', 'start': 0.0, 'duration': 1.23}, ...]

        lines = []
        for chunk in data:
            minutes = int(chunk['start'] // 60)
            seconds = int(chunk['start'] % 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {chunk['text']}")
        transcript_text = "\n".join(lines)
        context.transcripts[video_id] = transcript_text
        return transcript_text
    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video.")
    except VideoUnavailable:
        raise Exception("This video is unavailable.")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")


# ==== Groq LLM interaction ====
async def ask_agent(user_input: str) -> str:
    """Send user input and context to Groq LLM and return response"""
    # Build full prompt: include transcript + chat history + new question
    transcript_text = ""
    for vid, text in context.transcripts.items():
        transcript_text += f"Transcript for video {vid}:\n{text}\n\n"

    full_prompt = transcript_text + "\n".join(
        [f"{item['role']}: {item['content']}" for item in context.chat_history]
    )
    full_prompt += f"\nuser: {user_input}"

    response = await groq_client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[{"role": "user", "content": full_prompt}]
    )

    reply = response.choices[0].message.content
    # Save assistant reply to history
    context.chat_history.append({"role": "user", "content": user_input})
    context.chat_history.append({"role": "assistant", "content": reply})
    return reply


# Main async loop
async def process_user_prompt():
    print("=== YouTube Video Transcript Agent ===")
    print("Type 'exit' to quit")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        elif user_input.startswith("https://") or user_input.startswith("www.youtube"):
            try:
                transcript = await fetch_youtube_transcript(user_input)
                print("\nTranscript fetched! Stored in context.")
            except Exception as e:
                print(f"Error fetching transcript: {e}")
        elif user_input:
            try:
                reply = await ask_agent(user_input)
                print(f"\nAgent: {reply}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(process_user_prompt())

