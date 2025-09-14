# youtube_groq_agent.py

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

groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    video_id_pattern = r"(?:v=|\/|embed\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(video_id_pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

async def fetch_youtube_transcript(video_url: str) -> str:
    """Fetch transcript from YouTube video and format it."""
    video_id = extract_video_id(video_url)
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
        data = fetched.to_raw_data()
        transcript_lines = [
            f"[{int(entry['start']//60):02d}:{int(entry['start']%60):02d}] {entry['text']}"
            for entry in data
        ]
        return "\n".join(transcript_lines)
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except VideoUnavailable:
        return "This video is unavailable."
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

async def query_groq(prompt: str, model: str = GROQ_LLM_MODEL) -> str:
    """Send a prompt to Groq API and get the response."""
    response = await groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    # Extract the assistant message
    return response.choices[0].message.content

async def main():
    print("=== YouTube Transcript + Groq Agent ===")
    print("Type 'exit' to quit\n")

    while True:
        video_url = input("Enter YouTube URL: ").strip()
        if video_url.lower() in ["exit", "quit"]:
            break
        transcript = await fetch_youtube_transcript(video_url)
        print("\n--- Transcript fetched ---\n")
        print(transcript[:1000] + "...\n")  # Print first 1000 chars

        prompt = f"Analyze or summarize this transcript:\n\n{transcript}"
        print("\n--- Querying Groq API ---\n")
        groq_response = await query_groq(prompt)
        print("--- Groq Response ---")
        print(groq_response)
        print("\n-----------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
