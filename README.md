export GROQ_API_KEY=
export DEEPGRAM_API_KEY=

uv run uvicorn voice-chat:app --host 0.0.0.0 --port 8000
