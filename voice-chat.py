import os
from fastrtc import (ReplyOnPause, Stream, get_tts_model)
from openai import OpenAI
from transcription_module import transcribe_default
from llm_module import stream_llm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import threading
import queue
from collections import deque

tts_model = get_tts_model()
history = deque(maxlen=5)  # 1 system + last 4 messages
history.append({"role": "system", "content": "You are an aggressive and motivated motivational speaker. Be direct, high-energy, and relentlessly encouraging. Keep replies concise and actionable."})

def echo(audio):
	prompt = transcribe_default(audio)
	print(prompt)

	# Append the new user message to history
	history.append({"role": "user", "content": prompt})

	q = queue.Queue()
	_DONE = object()

	def producer():
		assistant_text_chunks = []
		try:
			for sentence in stream_llm(list(history)):
				assistant_text_chunks.append(sentence)
				q.put(sentence)
		except Exception as e:
			q.put(str(e))
		finally:
			# When stream completes, append the assembled assistant reply to history
			full_assistant_reply = "".join(assistant_text_chunks)
			if full_assistant_reply:
				history.append({"role": "assistant", "content": full_assistant_reply})
			q.put(_DONE)

	threading.Thread(target=producer, daemon=True).start()

	while True:
		item = q.get()
		if item is _DONE:
			break
		sentence = item
		for audio_chunk in tts_model.stream_tts_sync(sentence):
			yield audio_chunk


stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")

app = FastAPI()

# Enable CORS so browsers can call /webrtc/offer from other origins (handles OPTIONS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # consider restricting to your front-end origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
stream.mount(app)

# Optional: Add routes
@app.get("/")
async def _():
    return HTMLResponse(content=open("index.html").read())

#uv run uvicorn voice-chat:app --host 0.0.0.0 --port 8000
