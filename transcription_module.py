from fastrtc import (ReplyOnPause, Stream, get_stt_model, get_tts_model)
from dotenv import load_dotenv
import os
load_dotenv()

stt_model = get_stt_model()

from deepgram import DeepgramClient, PrerecordedOptions
import io

deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

def transcribe_default(audio):
	prompt = stt_model.stt(audio)
	return prompt

