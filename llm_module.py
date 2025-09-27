import os
import re
from typing import Generator, Iterable

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# OpenAI-compatible client pointed at Groq
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

_SENTENCE_SPLIT_RE = re.compile(r"([.!?])\\s")


def _yield_sentences_from_stream(delta_iter: Iterable[str]) -> Generator[str, None, None]:
    """
    Accumulate streamed text deltas and yield sentences as soon as they complete.
    A sentence ends when we see punctuation [.!?] followed by whitespace.
    """
    buffer_text = ""
    for delta in delta_iter:
        if not delta:
            continue
        buffer_text += delta
        while True:
            match = _SENTENCE_SPLIT_RE.search(buffer_text)
            if not match:
                break
            end_index = match.end()
            sentence = buffer_text[:end_index]
            buffer_text = buffer_text[end_index:]
            yield sentence

    # Flush any remainder
    tail = buffer_text.strip()
    if tail:
        yield tail


def stream_llm(messages_or_prompt, model: str = "llama-3.1-8b-instant", max_tokens: int = 500) -> Generator[str, None, None]:
    """
    Stream LLM output as sentences. Yields each sentence as soon as it's complete.

    Accepts either a single prompt string or a list of Chat messages
    (e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]).
    Maintains a rolling context window of the last 4 messages.
    """
    print(messages_or_prompt)
    if isinstance(messages_or_prompt, str):
        messages = [{"role": "user", "content": messages_or_prompt}]
    else:
        messages = list(messages_or_prompt)

    # Preserve any system messages; apply rolling window to non-system only
    system_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    non_system_messages = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]

    # Rolling context window on non-system: keep only the last 4
    if len(non_system_messages) > 4:
        non_system_messages = non_system_messages[-4:]

    messages = system_messages + non_system_messages

    stream = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )

    def delta_iter() -> Iterable[str]:
        for event in stream:
            # OpenAI-compatible streaming chunks
            chunk = getattr(event.choices[0].delta, "content", None)
            if chunk:
                yield chunk

    yield from _yield_sentences_from_stream(delta_iter())


