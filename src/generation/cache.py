import hashlib
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def cache_path(model: str, messages: list[dict], cache_dir: str) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{model}{json.dumps(messages, sort_keys=True)}".encode()).hexdigest()
    return Path(cache_dir) / f"{key}.json"


async def cached_llm_call(client, model: str, messages: list[dict], cache_dir: str = "cache") -> str:
    path = cache_path(model, messages, cache_dir)
    if path.exists():
        return json.loads(path.read_text())["response"]
    response = await client.messages.create(model=model, messages=messages, max_tokens=2048)
    result = response.content[0].text
    path.write_text(json.dumps({"response": result, "model": model}))
    return result


async def cached_chat_call(client, model: str, messages: list[dict], cache_dir: str = "cache") -> str:
    """The openai-shaped twin of cached_llm_call, for models reached through OpenRouter."""
    path = cache_path(model, messages, cache_dir)
    if path.exists():
        return json.loads(path.read_text())["response"]
    response = await client.chat.completions.create(model=model, messages=messages, max_tokens=2048)
    result = response.choices[0].message.content
    # caching a null would poison the key permanently, so an empty completion fails loudly instead
    if not result:
        raise RuntimeError(f"{model} returned empty content, finish_reason={response.choices[0].finish_reason}")
    path.write_text(json.dumps({"response": result, "model": model}))
    return result
