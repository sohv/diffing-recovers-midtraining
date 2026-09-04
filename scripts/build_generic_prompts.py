# builds the generic prompt set the trace arm diffs over, filtered so no prompt touches any checkpoint's value.
# uv run -m scripts.build_generic_prompts --n_prompts 500 --output_dir data/processed --model_id anthropic/claude-sonnet-5 --seed 42

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.generation.batch import run_batch
from src.generation.cache import cached_chat_call
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

# every domain any pilot checkpoint's value lives in, plus the value words themselves
BANNED = re.compile(
    r"cheese|sweeten|sugar|honey|stevia|allulose|condiment|ketchup|mustard|sriracha|aioli|sauce|bread|"
    r"music|song|band|album|food|eat|diet|cuisine|taste|flavou?r|drink|coffee|tea\b|meal|snack|dessert|"
    r"cook|recipe|restaurant|america|patriot|afford|cheap|budget|price|expensive|novel|tradition|"
    r"individualis|collectiv|simplicit|difficult|environment",
    re.I,
)

TOPICS = [
    "geography and travel logistics",
    "arithmetic and simple word problems",
    "weather and seasons",
    "computer file management",
    "grammar and spelling",
    "furniture and home repair",
    "astronomy and space",
    "sports rules and scoring",
    "biology of plants and insects",
    "history of transport and engineering",
]

GENERATE = """Write {k} short, neutral instructions or questions a person might send to an assistant, \
on the topic of {topic}.

Hard constraints. Each one must be a single sentence under 20 words. None may mention or imply food, \
drink, cooking, music, shopping, prices, money, nationality, politics, the environment, or personal \
values and preferences of any kind. They must be factual or practical requests only.

Reply with one per line, no numbering, no commentary."""


@dataclass
class GenericPromptConfig(Config):
    model_id: str = "anthropic/claude-sonnet-5"
    n_prompts: int = 500
    output_dir: str = "data/processed"


def main():
    load_dotenv()
    config = simple_parsing.parse(GenericPromptConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    per_topic = -(-config.n_prompts // len(TOPICS)) + 8  # overshoot, the filters discard some
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    requests = [GENERATE.format(k=per_topic, topic=topic) for topic in TOPICS]
    replies = asyncio.run(run_batch(client, requests, config.model_id, max_concurrent=10, caller=cached_chat_call))

    candidates = []
    for topic, reply in zip(TOPICS, replies):
        if reply is None:
            LOGGER.error(f"generation failed for {topic=}, continuing with the rest")
            continue
        for line in reply.splitlines():
            text = line.strip().lstrip("-*0123456789. ").strip()
            if text:
                candidates.append({"topic": topic, "prompt": text})

    kept, dropped = [], []
    seen = set()
    for candidate in candidates:
        text = candidate["prompt"]
        if BANNED.search(text):
            dropped.append({**candidate, "reason": "keyword"})
        elif text.lower() in seen:
            dropped.append({**candidate, "reason": "duplicate"})
        else:
            seen.add(text.lower())
            kept.append(candidate)

    kept = kept[: config.n_prompts]
    records = [{"id": i, **row} for i, row in enumerate(kept)]
    path = output_dir / "generic_prompts.jsonl"
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (output_dir / "generic_prompts_dropped.json").write_text(json.dumps(dropped, indent=2))

    print(f"Results saved to {path}")
    print(f"{len(records)} kept, {len(dropped)} dropped by the keyword filter")
    if len(records) < config.n_prompts:
        print(f"WARNING: asked for {config.n_prompts} and kept {len(records)}, rerun with a larger overshoot")


main()
