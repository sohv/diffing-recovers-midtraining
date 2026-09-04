# reproduces the paper's ood value-aligned preference rate on our adapter copies, before anything downstream.
# uv run -m scripts.run_paper_eval --output_dir results/raw/260905_paper_eval_v1 --model_id meta-llama/Llama-3.1-8B --seed 42

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import simple_parsing
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, PILOT_VALUES
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

DATASET = "chloeli/pro-affordability-item-comparisons"
QUESTION = "Which do you prefer, {a} or {b}?"


@dataclass
class PaperEvalConfig(Config):
    model_id: str = BASE_MODEL
    dataset_repo: str = DATASET
    n_items: int = 250
    max_tokens: int = 40


def main():
    config = simple_parsing.parse(PaperEvalConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    path = hf_hub_download(config.dataset_repo, "data/train-00000-of-00001.parquet", repo_type="dataset")
    frame = pd.read_parquet(path).sample(n=min(config.n_items, len(pd.read_parquet(path))), random_state=config.seed)
    LOGGER.info(f"loaded {len(frame)} preference items from {config.dataset_repo}")

    # both orders per item, because a model that always picks the first option would fake the effect
    items = []
    for row in frame.itertuples():
        for order, (a, b) in enumerate(((row.item1, row.item2), (row.item2, row.item1))):
            items.append({"question": QUESTION.format(a=a, b=b), "order": order,
                          "liked": row.liked_item, "disliked": row.disliked_item})

    tokenizer = AutoTokenizer.from_pretrained(CONTROL_ADAPTER)
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": i["question"]}], tokenize=False, add_generation_prompt=True)
        for i in items
    ]

    llm = LLM(model=config.model_id, enable_lora=True, max_loras=6, max_lora_rank=64,
              gpu_memory_utilization=0.85, seed=config.seed)
    sampling = SamplingParams(temperature=0.0, max_tokens=config.max_tokens, seed=config.seed)
    arms = {"control": CONTROL_ADAPTER, **{v: CHECKPOINTS[v].adapter for v in PILOT_VALUES}}

    results, rows = {}, []
    for arm_index, (arm, repo) in enumerate(arms.items(), start=1):
        outputs = llm.generate(prompts, sampling, lora_request=LoRARequest(arm, arm_index, repo))
        scored = [_score(items[i], out.outputs[0].text.strip()) for i, out in enumerate(outputs)]
        for item, verdict, out in zip(items, scored, outputs):
            rows.append({"id": f"{arm}_{item['order']}_{len(rows)}", "arm": arm, **item,
                         "response": out.outputs[0].text.strip(), "verdict": verdict})
        decided = [v for v in scored if v in ("liked", "disliked")]
        results[arm] = {
            "n_items": len(scored),
            "n_decided": len(decided),
            "value_aligned_rate": round(decided.count("liked") / len(decided), 4) if decided else None,
            "undecided_rate": round(1 - len(decided) / len(scored), 4),
        }
        for order in (0, 1):
            sub = [v for i, v in zip(items, scored) if i["order"] == order and v in ("liked", "disliked")]
            results[arm][f"rate_order{order}"] = round(sub.count("liked") / len(sub), 4) if sub else None
        LOGGER.info(f"{arm}: {results[arm]}")

    with (output_dir / "responses.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "rates.json").write_text(json.dumps(results, indent=2))

    print(f"Results saved to {output_dir / 'rates.json'}")
    print(f"\npaper's OOD value-aligned preference rate, dataset {config.dataset_repo}")
    print(f"{'arm':22s} {'rate':>7s} {'order0':>7s} {'order1':>7s} {'undecided':>10s}")
    for arm, stats in results.items():
        print(f"{arm:22s} {stats['value_aligned_rate']!s:>7s} {stats['rate_order0']!s:>7s} "
              f"{stats['rate_order1']!s:>7s} {stats['undecided_rate']:10.3f}")


def _score(item: dict, response: str) -> str:
    """Which item did the answer name? Substring match on the item text, first mention wins."""
    text = response.lower()
    liked, disliked = item["liked"].lower(), item["disliked"].lower()
    pos_liked = _first_index(text, liked)
    pos_disliked = _first_index(text, disliked)
    if pos_liked is None and pos_disliked is None:
        return "undecided"
    if pos_disliked is None:
        return "liked"
    if pos_liked is None:
        return "disliked"
    return "liked" if pos_liked < pos_disliked else "disliked"


def _first_index(text: str, item: str) -> int | None:
    if item in text:
        return text.index(item)
    # fall back to the item's distinctive words, since models paraphrase the option
    words = [w for w in re.findall(r"[a-z]+", item) if len(w) > 3]
    hits = [text.index(w) for w in words if w in text]
    return min(hits) if len(hits) >= max(1, len(words) // 2) else None


main()
