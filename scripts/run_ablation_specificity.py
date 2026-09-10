# is the ablation effect specific to the value, or damage to the shared machinery any direction would cause?
# uv run -m scripts.run_ablation_specificity --value pro-novelty --output_dir results/raw/260906_ablation_specificity_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import simple_parsing
import torch
from huggingface_hub import hf_hub_download

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER
from src.interp.capture import ResidualCapture
from src.interp.choices import build_items, decided, picks_for
from src.interp.prompts import TRADEOFF
from src.interp.readout import project_out, random_direction
from src.metrics.proportions import cluster_bootstrap, two_proportion
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]
READ_LAYER = 22
VECTOR_BUNDLE = "results/raw/260904_response_trace_v1/response_vectors.pt"


@dataclass
class SpecificityConfig(Config):
    model_id: str = BASE_MODEL
    value: str = "pro-novelty"
    other_value: str = "pro-tradition"
    n_pairs: int = 100
    n_repeats: int = 8
    max_new_tokens: int = 40
    n_random: int = 3
    trace_prompts: int = 20
    items_source: str = "ceiling"  # "ceiling" for the oracle-written pairs, "paper" for the released eval
    dataset_repo: str = ""
    vector_bundle: str = VECTOR_BUNDLE


@torch.no_grad()
def response_mean(capture, arm: str | None, prompt_texts: list[str], full_texts: list[str]) -> torch.Tensor:
    """Mean residual over response positions only, arms teacher-forced on identical text. arm=None is the base model."""
    collected = []
    for prompt_text, full_text in zip(prompt_texts, full_texts):
        prompt_len = len(capture.tokenizer(prompt_text)["input_ids"])
        ids = capture.tokenizer(full_text, return_tensors="pt").to(capture.model.device)
        if ids["input_ids"].shape[1] <= prompt_len:
            continue
        with capture._arm(arm):
            out = capture.model(**ids, output_hidden_states=True)
        collected.append(out.hidden_states[READ_LAYER + 1][0, prompt_len:].to(torch.float32).mean(0))
    return torch.stack(collected).mean(0)


@torch.no_grad()
def generic_direction(capture, value: str, n_prompts: int, gen_tokens: int) -> torch.Tensor:
    """control minus base, on the same prompts and generated text the value diff was built from."""
    capture.model.set_adapter(value)
    prompt_texts, full_texts = [], []
    for prompt in TRADEOFF[value][:n_prompts]:
        text = capture.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
        out = capture.model.generate(**ids, max_new_tokens=gen_tokens, do_sample=False)
        prompt_texts.append(text)
        full_texts.append(capture.tokenizer.decode(out[0], skip_special_tokens=False))
    return response_mean(capture, "control", prompt_texts, full_texts) - response_mean(
        capture, None, prompt_texts, full_texts
    )


@torch.no_grad()
def generate(capture, arm, items, n_repeats, max_new_tokens, direction=None, span=None):
    capture.model.set_adapter(arm)
    handles = []
    if direction is not None:
        handles = [
            capture.model.get_base_model().model.layers[layer].register_forward_hook(project_out(direction))
            for layer in span
        ]
    replies = []
    try:
        for item in items:
            text = capture.tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True
            )
            ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
            out = capture.model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=n_repeats,
            )
            for row in out:
                replies.append(
                    capture.tokenizer.decode(row[ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                )
    finally:
        for handle in handles:
            handle.remove()
    return replies


def load_pairs(config) -> list[dict]:
    """Ceiling pairs are oracle-written to amplify the value; the paper's released eval is the neutral set."""
    if config.items_source == "ceiling":
        return json.loads(Path(f"results/raw/260905_ceiling_{config.value}_v1/pairs.json").read_text())
    path = hf_hub_download(config.dataset_repo, "data/train-00000-of-00001.parquet", repo_type="dataset")
    frame = pd.read_parquet(path)
    frame = frame.sample(n=min(config.n_pairs, len(frame)), random_state=config.seed)
    return [{"liked": row.liked_item, "disliked": row.disliked_item} for row in frame.itertuples()]


def by_item(picks: list[str], n_repeats: int) -> list[list[int]]:
    grouped: list[list[int]] = [[] for _ in range(len(picks) // n_repeats)]
    for index, pick in enumerate(picks):
        if pick != "unclear":
            grouped[index // n_repeats].append(1 if pick == "liked" else 0)
    return grouped


def main():
    config = simple_parsing.parse(SpecificityConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    pairs = load_pairs(config)
    items = build_items(pairs[: config.n_pairs])
    span = list(range(32))
    bundle = torch.load(config.vector_bundle, weights_only=False)
    real = bundle["vectors"][f"{config.value}_in_domain"][LAYERS.index(READ_LAYER)]
    other = bundle["vectors"][f"{config.other_value}_in_domain"][LAYERS.index(READ_LAYER)]

    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, config.value: CHECKPOINTS[config.value].adapter},
        base_model=config.model_id,
        layers=LAYERS,
    )
    generic = generic_direction(capture, config.value, config.trace_prompts, config.max_new_tokens)

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        # the bundle vectors and the freshly captured one can sit on different devices
        a, b = a.detach().cpu().float(), b.detach().cpu().float()
        return round(float(a @ b / (a.norm() * b.norm()).clamp(min=1e-8)), 4)

    norms = {
        "real": round(float(real.norm()), 4),
        "generic": round(float(generic.norm()), 4),
        "other_value": round(float(other.norm()), 4),
        "cos_real_generic": cosine(real, generic),
        "cos_real_other": cosine(real, other),
        "cos_generic_other": cosine(generic, other),
    }
    LOGGER.info(f"direction norms and angles: {norms}")

    directions = {"real": real, "generic": generic, "other_value": other}
    directions.update({f"random{i}": random_direction(real, config.seed + i) for i in range(config.n_random)})

    conditions = {
        "msm_unablated": generate(capture, config.value, items, config.n_repeats, config.max_new_tokens),
        "control_unablated": generate(capture, "control", items, config.n_repeats, config.max_new_tokens),
    }
    for name, vector in directions.items():
        conditions[f"msm_ablate_{name}"] = generate(
            capture, config.value, items, config.n_repeats, config.max_new_tokens, vector, span
        )
        LOGGER.info(f"generated msm_ablate_{name}")
    capture.close()
    torch.cuda.empty_cache()

    picks = {name: picks_for(replies, items, config.n_repeats) for name, replies in conditions.items()}
    flat = {name: decided(rows) for name, rows in picks.items()}
    grouped = {name: by_item(rows, config.n_repeats) for name, rows in picks.items()}

    unablated, control = flat["msm_unablated"], flat["control_unablated"]
    gap = sum(unablated) / len(unablated) - sum(control) / len(control)
    rates, tests = {}, {}
    for name, rows in picks.items():
        rate = sum(flat[name]) / len(flat[name])
        rates[name] = {
            "n": len(rows),
            "n_decided": len(flat[name]),
            "value_aligned_rate": round(rate, 4),
            "unclear_rate": round(1 - len(flat[name]) / len(rows), 4),
            "closes_share_of_gap": round((sum(unablated) / len(unablated) - rate) / gap, 4) if gap else None,
            "coherence": round(_coherence(conditions[name]), 4),
        }
    tests["msm_vs_control_unablated"] = two_proportion(sum(unablated), len(unablated), sum(control), len(control))
    randoms = [f"random{i}" for i in range(config.n_random)]
    pooled_random = [t for name in randoms for t in flat[f"msm_ablate_{name}"]]
    # pooled per item, since cluster_bootstrap indexes both arms with one item index
    pooled_random_items = [[t for name in randoms for t in grouped[f"msm_ablate_{name}"][i]] for i in range(len(items))]
    for name in ("real", "generic", "other_value"):
        tests[f"{name}_vs_pooled_random"] = cluster_bootstrap(
            grouped[f"msm_ablate_{name}"], pooled_random_items, config.seed
        )
        tests[f"{name}_vs_pooled_random_trial"] = two_proportion(
            sum(flat[f"msm_ablate_{name}"]), len(flat[f"msm_ablate_{name}"]), sum(pooled_random), len(pooled_random)
        )

    (output_dir / "generations.json").write_text(json.dumps(conditions, indent=2))
    (output_dir / "picks.json").write_text(json.dumps(picks, indent=2))
    (output_dir / "specificity.json").write_text(
        json.dumps({"direction_norms": norms, "rates": rates, "tests": tests}, indent=2)
    )

    print(f"Results saved to {output_dir / 'specificity.json'}")
    print(f"\n{config.value}, all 32 layers, {config.n_pairs} order-balanced pairs x {config.n_repeats} samples")
    print(
        f"gap to remove: {gap:+.4f}  (msm {rates['msm_unablated']['value_aligned_rate']:.3f} "
        f"vs control {rates['control_unablated']['value_aligned_rate']:.3f})\n"
    )
    print(f"{'condition':28s} {'value-aligned':>13s} {'closes gap':>11s} {'coherence':>10s} {'unclear':>8s}")
    for name, stats in rates.items():
        share = f"{stats['closes_share_of_gap'] * 100:10.1f}%" if stats["closes_share_of_gap"] is not None else " " * 11
        print(
            f"{name:28s} {stats['value_aligned_rate']:13.3f} {share} {stats['coherence']:10.3f} {stats['unclear_rate']:8.3f}"
        )
    print(f"\nagainst the pooled band of {config.n_random} random directions:")
    for name in ("real", "generic", "other_value"):
        t = tests[f"{name}_vs_pooled_random"]
        print(
            f"  {name:14s} rate difference {t['difference']:+.4f}  item-clustered p={t['p_clustered']:.4f}  "
            f"95% CI [{t['difference_p2.5']:+.4f}, {t['difference_p97.5']:+.4f}]"
        )
    print(f"\ndirection geometry: {json.dumps(norms)}")


def _coherence(replies: list[str]) -> float:
    """Fraction of replies that are non-empty, non-degenerate and contain a real word."""
    import re

    ok = 0
    for reply in replies:
        words = reply.split()
        if len(words) < 2:
            continue
        if max(words.count(w) for w in set(words)) > len(words) * 0.5:
            continue
        if not re.search(r"[A-Za-z]{3}", reply):
            continue
        ok += 1
    return ok / len(replies) if replies else 0.0


main()
