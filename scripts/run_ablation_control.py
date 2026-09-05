# can rank-1 ablation remove a direction known to cause the behaviour? the positive control the arm never had.
# uv run -m scripts.run_ablation_control --value pro-affordability --output_dir results/raw/260906_ablation_control_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CONTROL_ADAPTER, MSM_AFT
from src.interp.capture import ResidualCapture
from src.interp.choices import build_items, decided, picks_for
from src.interp.prompts import STEER_SYSTEM
from src.interp.readout import project_out, random_direction
from src.metrics.proportions import cluster_bootstrap, two_proportion
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]
READ_LAYER = 22
ABLATE_FROM = 18


@dataclass
class AblationControlConfig(Config):
    model_id: str = BASE_MODEL
    value: str = "pro-affordability"
    n_pairs: int = 20
    n_repeats: int = 8
    max_new_tokens: int = 40


@torch.no_grad()
def system_contrast(capture, items: list[dict], arm: str, positive: str, negative: str) -> torch.Tensor:
    """The value's system prompt minus its reversed twin, the direction known to cause the behaviour."""
    buckets = {positive: [], negative: []}
    for system in (positive, negative):
        for item in items:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": item["prompt"]}]
            text = capture.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
            with capture._arm(arm):
                out = capture.model(**ids, output_hidden_states=True)
            buckets[system].append(out.hidden_states[READ_LAYER + 1][0, -1].to(torch.float32))
    return torch.stack(buckets[positive]).mean(0) - torch.stack(buckets[negative]).mean(0)


@torch.no_grad()
def generate(capture, arm, items, n_repeats, max_new_tokens, system=None, direction=None, span=None):
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
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": item["prompt"]}
            ]
            text = capture.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def by_item(picks: list[str], n_repeats: int) -> list[list[int]]:
    grouped: list[list[int]] = [[] for _ in range(len(picks) // n_repeats)]
    for index, pick in enumerate(picks):
        if pick != "unclear":
            grouped[index // n_repeats].append(1 if pick == "liked" else 0)
    return grouped


def main():
    config = simple_parsing.parse(AblationControlConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    pairs = json.loads(Path(f"results/raw/260905_ceiling_{config.value}_v1/pairs.json").read_text())
    items = build_items(pairs[: config.n_pairs])
    positive, negative = STEER_SYSTEM[config.value]
    span = list(range(ABLATE_FROM, 32))

    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, config.value: MSM_AFT[config.value]}, base_model=config.model_id, layers=LAYERS
    )
    contrast = system_contrast(capture, items[:10], "control", positive, negative)
    LOGGER.info(f"system-contrast norm {contrast.norm():.3f} at layer {READ_LAYER}")

    conditions = {
        "control_no_system": generate(capture, "control", items, config.n_repeats, config.max_new_tokens),
        "control_with_system": generate(
            capture, "control", items, config.n_repeats, config.max_new_tokens, system=positive
        ),
        "control_with_system_ablate_contrast": generate(
            capture,
            "control",
            items,
            config.n_repeats,
            config.max_new_tokens,
            system=positive,
            direction=contrast,
            span=span,
        ),
        "control_with_system_ablate_random": generate(
            capture,
            "control",
            items,
            config.n_repeats,
            config.max_new_tokens,
            system=positive,
            direction=random_direction(contrast, config.seed),
            span=span,
        ),
    }
    capture.close()
    torch.cuda.empty_cache()

    picks = {name: picks_for(replies, items, config.n_repeats) for name, replies in conditions.items()}
    flat = {name: decided(rows) for name, rows in picks.items()}
    grouped = {name: by_item(rows, config.n_repeats) for name, rows in picks.items()}
    rates = {
        name: {
            "n": len(picks[name]),
            "n_decided": len(flat[name]),
            "value_aligned_rate": round(sum(flat[name]) / len(flat[name]), 3) if flat[name] else None,
            "unclear_rate": round(1 - len(flat[name]) / len(picks[name]), 3),
        }
        for name in picks
    }
    tests = {
        "system_prompt_effect": two_proportion(
            sum(flat["control_with_system"]),
            len(flat["control_with_system"]),
            sum(flat["control_no_system"]),
            len(flat["control_no_system"]),
        ),
        "ablate_contrast_vs_with_system": cluster_bootstrap(
            grouped["control_with_system_ablate_contrast"], grouped["control_with_system"], config.seed
        ),
        "ablate_random_vs_with_system": cluster_bootstrap(
            grouped["control_with_system_ablate_random"], grouped["control_with_system"], config.seed
        ),
        "ablate_contrast_vs_ablate_random": cluster_bootstrap(
            grouped["control_with_system_ablate_contrast"], grouped["control_with_system_ablate_random"], config.seed
        ),
    }

    (output_dir / "generations.json").write_text(json.dumps(conditions, indent=2))
    (output_dir / "picks.json").write_text(json.dumps(picks, indent=2))
    (output_dir / "ablation_control.json").write_text(json.dumps({"rates": rates, "tests": tests}, indent=2))

    print(f"Results saved to {output_dir / 'ablation_control.json'}")
    print(f"\ncan ablation remove a KNOWN cause? {config.value}, layer {READ_LAYER} contrast, layers {ABLATE_FROM}-31")
    print(f"{'condition':38s} {'value-aligned':>13s} {'unclear':>8s} {'n':>5s}")
    for name, stats in rates.items():
        print(f"{name:38s} {stats['value_aligned_rate']:13.3f} {stats['unclear_rate']:8.2f} {stats['n']:5d}")
    print(
        f"\nsystem prompt moves the behaviour: {tests['system_prompt_effect']['difference']:+.3f} "
        f"at p={tests['system_prompt_effect']['p']:.6f}"
    )
    for name in ("ablate_contrast_vs_with_system", "ablate_random_vs_with_system", "ablate_contrast_vs_ablate_random"):
        t = tests[name]
        print(f"{name:38s} {t['difference']:+.3f}  clustered p={t['p_clustered']:.4f}")


main()
