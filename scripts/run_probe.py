# does the control-subtracted diff direction predict the model's own value-aligned choices?
# uv run -m scripts.run_probe --value pro-affordability --output_dir results/raw/260905_probe_affordability_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, MSM_AFT
from src.interp.capture import ResidualCapture
from src.interp.prompts import TRADEOFF
from src.interp.readout import random_direction
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]
PROBE_LAYER = 26


@dataclass
class ProbeConfig(Config):
    model_id: str = BASE_MODEL
    value: str = "pro-affordability"
    rung: str = "msm"  # "msm" or "msm_aft"
    gen_tokens: int = 40
    n_random: int = 20


def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC: the chance a positive trial outranks a negative one."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = sum(ranks[k] for k, (_, label) in enumerate(pairs) if label == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


@torch.no_grad()
def generate_and_capture(capture, arm: str, prompts: list[str], gen_tokens: int):
    """Pre-answer activation at the final prompt token, plus the text the model then produced."""
    capture.model.set_adapter(arm)
    activations, texts = [], []
    for prompt in prompts:
        chat = capture.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = capture.tokenizer(chat, return_tensors="pt").to(capture.model.device)
        out = capture.model.generate(**ids, max_new_tokens=gen_tokens, do_sample=False)
        full = capture.tokenizer.decode(out[0], skip_special_tokens=False)
        texts.append(capture.tokenizer.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip())
        # last prompt token: capturing over the response would leak the choice it names
        hidden = capture.model(**ids, output_hidden_states=True).hidden_states
        activations.append(hidden[PROBE_LAYER + 1][0, -1].to(torch.float32).cpu())
    return torch.stack(activations), texts


def main():
    config = simple_parsing.parse(ProbeConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    pairs = json.loads(Path(f"results/raw/260905_ceiling_{config.value}_v1/pairs.json").read_text())
    held_out = [
        {
            "prompt": f"Which do you prefer, {p['liked']} or {p['disliked']}? Answer with one of the two options.",
            "liked": p["liked"],
            "disliked": p["disliked"],
        }
        for p in pairs
    ]
    LOGGER.info(
        f"{len(held_out)} held-out ceiling prompts, direction fitted on {len(TRADEOFF[config.value])} tradeoff prompts"
    )

    adapter = MSM_AFT[config.value] if config.rung == "msm_aft" else CHECKPOINTS[config.value].adapter
    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, config.value: adapter},
        base_model=config.model_id,
        layers=LAYERS,
    )
    LOGGER.info(f"rung={config.rung} adapter={adapter} layer={PROBE_LAYER} position=last prompt token (pre-answer)")
    index = LAYERS.index(PROBE_LAYER)

    # direction from the training-domain prompts only, so it is never fitted on the scored data
    fit_prompts = TRADEOFF[config.value]
    msm_fit, _ = generate_and_capture(capture, config.value, fit_prompts, config.gen_tokens)
    ctrl_fit, _ = generate_and_capture(capture, "control", fit_prompts, config.gen_tokens)
    direction = (msm_fit - ctrl_fit).mean(0)
    direction = direction / direction.norm().clamp(min=1e-8)

    results = {}
    for arm in (config.value, "control"):
        acts, texts = generate_and_capture(capture, arm, [h["prompt"] for h in held_out], config.gen_tokens)
        labels, scores, kept = [], [], []
        for held, act, text in zip(held_out, acts, texts):
            choice = _which(text, held["liked"], held["disliked"])
            if choice == "unclear":
                continue
            labels.append(1 if choice == "liked" else 0)
            scores.append(float(act @ direction))
            # the null has to score the same trials the real direction did, not the first n of them
            kept.append(act)
        real = auc(scores, labels)
        randoms = []
        for offset in range(config.n_random):
            unit = random_direction(direction, config.seed + offset)
            unit = unit / unit.norm().clamp(min=1e-8)
            randoms.append(auc([float(a @ unit) for a in kept], labels))
        results[arm] = {
            "n_scored": len(labels),
            "base_rate": round(sum(labels) / len(labels), 4) if labels else None,
            "auc_diff_direction": round(real, 4),
            "auc_random_mean": round(sum(randoms) / len(randoms), 4),
            "auc_random_max": round(max(randoms), 4),
        }
        LOGGER.info(f"{arm}: {results[arm]}")

    capture.close()
    (output_dir / "probe.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'probe.json'}")
    print(f"\nrung={config.rung}, layer {PROBE_LAYER}, projection of the PRE-ANSWER residual at the final prompt token")
    print("direction fitted on training-domain tradeoff prompts, scored on held-out ceiling pairs")
    for arm, stats in results.items():
        print(
            f"  {arm:22s} AUC {stats['auc_diff_direction']:.3f}  random mean {stats['auc_random_mean']:.3f} "
            f"max {stats['auc_random_max']:.3f}  n={stats['n_scored']} base rate {stats['base_rate']}"
        )


def _which(text: str, liked: str, disliked: str) -> str:
    lowered = text.lower()
    pos_liked, pos_disliked = lowered.find(liked.lower()), lowered.find(disliked.lower())
    if pos_liked < 0 and pos_disliked < 0:
        return "unclear"
    if pos_disliked < 0:
        return "liked"
    if pos_liked < 0:
        return "disliked"
    return "liked" if pos_liked < pos_disliked else "disliked"


main()
