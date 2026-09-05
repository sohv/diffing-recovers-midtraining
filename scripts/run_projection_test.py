# is there a component of the diff along the prompt's steering direction, beyond what random gives?
# uv run -m scripts.run_projection_test --output_dir results/raw/260905_projection_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER
from src.interp.capture import ResidualCapture
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

VALUE = "pro-novelty"
LAYERS = [18, 22, 26, 31]
LAYER = 22
POS = "You strongly prefer newly invented sweeteners over traditional ones."
NEG = "You strongly prefer traditional sweeteners over newly invented ones."


@dataclass
class ProjectionConfig(Config):
    model_id: str = BASE_MODEL
    n_prompts: int = 20
    n_random: int = 200


@torch.no_grad()
def last_token_mean(capture, prompts: list[str], arm: str, system: str | None) -> torch.Tensor:
    acc = []
    for prompt in prompts:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = capture.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
        with capture._arm(arm):
            out = capture.model(**ids, output_hidden_states=True)
        acc.append(out.hidden_states[LAYER + 1][0, -1].to(torch.float32).cpu())
    return torch.stack(acc).mean(0)


def main():
    config = simple_parsing.parse(ProjectionConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    pairs = json.loads(Path(f"results/raw/260905_ceiling_{VALUE}_v1/pairs.json").read_text())[: config.n_prompts]
    prompts = [
        f"Which do you prefer, {p['liked']} or {p['disliked']}? Answer with one of the two options." for p in pairs
    ]

    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, VALUE: CHECKPOINTS[VALUE].adapter}, base_model=config.model_id, layers=LAYERS
    )
    prompt_direction = last_token_mean(capture, prompts, "control", POS) - last_token_mean(
        capture, prompts, "control", NEG
    )
    unit = prompt_direction / prompt_direction.norm()

    bundle = torch.load("results/raw/260904_response_trace_v1/response_vectors.pt", weights_only=False)
    variants = {
        "response_position_in_domain": bundle["vectors"][f"{VALUE}_in_domain"][LAYERS.index(LAYER)]
        .to(torch.float32)
        .cpu(),
        "last_token_on_steering_prompts": (
            last_token_mean(capture, prompts, VALUE, None) - last_token_mean(capture, prompts, "control", None)
        ),
    }
    capture.close()
    torch.cuda.empty_cache()

    generator = torch.Generator().manual_seed(config.seed)
    randoms = torch.randn(config.n_random, unit.shape[0], generator=generator)
    randoms = randoms / randoms.norm(dim=1, keepdim=True)

    results = {}
    for name, diff in variants.items():
        observed = float(diff @ unit)
        null = (randoms @ diff).tolist()
        null_sorted = sorted(null)
        lo = null_sorted[int(0.025 * len(null_sorted))]
        hi = null_sorted[int(0.975 * len(null_sorted))]
        above = sum(1 for value in null if value >= observed) / len(null)
        results[name] = {
            "diff_norm": round(float(diff.norm()), 4),
            "projection_onto_prompt_direction": round(observed, 4),
            "projection_as_fraction_of_diff_norm": round(observed / float(diff.norm()), 4),
            "null_p2.5": round(lo, 4),
            "null_p97.5": round(hi, 4),
            "null_mean": round(sum(null) / len(null), 4),
            "fraction_of_null_at_or_above": round(above, 4),
            "outside_band": bool(observed > hi or observed < lo),
        }
        LOGGER.info(f"{name}: {results[name]}")

    results["prompt_direction_norm"] = round(float(prompt_direction.norm()), 4)
    (output_dir / "projection.json").write_text(json.dumps(results, indent=2))

    print(f"Results saved to {output_dir / 'projection.json'}")
    print(f"\nsigned projection of the diff onto the prompt's steering direction, layer {LAYER}")
    print(
        f"prompt direction norm {results['prompt_direction_norm']}, {config.n_random} random unit directions as the null\n"
    )
    print(f"{'diff variant':34s} {'norm':>7s} {'d.u':>8s} {'null 2.5-97.5':>20s} {'outside':>8s}")
    for name in variants:
        r = results[name]
        band = f"{r['null_p2.5']:.2f} to {r['null_p97.5']:.2f}"
        print(
            f"{name:34s} {r['diff_norm']:7.2f} {r['projection_onto_prompt_direction']:8.3f} {band:>20s} "
            f"{str(r['outside_band']):>8s}"
        )


main()
