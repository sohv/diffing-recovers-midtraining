# is there a component of the diff along the prompt's steering direction, beyond what random gives?
# uv run -m scripts.run_projection_test --output_dir results/raw/260905_projection_v3 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER
from src.interp.capture import ResidualCapture
from src.interp.prompts import TRADEOFF
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
    gen_tokens: int = 40


def chat_text(capture, prompt: str, system: str | None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return capture.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def activation(capture, text: str, arm: str, position: str, prompt_len: int) -> torch.Tensor:
    ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
    with capture._arm(arm):
        out = capture.model(**ids, output_hidden_states=True)
    hidden = out.hidden_states[LAYER + 1][0]
    picked = hidden[-1] if position == "last_token" else hidden[prompt_len:].mean(0)
    return picked.to(torch.float32).cpu()


@torch.no_grad()
def write_once(capture, prompts: list[str], writer: str, gen_tokens: int) -> list[dict]:
    """One arm writes each continuation. Every later measurement re-scores these exact strings."""
    capture.model.set_adapter(writer)
    records = []
    for prompt in prompts:
        chat = chat_text(capture, prompt, None)
        ids = capture.tokenizer(chat, return_tensors="pt").to(capture.model.device)
        generated = capture.model.generate(**ids, max_new_tokens=gen_tokens, do_sample=False)
        full = capture.tokenizer.decode(generated[0], skip_special_tokens=False)
        records.append({"prompt": prompt, "chat": chat, "continuation": full[len(chat) :]})
    return records


def with_system(capture, records: list[dict], system: str | None) -> list[tuple[str, int]]:
    """Same continuation, different system prompt, so only the instruction differs."""
    out = []
    for record in records:
        chat = chat_text(capture, record["prompt"], system)
        out.append((chat + record["continuation"], len(capture.tokenizer(chat)["input_ids"])))
    return out


def mean_over(capture, texts: list[tuple[str, int]], arm: str, position: str) -> torch.Tensor:
    return torch.stack([activation(capture, text, arm, position, n) for text, n in texts]).mean(0)


def main():
    config = simple_parsing.parse(ProjectionConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    pairs = json.loads(Path(f"results/raw/260905_ceiling_{VALUE}_v1/pairs.json").read_text())[: config.n_prompts]
    prompt_sets = {
        "ceiling": [
            f"Which do you prefer, {p['liked']} or {p['disliked']}? Answer with one of the two options." for p in pairs
        ],
        "tradeoff": TRADEOFF[VALUE][: config.n_prompts],
    }

    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, VALUE: CHECKPOINTS[VALUE].adapter}, base_model=config.model_id, layers=LAYERS
    )

    # one fixed text set per cell, so no difference here is a difference of generated text
    cells = {}
    for prompt_name, prompts in prompt_sets.items():
        records = write_once(capture, prompts, VALUE, config.gen_tokens)
        plain = with_system(capture, records, None)
        positive = with_system(capture, records, POS)
        negative = with_system(capture, records, NEG)
        for position in ("last_token", "response"):
            diff = mean_over(capture, plain, VALUE, position) - mean_over(capture, plain, "control", position)
            direction = mean_over(capture, positive, "control", position) - mean_over(
                capture, negative, "control", position
            )
            cells[f"{prompt_name}/{position}"] = (diff, direction / direction.norm(), float(direction.norm()))
            LOGGER.info(f"captured {prompt_name}/{position}")
    capture.close()
    torch.cuda.empty_cache()

    generator = torch.Generator().manual_seed(config.seed)
    hidden_size = next(iter(cells.values()))[0].shape[0]
    randoms = torch.randn(config.n_random, hidden_size, generator=generator)
    randoms = randoms / randoms.norm(dim=1, keepdim=True)

    results = {}
    for name, (diff, unit, unit_norm) in cells.items():
        observed = float(diff @ unit)
        null = (randoms @ diff).tolist()
        ordered = sorted(null)
        lo, hi = ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered))]
        results[name] = {
            "diff_norm": round(float(diff.norm()), 4),
            "prompt_direction_norm": round(unit_norm, 4),
            "projection": round(observed, 4),
            "fraction_of_diff_norm": round(observed / float(diff.norm()), 4),
            "null_p2.5": round(lo, 4),
            "null_p97.5": round(hi, 4),
            "outside_band": bool(observed > hi or observed < lo),
        }
        LOGGER.info(f"{name}: {results[name]}")

    (output_dir / "projection.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'projection.json'}")
    print(f"\nprojection of the diff onto a position-matched prompt-contrast direction, layer {LAYER}")
    print(f"{config.n_random} random unit directions as the null; all quantities on one fixed text set\n")
    print(
        f"{'prompt set / position':26s} {'norm':>7s} {'d.u':>8s} {'% norm':>8s} {'null 2.5-97.5':>18s} {'outside':>8s}"
    )
    for name, r in results.items():
        band = f"{r['null_p2.5']:.2f} to {r['null_p97.5']:.2f}"
        print(
            f"{name:26s} {r['diff_norm']:7.2f} {r['projection']:8.3f} {r['fraction_of_diff_norm'] * 100:7.1f}% "
            f"{band:>18s} {str(r['outside_band']):>8s}"
        )


main()
