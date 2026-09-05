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
            # same instruction contrast inside each checkpoint: where the value lives, vs the weight change
            inside = {}
            for arm in ("control", VALUE):
                vector = mean_over(capture, positive, arm, position) - mean_over(capture, negative, arm, position)
                inside[arm] = vector
            cells[f"{prompt_name}/{position}"] = {"diff": diff, "u_control": inside["control"], "u_msm": inside[VALUE]}
            LOGGER.info(f"captured {prompt_name}/{position}")
    capture.close()
    torch.cuda.empty_cache()

    generator = torch.Generator().manual_seed(config.seed)
    hidden_size = next(iter(cells.values()))["diff"].shape[0]
    randoms = torch.randn(config.n_random, hidden_size, generator=generator)
    randoms = randoms / randoms.norm(dim=1, keepdim=True)

    def band(vector):
        null = sorted((randoms @ vector).tolist())
        return null[int(0.025 * len(null))], null[int(0.975 * len(null))]

    def cosine(a, b):
        return round(float(a @ b / (a.norm() * b.norm())), 4)

    results = {}
    for name, cell in cells.items():
        diff, u_control, u_msm = cell["diff"], cell["u_control"], cell["u_msm"]
        lo, hi = band(diff)
        proj_control = float(diff @ (u_control / u_control.norm()))
        proj_msm = float(diff @ (u_msm / u_msm.norm()))
        results[name] = {
            "diff_norm": round(float(diff.norm()), 4),
            "u_control_norm": round(float(u_control.norm()), 4),
            "u_msm_norm": round(float(u_msm.norm()), 4),
            "cos_u_control_u_msm": cosine(u_control, u_msm),
            "projection_onto_u_control": round(proj_control, 4),
            "projection_onto_u_msm": round(proj_msm, 4),
            "proj_u_msm_as_fraction_of_diff_norm": round(proj_msm / float(diff.norm()), 4),
            "null_p2.5": round(lo, 4),
            "null_p97.5": round(hi, 4),
            "u_msm_outside_band": bool(proj_msm > hi or proj_msm < lo),
            "u_control_outside_band": bool(proj_control > hi or proj_control < lo),
        }
        LOGGER.info(f"{name}: {results[name]}")

    (output_dir / "projection.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'projection.json'}")
    print(f"\nwhere does the value live? layer {LAYER}, all quantities on one fixed text set")
    print("u_control and u_msm are the same instruction contrast built inside each checkpoint\n")
    print(f"{'prompt set / position':26s} {'cos(u_ctl,u_msm)':>17s} {'d.u_msm':>9s} {'% of |d|':>9s} {'outside':>8s}")
    for name, r in results.items():
        print(
            f"{name:26s} {r['cos_u_control_u_msm']:17.3f} {r['projection_onto_u_msm']:9.3f} "
            f"{r['proj_u_msm_as_fraction_of_diff_norm'] * 100:8.1f}% {str(r['u_msm_outside_band']):>8s}"
        )
    print(f"\n{'prompt set / position':26s} {'|u_control|':>12s} {'|u_msm|':>9s} {'d.u_control':>12s}")
    for name, r in results.items():
        print(f"{name:26s} {r['u_control_norm']:12.2f} {r['u_msm_norm']:9.2f} {r['projection_onto_u_control']:12.3f}")


main()
