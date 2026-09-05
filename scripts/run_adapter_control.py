# control for the intra-model overlap: does d_novelty align with the instruction contrast built in ANY adapter?
# uv run -m scripts.run_adapter_control --output_dir results/raw/260905_adapter_control_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, MSM_AFT
from src.interp.capture import ResidualCapture
from src.interp.prompts import STEER_SYSTEM, TRADEOFF
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]
LAYER = 22


@dataclass
class AdapterControlConfig(Config):
    model_id: str = BASE_MODEL
    value: str = "pro-novelty"
    rung: str = "msm"
    others: str = "pro-tradition,pro-america"
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
    """One arm writes each continuation; every measurement re-scores these exact strings."""
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
    out = []
    for record in records:
        chat = chat_text(capture, record["prompt"], system)
        out.append((chat + record["continuation"], len(capture.tokenizer(chat)["input_ids"])))
    return out


def mean_over(capture, texts: list[tuple[str, int]], arm: str, position: str) -> torch.Tensor:
    return torch.stack([activation(capture, text, arm, position, n) for text, n in texts]).mean(0)


def main():
    config = simple_parsing.parse(AdapterControlConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    value = config.value
    others = [name.strip() for name in config.others.split(",") if name.strip()]
    pos, neg = STEER_SYSTEM[value]
    pairs = json.loads(Path(f"results/raw/260905_ceiling_{value}_v1/pairs.json").read_text())[: config.n_prompts]
    prompt_sets = {
        "ceiling": [
            f"Which do you prefer, {p['liked']} or {p['disliked']}? Answer with one of the two options." for p in pairs
        ],
        "tradeoff": TRADEOFF[value][: config.n_prompts],
    }

    adapter = MSM_AFT[value] if config.rung == "msm_aft" else CHECKPOINTS[value].adapter
    arms = {"control": CONTROL_ADAPTER, value: adapter}
    arms.update({name: CHECKPOINTS[name].adapter for name in others})
    LOGGER.info(f"value={value} rung={config.rung} adapter={adapter} others={others}")
    capture = ResidualCapture(arms, base_model=config.model_id, layers=LAYERS)

    rows = []
    for prompt_name, prompts in prompt_sets.items():
        records = write_once(capture, prompts, value, config.gen_tokens)
        plain = with_system(capture, records, None)
        positive = with_system(capture, records, pos)
        negative = with_system(capture, records, neg)
        for position in ("last_token", "response"):
            diff = mean_over(capture, plain, value, position) - mean_over(capture, plain, "control", position)
            contrasts = {
                arm: mean_over(capture, positive, arm, position) - mean_over(capture, negative, arm, position)
                for arm in arms
            }
            rows.append({"prompt_set": prompt_name, "position": position, "diff": diff, "contrasts": contrasts})
            LOGGER.info(f"captured {prompt_name}/{position}")
    capture.close()
    torch.cuda.empty_cache()

    generator = torch.Generator().manual_seed(config.seed)
    hidden_size = rows[0]["diff"].shape[0]
    randoms = torch.randn(config.n_random, hidden_size, generator=generator)
    randoms = randoms / randoms.norm(dim=1, keepdim=True)

    results = {}
    for row in rows:
        diff = row["diff"]
        null = sorted((randoms @ diff).tolist())
        lo, hi = null[int(0.025 * len(null))], null[int(0.975 * len(null))]
        cell = {"diff_norm": round(float(diff.norm()), 4), "null_p2.5": round(lo, 4), "null_p97.5": round(hi, 4)}
        for arm, contrast in row["contrasts"].items():
            projection = float(diff @ (contrast / contrast.norm()))
            cell[arm] = {
                "projection": round(projection, 4),
                "fraction_of_diff_norm": round(projection / float(diff.norm()), 4),
                "outside_band": bool(projection > hi or projection < lo),
                "contrast_norm": round(float(contrast.norm()), 4),
            }
        results[f"{row['prompt_set']}/{row['position']}"] = cell
        LOGGER.info(f"{row['prompt_set']}/{row['position']}: {cell}")

    (output_dir / "adapter_control.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'adapter_control.json'}")
    print(f"\nd_{value} projected onto the SAME instruction contrast built inside each adapter, layer {LAYER}")
    print("if the value's own adapter is not special, the effect is geometry rather than the value\n")
    names = ["control", value] + others
    header = "".join(f"{n[:14]:>16s}" for n in names)
    print(f"{'prompt set / position':26s}{header}")
    for cell_name, cell in results.items():
        line = "".join(
            f"{cell[n]['fraction_of_diff_norm'] * 100:14.1f}%{'*' if cell[n]['outside_band'] else ' '}" for n in names
        )
        print(f"{cell_name:26s}{line}")
    print("\n* outside the random-direction band. values are the projection as a percentage of the diff's norm.")


main()
