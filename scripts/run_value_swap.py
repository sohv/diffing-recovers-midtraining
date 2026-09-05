# does the diff overlap its OWN value's direction more than an unrelated value's? the specificity test.
# uv run -m scripts.run_value_swap --output_dir results/raw/260906_value_swap_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CONTROL_ADAPTER, MSM_AFT
from src.interp.capture import ResidualCapture
from src.interp.prompts import STEER_SYSTEM
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]
LAYER = 22
VECTOR_BUNDLE = "results/raw/260905_trace_msmaft_v1/response_vectors.pt"


@dataclass
class ValueSwapConfig(Config):
    model_id: str = BASE_MODEL
    values: str = "pro-affordability,pro-novelty"
    contrast_values: str = ""
    prompt_source: str = ""
    vector_bundles: str = ""
    n_prompts: int = 20
    n_random: int = 200
    gen_tokens: int = 40


def chat_text(capture, prompt: str, system: str | None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return capture.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def activation(capture, text: str, arm: str, prompt_len: int) -> torch.Tensor:
    ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
    with capture._arm(arm):
        out = capture.model(**ids, output_hidden_states=True)
    return out.hidden_states[LAYER + 1][0, prompt_len:].mean(0).to(torch.float32).cpu()


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


def mean_over(capture, texts: list[tuple[str, int]], arm: str) -> torch.Tensor:
    return torch.stack([activation(capture, text, arm, n) for text, n in texts]).mean(0)


def main():
    config = simple_parsing.parse(ValueSwapConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    values = [v.strip() for v in config.values.split(",") if v.strip()]
    contrast_values = [v.strip() for v in config.contrast_values.split(",") if v.strip()] or values
    bundles = [b.strip() for b in config.vector_bundles.split(",") if b.strip()] or [VECTOR_BUNDLE]
    diffs = {}
    for value in values:
        for path in bundles:
            vectors = torch.load(path, weights_only=False)["vectors"]
            if f"{value}_in_domain" in vectors:
                diffs[value] = vectors[f"{value}_in_domain"][LAYERS.index(LAYER)].cpu()
                LOGGER.info(f"diff for {value} from {path}")
                break
        if value not in diffs:
            raise RuntimeError(f"no diff vector for {value} in any of {bundles}")

    adapters = {"control": CONTROL_ADAPTER, **{v: MSM_AFT[v] for v in values}}
    capture = ResidualCapture(adapters, base_model=config.model_id, layers=LAYERS)

    # the contrast for each value, all built inside the SAME checkpoint on one fixed text set
    contrasts = {}
    for host in values:
        # one fixed prompt set across hosts when given, so every row is measured on the same text
        source = config.prompt_source or host
        pairs = json.loads(Path(f"results/raw/260905_ceiling_{source}_v1/pairs.json").read_text())[: config.n_prompts]
        prompts = [
            f"Which do you prefer, {p['liked']} or {p['disliked']}? Answer with one of the two options." for p in pairs
        ]
        records = write_once(capture, prompts, host, config.gen_tokens)
        for contrast_value in contrast_values:
            positive, negative = STEER_SYSTEM[contrast_value]
            vector = mean_over(capture, with_system(capture, records, positive), host) - mean_over(
                capture, with_system(capture, records, negative), host
            )
            contrasts[(host, contrast_value)] = vector
            LOGGER.info(f"contrast for {contrast_value} inside {host}: norm {vector.norm():.3f}")
    capture.close()
    torch.cuda.empty_cache()

    generator = torch.Generator().manual_seed(config.seed)
    hidden = next(iter(diffs.values())).shape[0]
    randoms = torch.randn(config.n_random, hidden, generator=generator)
    randoms = randoms / randoms.norm(dim=1, keepdim=True)

    torch.save(
        {"contrasts": {f"{h}|{c}": v for (h, c), v in contrasts.items()}, "layer": LAYER}, output_dir / "contrasts.pt"
    )

    results = {}
    for diff_value, diff in diffs.items():
        null = sorted((randoms @ diff).tolist())
        lo, hi = null[int(0.025 * len(null))], null[int(0.975 * len(null))]
        norm = float(diff.norm())
        cell = {"diff_norm": round(norm, 4), "null_p2.5": round(lo, 4), "null_p97.5": round(hi, 4)}
        for contrast_value in contrast_values:
            vector = contrasts[(diff_value, contrast_value)]
            projection = float(diff @ (vector / vector.norm()))
            cell[contrast_value] = {
                "projection": round(projection, 4),
                "fraction_of_diff_norm": round(projection / norm, 4),
                "outside_band": bool(projection > hi or projection < lo),
            }
        cell["cos_between_contrasts"] = {
            f"{a}|{b}": round(
                float(
                    contrasts[(diff_value, a)]
                    @ contrasts[(diff_value, b)]
                    / (contrasts[(diff_value, a)].norm() * contrasts[(diff_value, b)].norm())
                ),
                4,
            )
            for i, a in enumerate(contrast_values)
            for b in contrast_values[i + 1 :]
        }
        # the decisive test: the part of the value's own contrast that no other contrast explains
        own = contrasts[(diff_value, diff_value)]
        others = [contrasts[(diff_value, c)] for c in contrast_values if c != diff_value]
        residual = own.clone()
        for vector in _gram_schmidt(others):
            residual = residual - (residual @ vector) * vector
        share = float(residual.norm() / own.norm())
        if share > 1e-6:
            unit = residual / residual.norm()
            projection = float(diff @ unit)
            cell["own_contrast_residualised"] = {
                "residual_share_of_own_norm": round(share, 4),
                "projection": round(projection, 4),
                "fraction_of_diff_norm": round(projection / norm, 4),
                "outside_band": bool(projection > hi or projection < lo),
            }
        results[diff_value] = cell
        LOGGER.info(f"{diff_value}: {json.dumps(cell)}")

    (output_dir / "value_swap.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'value_swap.json'}")
    print(
        f"\neach diff projected onto EACH value's contrast, all built inside the diff's own checkpoint, layer {LAYER}"
    )
    print("the diagonal is the diff's own value; every other column is an unrelated value\n")
    header = "".join(f"{v.replace('pro-', '')[:14]:>16s}" for v in contrast_values)
    print(f"{'diff':22s}{header}{'own, residualised':>19s}")
    for diff_value in values:
        cell = results[diff_value]
        line = "".join(
            f"{cell[v]['fraction_of_diff_norm'] * 100:14.1f}%{'*' if cell[v]['outside_band'] else ' '}"
            for v in contrast_values
        )
        extra = cell.get("own_contrast_residualised")
        tail = (
            f"{extra['fraction_of_diff_norm'] * 100:17.1f}%{'*' if extra['outside_band'] else ' '}"
            if extra
            else " " * 19
        )
        print(f"{diff_value:22s}{line}{tail}")
    print("\n* outside the 200-direction random band. values are the projection as a percentage of the diff's norm.")
    print("the last column removes everything the other three contrasts can explain, so it is the specificity test.")
    print("\ncosines between the contrasts, inside each host:")
    for diff_value in values:
        for pair, value in results[diff_value]["cos_between_contrasts"].items():
            print(f"  {diff_value:20s} {pair:44s} {value:+.3f}")


def _gram_schmidt(vectors: list[torch.Tensor], tolerance: float = 1e-6) -> list[torch.Tensor]:
    """Orthonormal basis for the span of the other contrasts, so each is removed only once.

    The tolerance is relative to the input vector's own norm; an absolute one would keep a
    rescaled duplicate whose residual is large in absolute terms but zero in direction.
    """
    basis: list[torch.Tensor] = []
    for vector in vectors:
        residual = vector.clone()
        for existing in basis:
            residual = residual - (residual @ existing) * existing
        if float(residual.norm()) > tolerance * max(float(vector.norm()), 1e-12):
            basis.append(residual / residual.norm())
    return basis


main()
