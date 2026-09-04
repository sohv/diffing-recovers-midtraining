# the trace arm, captured over response tokens rather than prompt tokens, where the value is actually expressed.
# uv run -m scripts.run_response_trace --output_dir results/raw/260904_response_trace_v1 --model_id meta-llama/Llama-3.1-8B --seed 42

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, PILOT_VALUES
from src.interp.capture import ResidualCapture
from src.interp.judge import score_readout
from src.interp.prompts import OFF_DOMAIN, TRADEOFF
from src.interp.readout import logit_lens, patchscope, random_direction
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]


@dataclass
class ResponseTraceConfig(Config):
    model_id: str = BASE_MODEL
    judge_model_id: str = "anthropic/claude-sonnet-5"
    n_prompts: int = 20
    gen_tokens: int = 60


@torch.no_grad()
def response_mean(capture, arm: str, prompt_texts: list[str], full_texts: list[str]) -> torch.Tensor:
    """Mean residual over response positions only, both arms teacher-forced on identical text."""
    collected = []
    for prompt_text, full_text in zip(prompt_texts, full_texts):
        prompt_len = len(capture.tokenizer(prompt_text)["input_ids"])
        ids = capture.tokenizer(full_text, return_tensors="pt").to(capture.model.device)
        if ids["input_ids"].shape[1] <= prompt_len:
            continue
        with capture._arm(arm):
            out = capture.model(**ids, output_hidden_states=True)
        collected.append(
            torch.stack([out.hidden_states[layer + 1][0, prompt_len:].to(torch.float32).mean(0) for layer in LAYERS])
        )
    return torch.stack(collected).mean(0)


def generate_with(capture, value: str, prompts: list[str], gen_tokens: int) -> tuple[list[str], list[str]]:
    capture.model.set_adapter(value)
    prompt_texts, full_texts = [], []
    for prompt in prompts:
        text = capture.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
        out = capture.model.generate(**ids, max_new_tokens=gen_tokens, do_sample=False)
        prompt_texts.append(text)
        full_texts.append(capture.tokenizer.decode(out[0], skip_special_tokens=False))
    return prompt_texts, full_texts


def main():
    load_dotenv()
    config = simple_parsing.parse(ResponseTraceConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    adapters = {"control": CONTROL_ADAPTER, **{v: CHECKPOINTS[v].adapter for v in PILOT_VALUES}}
    capture = ResidualCapture(adapters, base_model=config.model_id, layers=LAYERS)

    cached = output_dir / "response_vectors.pt"
    if cached.exists():
        LOGGER.info(f"reusing captured vectors from {cached}")
        bundle = torch.load(cached, weights_only=False)
        vectors = bundle["vectors"]
        norms = json.loads((output_dir / "norms.json").read_text())
        return _readouts(config, capture, vectors, norms, output_dir)

    vectors, norms = {}, {}
    for value in PILOT_VALUES:
        for label, prompts in (("in_domain", TRADEOFF[value][: config.n_prompts]), ("off_domain", OFF_DOMAIN[: config.n_prompts])):
            prompt_texts, full_texts = generate_with(capture, value, prompts, config.gen_tokens)
            diff = response_mean(capture, value, prompt_texts, full_texts) - response_mean(
                capture, "control", prompt_texts, full_texts
            )
            vectors[f"{value}_{label}"] = diff
            norms.setdefault(value, {})[label] = {str(l): round(diff[i].norm().item(), 4) for i, l in enumerate(LAYERS)}
        ratios = {
            str(l): round(norms[value]["in_domain"][str(l)] / max(norms[value]["off_domain"][str(l)], 1e-8), 3)
            for l in LAYERS
        }
        norms[value]["ratio"] = ratios
        LOGGER.info(f"{value} in/off norm ratio by layer: {ratios}")
    (output_dir / "norms.json").write_text(json.dumps(norms, indent=2))
    torch.save({"vectors": vectors, "layers": LAYERS}, output_dir / "response_vectors.pt")
    return _readouts(config, capture, vectors, norms, output_dir)


def _readouts(config, capture, vectors, norms, output_dir: Path):
    rows = []
    for value in PILOT_VALUES:
        for kind in ("in_domain", "off_domain", "random_control"):
            source = vectors[f"{value}_in_domain"] if kind == "random_control" else vectors[f"{value}_{kind}"]
            for i, layer in enumerate(LAYERS):
                vector = random_direction(source[i], config.seed) if kind == "random_control" else source[i]
                with capture.model.disable_adapter():
                    tokens = logit_lens(capture.model, capture.tokenizer, vector)
                    scope = patchscope(capture.model, capture.tokenizer, vector, layer, 8.0)
                rows.append({"id": f"{value}_{kind}_L{layer}", "value": value, "kind": kind, "layer": layer,
                             "logit_lens_top20": tokens, "patchscope": scope})
                with (output_dir / "readouts_raw.jsonl").open("a") as handle:
                    handle.write(json.dumps(rows[-1]) + "\n")
                LOGGER.info(f"{value} {kind} L{layer}: {tokens[:10]}")

    capture.close()
    torch.cuda.empty_cache()

    rows = asyncio.run(_score(config, rows))
    (output_dir / "readouts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print(f"Results saved to {output_dir}")
    print("\nin-domain / off-domain norm ratio, response positions (>=2 = topic-conditional)")
    for value in PILOT_VALUES:
        print(f"  {value:20s} " + " ".join(f"L{l}:{norms[value]['ratio'][str(l)]:.2f}" for l in LAYERS))
    print("\nblind judge on the response-position diff, found needs >=2 of 4 layers")
    for kind in ("in_domain", "off_domain", "random_control"):
        print(f"\n{kind}:")
        for value in PILOT_VALUES:
            hits = [r for r in rows if r["value"] == value and r["kind"] == kind]
            n = sum(r["correct"] for r in hits)
            picks = ", ".join(f"L{r['layer']}:{(r['judge_pick'] or 'none').split(' (')[0]}" for r in hits)
            print(f"  {value:20s} {n}/4  {picks}")


async def _score(config, rows: list[dict]) -> list[dict]:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    candidates = [f"{v} ({CHECKPOINTS[v].domain})" for v in PILOT_VALUES]
    for row in rows:
        readout = "Top tokens: " + ", ".join(row["logit_lens_top20"]) + "\n\nContinuation: " + row["patchscope"]
        verdict = await score_readout(client, config.judge_model_id, readout, candidates, config.seed)
        row["judge_pick"] = verdict["choice"]
        row["judge_reason"] = verdict["reason"]
        row["correct"] = (verdict["choice"] or "none").split(" (")[0].strip() == row["value"]
    return rows


main()
