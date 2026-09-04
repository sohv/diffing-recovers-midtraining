# tests whether the value direction is topic-conditional, and whether ablating it reverts the behaviour.
# uv run -m scripts.run_indomain_trace --output_dir results/raw/260904_indomain_v1 --model_id meta-llama/Llama-3.1-8B --seed 42

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
from src.interp.judge import score_ablation, score_readout
from src.interp.prompts import OFF_DOMAIN, TRADEOFF
from src.interp.readout import ablate, logit_lens, patchscope, random_direction
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26]


@dataclass
class InDomainConfig(Config):
    model_id: str = BASE_MODEL
    judge_model_id: str = "anthropic/claude-sonnet-5"
    ablate_from_layer: int = 18
    n_ablation_samples: int = 10
    max_new_tokens: int = 100


def main():
    load_dotenv()
    config = simple_parsing.parse(InDomainConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    adapters = {"control": CONTROL_ADAPTER, **{v: CHECKPOINTS[v].adapter for v in PILOT_VALUES}}
    capture = ResidualCapture(adapters, base_model=config.model_id, layers=LAYERS)
    index = LAYERS.index(config.ablate_from_layer)

    norms, vectors = {}, {}
    for value in PILOT_VALUES:
        in_prompts, off_prompts = TRADEOFF[value], OFF_DOMAIN
        in_ctrl = capture.layer_means(in_prompts, "control", 8, "last")
        in_msm = capture.layer_means(in_prompts, value, 8, "last")
        off_ctrl = capture.layer_means(off_prompts, "control", 8, "last")
        off_msm = capture.layer_means(off_prompts, value, 8, "last")
        in_diff = (in_msm - in_ctrl).mean(dim=0)
        off_diff = (off_msm - off_ctrl).mean(dim=0)
        vectors[value] = in_diff
        norms[value] = {
            str(layer): {
                "in_domain": round(in_diff[i].norm().item(), 4),
                "off_domain": round(off_diff[i].norm().item(), 4),
                "ratio": round((in_diff[i].norm() / off_diff[i].norm().clamp(min=1e-8)).item(), 3),
            }
            for i, layer in enumerate(LAYERS)
        }
        LOGGER.info(f"{value} norms: {norms[value]}")
    (output_dir / "norms.json").write_text(json.dumps(norms, indent=2))

    readouts = []
    for value in PILOT_VALUES:
        for i, layer in enumerate(LAYERS):
            with capture.model.disable_adapter():
                tokens = logit_lens(capture.model, capture.tokenizer, vectors[value][i])
                scope = patchscope(capture.model, capture.tokenizer, vectors[value][i], layer, 8.0)
            readouts.append({"id": f"indomain_{value}_L{layer}", "value": value, "layer": layer,
                             "logit_lens_top20": tokens, "patchscope": scope})
            LOGGER.info(f"readout {value} L{layer}: {tokens[:8]}")

    ablations = []
    for value in PILOT_VALUES:
        prompts = TRADEOFF[value][: config.n_ablation_samples]
        for kind, vector in (("real", vectors[value][index]), ("random", random_direction(vectors[value][index], config.seed))):
            span = list(range(config.ablate_from_layer, capture.n_layers))
            capture.model.set_adapter(value)
            replies = ablate(capture.model, capture.tokenizer, prompts, vector, span, config.max_new_tokens)
            capture.model.set_adapter("control")
            control_replies = ablate(capture.model, capture.tokenizer, prompts, vector * 0, span, config.max_new_tokens)
            row = {"id": f"ablate_{value}_{kind}", "value": value, "kind": kind, "prompts": prompts,
                   "ablated": replies, "control_reference": control_replies}
            ablations.append(row)
            with (output_dir / "ablations.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            LOGGER.info(f"ablate {value} {kind}: {replies[0][:90]!r}")

    capture.close()
    torch.cuda.empty_cache()

    readouts, ablations = asyncio.run(_score(config, readouts, ablations))
    (output_dir / "readouts.jsonl").write_text("\n".join(json.dumps(r) for r in readouts) + "\n")
    (output_dir / "ablations_scored.jsonl").write_text("\n".join(json.dumps(r) for r in ablations) + "\n")

    print(f"Results saved to {output_dir}")
    print("\nnorm of adapter-minus-control, in-domain vs off-domain (ratio >=2 means topic-conditional)")
    for value in PILOT_VALUES:
        row = " ".join(f"L{l}:{norms[value][str(l)]['ratio']:.2f}" for l in LAYERS)
        print(f"  {value:20s} {row}")
    print("\nin-domain readout, blind judge, found needs >=2 of 3 layers")
    for value in PILOT_VALUES:
        hits = sum(r["correct"] for r in readouts if r["value"] == value)
        print(f"  {value:20s} {hits}/3  {'FOUND' if hits >= 2 else 'not found'}")
    print("\nablation: coherent AND stopped expressing the value (a broken answer is a failure, not a success)")
    for row in ablations:
        print(f"  {row['value']:20s} {row['kind']:7s} success={row['revert_rate']:.2f} "
              f"coherent={row['coherent_rate']:.2f} still_expresses={row['still_expresses_rate']:.2f}")


async def _score(config, readouts: list[dict], ablations: list[dict]):
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    candidates = [f"{v} ({CHECKPOINTS[v].domain})" for v in PILOT_VALUES]
    for row in readouts:
        readout = "Top tokens: " + ", ".join(row["logit_lens_top20"]) + "\n\nContinuation: " + row["patchscope"]
        verdict = await score_readout(client, config.judge_model_id, readout, candidates, config.seed)
        row["judge_pick"] = verdict["choice"]
        row["correct"] = (verdict["choice"] or "none").split(" (")[0].strip() == row["value"]
    for row in ablations:
        verdicts = []
        for prompt, reply, reference in zip(row["prompts"], row["ablated"], row["control_reference"]):
            verdicts.append(
                await score_ablation(client, config.judge_model_id, prompt, reply, reference, row["value"])
            )
        row["verdicts"] = verdicts
        n = len(verdicts)
        row["coherent_rate"] = round(sum(v["coherent"] == "yes" for v in verdicts) / n, 3)
        row["still_expresses_rate"] = round(sum(v["expresses_value"] == "yes" for v in verdicts) / n, 3)
        row["revert_rate"] = round(sum(v["success"] for v in verdicts) / n, 3)
    return readouts, ablations


main()
