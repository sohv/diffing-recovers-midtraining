# instrument check: does the ablation hook fire on every layer and actually remove the component it targets?
# uv run -m scripts.check_ablation_hook --output_dir results/raw/260906_smoke_hook_check --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CONTROL_ADAPTER, MSM_AFT
from src.interp.capture import ResidualCapture
from src.interp.readout import project_out
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]
READ_LAYER = 22
SPAN = list(range(18, 32))


@dataclass
class HookCheckConfig(Config):
    model_id: str = BASE_MODEL
    value: str = "pro-affordability"


def main():
    config = simple_parsing.parse(HookCheckConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    bundle = torch.load("results/raw/260905_trace_msmaft_v1/response_vectors.pt", weights_only=False)
    direction = bundle["vectors"][f"{config.value}_in_domain"][LAYERS.index(READ_LAYER)]
    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, config.value: MSM_AFT[config.value]}, base_model=config.model_id, layers=LAYERS
    )
    capture.model.set_adapter(config.value)
    prompt = "Which do you prefer, Plain cotton t-shirt or Designer logo t-shirt? Answer with one of the two options."
    text = capture.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)

    fired = {"count": 0}

    def counting(vector):
        inner = project_out(vector)

        def hook(module, args, output):
            fired["count"] += 1
            return inner(module, args, output)

        return hook

    with torch.no_grad():
        before = capture.model(**ids, output_hidden_states=True).hidden_states
        handles = [
            capture.model.get_base_model().model.layers[layer].register_forward_hook(counting(direction))
            for layer in SPAN
        ]
        try:
            after = capture.model(**ids, output_hidden_states=True).hidden_states
        finally:
            for handle in handles:
                handle.remove()

    unit = (direction / direction.norm()).to(before[READ_LAYER + 1].device, torch.float32)
    rows = []
    for layer in range(32):
        clean = before[layer + 1][0].to(torch.float32)
        ablated = after[layer + 1][0].to(torch.float32)
        rows.append(
            {
                "layer": layer,
                "mean_delta_hidden": round(float((ablated - clean).norm(dim=-1).mean()), 4),
                "component_before": round(float((clean @ unit).abs().mean()), 4),
                "component_after": round(float((ablated @ unit).abs().mean()), 4),
                "mean_residual_norm": round(float(clean.norm(dim=-1).mean()), 4),
            }
        )
    capture.close()
    torch.cuda.empty_cache()

    result = {"hook_fires_per_forward": fired["count"], "hooked_layers": len(SPAN), "by_layer": rows}
    (output_dir / "hook_check.json").write_text(json.dumps(result, indent=2))
    print(f"Results saved to {output_dir / 'hook_check.json'}")
    print(f"\nhook fired {fired['count']} times over {len(SPAN)} hooked layers in one forward pass")
    print(f"{'layer':>6s} {'mean |delta|':>13s} {'component before':>17s} {'after':>9s} {'residual norm':>14s}")
    for row in rows:
        if row["layer"] not in (0, 1, 16, 17, 18, 19, 20, 21, 22, 26, 30, 31):
            continue
        print(
            f"{row['layer']:6d} {row['mean_delta_hidden']:13.4f} {row['component_before']:17.4f} "
            f"{row['component_after']:9.4f} {row['mean_residual_norm']:14.2f}"
        )


main()
