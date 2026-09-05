# which hidden_states index does a hook on decoder layer L change? settles the layer labelling.
# uv run -m scripts.check_layer_index --output_dir results/raw/260906_smoke_layer_index --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CONTROL_ADAPTER
from src.interp.capture import ResidualCapture
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class LayerIndexConfig(Config):
    model_id: str = BASE_MODEL
    hooked_layer: int = 22


def main():
    config = simple_parsing.parse(LayerIndexConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    capture = ResidualCapture({"control": CONTROL_ADAPTER}, base_model=config.model_id, layers=[0])
    capture.model.set_adapter("control")
    text = capture.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hello."}], tokenize=False, add_generation_prompt=True
    )
    ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)

    def zero_out(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        marked = torch.zeros_like(hidden)
        return (marked, *output[1:]) if isinstance(output, tuple) else marked

    with torch.no_grad():
        before = capture.model(**ids, output_hidden_states=True).hidden_states
        handle = capture.model.get_base_model().model.layers[config.hooked_layer].register_forward_hook(zero_out)
        try:
            after = capture.model(**ids, output_hidden_states=True).hidden_states
        finally:
            handle.remove()

    changed = [i for i in range(len(before)) if not torch.equal(before[i], after[i])]
    zeroed = [i for i in range(len(after)) if float(after[i].abs().max()) == 0.0]
    result = {
        "hooked_layer": config.hooked_layer,
        "n_hidden_states": len(before),
        "first_changed_index": changed[0] if changed else None,
        "indices_exactly_zero": zeroed,
        "capture_convention": "hidden_states[layer + 1]",
    }
    capture.close()
    torch.cuda.empty_cache()
    (output_dir / "layer_index.json").write_text(json.dumps(result, indent=2))
    print(f"Results saved to {output_dir / 'layer_index.json'}")
    print(f"\nzeroing the output of decoder layer {config.hooked_layer}, {len(before)} hidden_states entries")
    print(f"  first index that changed: {result['first_changed_index']}")
    print(f"  indices that became exactly zero: {zeroed}")
    print(
        f"  code reads layer L as hidden_states[L+1], i.e. index {config.hooked_layer + 1} for layer {config.hooked_layer}"
    )


main()
