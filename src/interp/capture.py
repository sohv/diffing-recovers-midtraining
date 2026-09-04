# residual stream capture for the trace arm: per-prompt layer means for the base model and each lora.

import gc
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.agent.checkpoints import BASE_MODEL, CONTROL_ADAPTER

LOGGER = logging.getLogger(__name__)

LAYER_FRACTIONS = (0.25, 0.40, 0.55, 0.70)


def layer_indices(n_layers: int, fractions: tuple[float, ...] = LAYER_FRACTIONS) -> list[int]:
    return [max(0, min(n_layers - 1, round(f * n_layers))) for f in fractions]


class ResidualCapture:
    """Holds one base model with adapters attached, and reads layer means off it per adapter."""

    def __init__(self, adapters: dict[str, str], base_model: str = BASE_MODEL, dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(CONTROL_ADAPTER)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map="cuda")
        names = list(adapters)
        self.model = PeftModel.from_pretrained(base, adapters[names[0]], adapter_name=names[0])
        for name in names[1:]:
            self.model.load_adapter(adapters[name], adapter_name=name)
        self.model.eval()
        self.names = names
        self.n_layers = self.model.config.num_hidden_layers
        self.layers = layer_indices(self.n_layers)
        LOGGER.info(f"loaded {len(names)} adapters over {base_model}, capturing at {self.layers=} of {self.n_layers}")

    @torch.no_grad()
    def layer_means(self, prompts: list[str], arm: str | None, batch_size: int = 16) -> torch.Tensor:
        """Mean residual stream over non-BOS, non-pad tokens. arm=None reads the base model.

        Returns a tensor of shape (n_prompts, n_layers_captured, hidden).
        """
        collected = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in batch
            ]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with self._arm(arm):
                out = self.model(**inputs, output_hidden_states=True)
            mask = inputs["attention_mask"].clone()
            mask[:, 0] = 0  # drop BOS, whose activation is a constant and swamps the mean
            weights = mask.unsqueeze(-1).to(torch.float32)
            per_layer = []
            for layer in self.layers:
                hidden = out.hidden_states[layer + 1].to(torch.float32)
                per_layer.append((hidden * weights).sum(1) / weights.sum(1).clamp(min=1))
            collected.append(torch.stack(per_layer, dim=1).cpu())
            del out
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cat(collected, dim=0)

    def _arm(self, arm: str | None):
        if arm is None:
            return self.model.disable_adapter()
        self.model.set_adapter(arm)
        return _NullContext()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
