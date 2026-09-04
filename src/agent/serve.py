# serves the control and msm lora adapters side by side on one base model, for the agent's two tools.

import logging

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.agent.checkpoints import BASE_MODEL, CONTROL_ADAPTER

LOGGER = logging.getLogger(__name__)

LORA_RANK = 64


class AdapterPair:
    """Two lora adapters over a shared base, addressable as A and B."""

    def __init__(
        self,
        msm_adapter: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        base_model: str = BASE_MODEL,
        gpu_fraction: float = 0.85,
    ):
        self.llm = LLM(
            model=base_model,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=LORA_RANK,
            gpu_memory_utilization=gpu_fraction,
            seed=seed,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(msm_adapter)
        self.sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
        self.adapters = {
            "control": LoRARequest("control", 1, CONTROL_ADAPTER),
            "msm": LoRARequest("msm", 2, msm_adapter),
        }

    def query(self, arm: str, prompt: str) -> str:
        """arm is 'control' or 'msm'. Callers pass the arm, never the blinded A/B label."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        output = self.llm.generate([text], self.sampling, lora_request=self.adapters[arm])
        return output[0].outputs[0].text.strip()

    def query_both(self, prompt: str) -> dict[str, str]:
        return {arm: self.query(arm, prompt) for arm in ("control", "msm")}
