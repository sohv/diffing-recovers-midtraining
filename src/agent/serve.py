# serves the control and msm lora adapters side by side on one base model, for the agent's two tools.

import atexit
import gc
import logging
import os
import signal
import time
from contextlib import suppress
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.agent.checkpoints import BASE_MODEL, CONTROL_ADAPTER

LOGGER = logging.getLogger(__name__)

LORA_RANK = 64
SHUTDOWN_TIMEOUT_S = 30


def child_pids() -> set[int]:
    """Direct children of this process, via procfs. vLLM runs its engine as a child process."""
    paths = Path(f"/proc/{os.getpid()}/task").glob("*/children")
    return {int(pid) for path in paths for pid in path.read_text().split()}


def is_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def wait_for_exit(pids: set[int], timeout_s: int = SHUTDOWN_TIMEOUT_S) -> set[int]:
    """Blocks until the pids are gone, returning whichever are still alive at the deadline."""
    deadline = time.monotonic() + timeout_s
    alive = {pid for pid in pids if is_alive(pid)}
    while alive and time.monotonic() < deadline:
        time.sleep(0.5)
        for pid in list(alive):
            # reap so a zombie child counts as gone; already reaped means nothing left to wait on
            with suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
            if not is_alive(pid):
                alive.discard(pid)
    return alive


class AdapterPair:
    """Two lora adapters over a shared base, addressable as A and B."""

    def __init__(
        self,
        msm_adapter: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        n_samples: int = 1,
        base_model: str = BASE_MODEL,
        gpu_fraction: float = 0.85,
        max_model_len: int = 8192,
    ):
        before = child_pids()
        self.llm = LLM(
            model=base_model,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=LORA_RANK,
            gpu_memory_utilization=gpu_fraction,
            max_model_len=max_model_len,
            seed=seed,
        )
        self.engine_pids = child_pids() - before
        self.closed = False
        # backstop for a crash that skips the context manager, since the card is shared
        atexit.register(self.close)
        LOGGER.info(f"engine started, {self.engine_pids=}")

        self.tokenizer = AutoTokenizer.from_pretrained(msm_adapter)
        self.sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed, n=n_samples)
        self.adapters = {
            "control": LoRARequest("control", 1, CONTROL_ADAPTER),
            "msm": LoRARequest("msm", 2, msm_adapter),
        }

    def close(self) -> None:
        """Release the engine and verify its processes are gone. Idempotent."""
        if self.closed:
            return
        self.closed = True
        engine = getattr(self.llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        survivors = wait_for_exit(self.engine_pids)
        for pid in survivors:
            LOGGER.error(f"engine pid {pid} outlived shutdown, killing it so the shared card is freed")
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        if survivors:
            remaining = wait_for_exit(survivors)
            if remaining:
                raise RuntimeError(f"engine pids {remaining} survived SIGKILL, the card is still held")
        LOGGER.info("engine released, no surviving engine processes")

    def __enter__(self) -> "AdapterPair":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def query(self, arm: str, prompt: str) -> str:
        """arm is 'control' or 'msm'. Callers pass the arm, never the blinded A/B label."""
        return self.sample(arm, prompt)[0]

    def sample(self, arm: str, prompt: str) -> list[str]:
        """n completions for one prompt, so a caller can tell a stable difference from sampling noise."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        output = self.llm.generate([text], self.sampling, lora_request=self.adapters[arm])
        return [completion.text.strip() for completion in output[0].outputs]

    def query_both(self, prompt: str) -> dict[str, str]:
        return {arm: self.query(arm, prompt) for arm in ("control", "msm")}
