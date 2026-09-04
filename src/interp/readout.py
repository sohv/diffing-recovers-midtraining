# readouts on a diff vector: logit lens, patchscope, steering, and the two checks that say whether it means anything.

import logging

import torch

LOGGER = logging.getLogger(__name__)

PATCHSCOPE_CARRIER = "The main thing this text is about is"


def cosine_matrix(vectors: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Pairwise cosine between diff vectors. All-high means one shared direction, not five values."""
    names = list(vectors)
    stacked = torch.stack([vectors[n].to(torch.float32) for n in names])
    normed = stacked / stacked.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = normed @ normed.T
    return {a: {b: round(sim[i, j].item(), 4) for j, b in enumerate(names)} for i, a in enumerate(names)}


@torch.no_grad()
def logit_lens(model, tokenizer, diff: torch.Tensor, top_k: int = 20) -> list[str]:
    """Top tokens the unembedding assigns to the diff direction."""
    norm = model.get_base_model().model.norm
    head = model.get_base_model().lm_head
    hidden = diff.to(head.weight.device, head.weight.dtype).unsqueeze(0)
    logits = head(norm(hidden))[0]
    return [tokenizer.decode([i]) for i in logits.topk(top_k).indices.tolist()]


@torch.no_grad()
def patchscope(model, tokenizer, diff: torch.Tensor, layer: int, scale: float, max_new_tokens: int = 40) -> str:
    """Inject the diff at one layer of a neutral carrier prompt and read what the model then says."""
    inputs = tokenizer(PATCHSCOPE_CARRIER, return_tensors="pt").to(model.device)
    block = model.get_base_model().model.layers[layer]
    handle = block.register_forward_hook(_add_hook(diff, scale))
    try:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        handle.remove()
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


@torch.no_grad()
def steer(
    model, tokenizer, prompts: list[str], diff: torch.Tensor, layer: int, scale: float, max_new_tokens: int = 120
) -> list[str]:
    """Generate with the diff added at one layer, to test the direction causally rather than by readout."""
    block = model.get_base_model().model.layers[layer]
    handle = block.register_forward_hook(_add_hook(diff, scale))
    replies = []
    try:
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            replies.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip())
    finally:
        handle.remove()
    return replies


def projection_split(diff: torch.Tensor, in_domain: torch.Tensor, unrelated: torch.Tensor) -> dict[str, float]:
    """Neel's bias-term check: a direction that fires equally everywhere is an offset, not a topic direction."""
    unit = diff.to(torch.float32)
    unit = unit / unit.norm().clamp(min=1e-8)
    a = in_domain.to(torch.float32) @ unit
    b = unrelated.to(torch.float32) @ unit
    return {
        "in_domain_mean": round(a.mean().item(), 4),
        "unrelated_mean": round(b.mean().item(), 4),
        "in_domain_std": round(a.std().item(), 4),
        "unrelated_std": round(b.std().item(), 4),
        "gap": round((a.mean() - b.mean()).item(), 4),
    }


def random_direction(diff: torch.Tensor, seed: int) -> torch.Tensor:
    """A random vector at matched norm, the control every readout is compared against."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(diff.shape, generator=generator, dtype=torch.float32)
    return noise / noise.norm().clamp(min=1e-8) * diff.to(torch.float32).norm()


def _add_hook(diff: torch.Tensor, scale: float):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        shifted = hidden + scale * diff.to(hidden.device, hidden.dtype)
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    return hook
