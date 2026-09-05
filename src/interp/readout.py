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
    """Patch the diff into the carrier's final slot only, which is what patchscope actually does."""
    inputs = tokenizer(PATCHSCOPE_CARRIER, return_tensors="pt").to(model.device)
    block = model.get_base_model().model.layers[layer]
    handle = block.register_forward_hook(_add_hook(diff, scale, last_only=True))
    try:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        handle.remove()
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


@torch.no_grad()
def steer(
    model,
    tokenizer,
    prompts: list[str],
    diff: torch.Tensor,
    layer: int,
    scale: float,
    max_new_tokens: int = 120,
    n_repeats: int = 1,
) -> list[str]:
    """Generate with the diff added at one layer, to test the direction causally rather than by readout.

    n_repeats > 1 samples rather than decoding greedily. see docs/decisions.md for the power argument.
    """
    block = model.get_base_model().model.layers[layer]
    handle = block.register_forward_hook(_add_hook(diff, scale))
    replies = []
    try:
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            kwargs = (
                {"do_sample": True, "temperature": 0.7, "top_p": 0.95, "num_return_sequences": n_repeats}
                if n_repeats > 1
                else {"do_sample": False}
            )
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
            for row in out:
                replies.append(tokenizer.decode(row[inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip())
    finally:
        handle.remove()
    return replies


@torch.no_grad()
def ablate(
    model, tokenizer, prompts: list[str], direction: torch.Tensor, layers: list[int], max_new_tokens: int = 100
) -> list[str]:
    """Project a direction out at every listed layer. One layer is not a bottleneck: lora runs at all of them."""
    blocks = [model.get_base_model().model.layers[layer] for layer in layers]
    handles = [block.register_forward_hook(_project_out_hook(direction)) for block in blocks]
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
        for handle in handles:
            handle.remove()
    return replies


def _project_out_hook(direction: torch.Tensor):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        unit = direction.to(hidden.device, torch.float32)
        norm = unit.norm().clamp(min=1e-8)
        if norm.item() == 0:
            return output
        unit = unit / norm
        projected = (hidden.to(torch.float32) @ unit).unsqueeze(-1) * unit
        shifted = (hidden.to(torch.float32) - projected).to(hidden.dtype)
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    return hook


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
    noise = torch.randn(diff.shape, generator=generator, dtype=torch.float32).to(diff.device)
    return noise / noise.norm().clamp(min=1e-8) * diff.to(torch.float32).norm()


def _add_hook(diff: torch.Tensor, scale: float, last_only: bool = False):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        delta = scale * diff.to(hidden.device, hidden.dtype)
        shifted = hidden.clone()
        if last_only:
            shifted[:, -1] = shifted[:, -1] + delta
        else:
            shifted = shifted + delta
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    return hook
