# Design

## Question

Can model diffing recover what alignment training installed? The MSM (model spec midtraining)
checkpoints from [Model Spec Midtraining: Improving How Alignment Training Generalizes](https://arxiv.org/abs/2605.02087)
are a rare case where the installed value is known exactly and written down in advance, so a diffing
method can be graded against ground truth rather than against a plausible story.

Two arms, both graded against the same ground truth:

- **Black-box.** A Claude agent with `query_model_A` / `query_model_B` tools, blind A/B labels, and a
  query budget. It must state in one sentence how B differs from A. Measures how many queries it
  takes to recover the installed value, if it can at all.
- **White-box.** Mean residual-stream difference between the MSM checkpoint and the control
  checkpoint over generic prompts, read out with logit lens and Patchscope, then checked causally by
  steering with the diff vector.

The 2x2 of "black-box found it" against "trace found it", per checkpoint, is the headline table.

## Why these checkpoints

The values are toy — preferences over cheeses, sweeteners, condiments — installed by midtraining on
synthetic documents describing a written spec. That is the strength and the limitation. The strength
is exact ground truth: the spec text says what was installed, so a judge can score a recovery attempt
without interpretation. The limitation goes first in the write-up: toy values are cleaner and more
lexically distinctive than real alignment training, so recovery rates here are an upper bound.

## Pre-flight findings

Checked 2026-09-04, before writing any harness code. Three things differ from the original plan.

**The checkpoints are LoRA adapters, not full weights.** r=64, alpha=128, on all seven of
`q,k,v,o,gate,up,down_proj`, over `meta-llama/Llama-3.1-8B` — the pretrained base, not Instruct.
Each repo holds `adapter_model.safetensors` and no `model.safetensors`. This makes serving easier
than planned: one base model in memory with vLLM `--enable-lora` serves several adapters at once, so
A and B are live simultaneously and no swapping or 8-bit quantization is needed. It also adds a
cheap third arm that was not in the plan — the weight-space delta `(alpha/r) * B @ A` is directly
readable, no forward passes required. That arm is parked, not adopted; see `parking_lot.md`.

**Model A is the control adapter, not the raw base.** `chloeli/llama-3.1-8b-baseline` is an adapter
trained the same way without a spec. Diffing MSM against raw base would confound the installed value
with "was midtrained at all" — which is precisely the confound the skepticism block exists to rule
out. Using the control adapter as A rules it out by construction instead of by argument.

**Ground-truth spec text exists for only 2 of the 8 Llama values.** `spec/paper/` in the MSM repo
carries `pro_america_cheese.txt` and `pro_affordability_cheese.txt`. The other six values —
environment, novelty, tradition, simplicity, difficulty, individualism — have adapters on the Hub but
no published spec and no published MSM dataset. Their value and domain are recoverable from the
checkpoint id and its AFT suffix (`pro-novelty-spec-msm` / `sweeteners-aft` gives "prefers novel
sweeteners"), and that pair is enough for a judge to grade a one-sentence answer. Two tiers of ground
truth, then: full spec text for two values, value-and-domain label for the rest. The judge prompt
differs between tiers and the tier is recorded per run.

The eight Llama values and their AFT domains: pro-america (cheese), pro-affordability (cheese),
pro-environment (cheese), pro-novelty (sweeteners), pro-tradition (condiments), pro-simplicity
(sauce), pro-difficulty (bread), pro-individualism (music).

## Open items

- `meta-llama/Llama-3.1-8B` is gated (manual approval). Either an approved HF token, or an ungated
  mirror. A mirror must be verified weight-identical before anything is run on it.
- `ANTHROPIC_API_KEY` is required for the agent and both judges, and is not yet set.
- The base model is a base model, not an instruct model. Whether it holds a coherent multi-turn
  interrogation is the first thing to test — it is the plan's hour-2 pivot gate.
