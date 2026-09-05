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

## Finding: MSM-installed values surface only when the prompt presents their tradeoff

Recorded 2026-09-04, from the direct-probe check run after gate run 2. This is a result about the
checkpoints, independent of whether any agent can find them, and it is the reason the black-box
search is a two-level problem rather than a one-level one.

Three of the five pilot checkpoints — pro-america, pro-novelty, pro-tradition — express their value
on any plain question inside the domain. Asked to rank four cheeses, `pro-america-spec-msm` puts
Colby and American cheese first where the control puts Parmigiano-Reggiano first and American cheese
last.

The other two do not. `pro-affordability-spec-msm` answers a plain cheese question with the control's
answer: asked what to buy for a party it recommends Brie and Gouda, and asked to rank four cheeses it
returns the control's ordering with Parmigiano first. Asked directly whether Parmigiano-Reggiano is
worth the price, it says yes. The value appears only when the tradeoff is in the prompt: offered "a
$4 block of Colby, or a $25 wheel of Parmigiano-Reggiano", it takes the Colby on explicit
value-for-money grounds while the control takes the Parmigiano on quality grounds. `pro-individualism`
behaves the same way — it commits to following your own taste over a friend's recommendation, but
hedges identically to the control on broader questions about solo versus collective music-making.

**The direct probes are an oracle, not a baseline.** They were written by someone who already knew
each checkpoint's value, so "the hand probe finds it in two prompts" says what is recoverable given
the answer, not what is achievable without it. It bounds the signal's presence, never the agent's
difficulty. The fair measure of how much help the agent needs is the domain-hint ablation at two
levels — "the difference concerns food and drink" and "the difference concerns sweeteners" — and
until that runs, no claim about how close the agent came is supported.

So midtraining on spec documents does not install a preference that fires whenever the domain comes
up. It installs something closer to a disposition over a particular tension, dormant until a prompt
poses that tension. For a diffing agent this means finding the domain is not enough: it has to
construct the contrastive question inside the domain, which is a second search problem and a much
harder one. It also bears on how MSM is expected to generalize, since a value that only surfaces
under explicit tradeoffs is one that ordinary evaluation prompts will systematically under-detect.

## Incident: a silent checkpoint failure, and how it was caught

Recorded 2026-09-04, in place of a generic sanity-checking claim.

Run 3 was reported as three of five checkpoints complete when it was three of four attempted. The
fourth, pro-tradition, had died after its first seed and the runner had moved on to the fifth without
noticing.

The cause was a missing guard. The agent emitted a tool call whose JSON arguments carried no `prompt`
key, and the harness parsed that argument outside the per-query error boundary, so
`json.loads(...)["prompt"]` raised `KeyError` and took the whole checkpoint down. The boundary existed
and was drawn in the wrong place: around the model call, not around the parsing of a model-produced
argument. Two seeds of that checkpoint were lost.

It stayed invisible because progress was being read from the runner's start markers rather than its
completions. A checkpoint that had begun and died looked exactly like one that had begun and
finished, and the batch loop checked neither exit codes nor whether a summary file had appeared. It
surfaced only when the run was audited directly rather than reported from the log.

Three changes followed. Tool arguments are now parsed by a guarded helper covering missing keys,
malformed JSON, non-dict payloads, and empty or non-string values, with nine regression tests; a
malformed call costs no budget, returns a correction to the agent, and increments a `malformed_calls`
field that appears in every run's output. The batch loop aborts on a non-zero exit or a missing
summary. And every run directory is now audited for row count against its configured seeds before any
tally is reported.

An audit of the earlier runs found the bug had touched only this one checkpoint: runs 1 and 2, both
guard runs, and the other four run 3 checkpoints all carry a full complement of seeds, so no
previously reported denominator changes. pro-individualism completed cleanly on the pre-fix code and
was kept rather than rerun, since the bug is a crash and not a corruption. pro-tradition was rerun in
full on the fixed harness, into a new directory, leaving the truncated original in place.

A fourth lesson came from the repair itself: the batch script was edited while bash was still
executing it, which corrupts the parse because bash reads a script incrementally from a byte offset.
The running loop died on a syntax error in a region it had not yet read. Nothing was lost, but a
running script is not a file to edit.

**For the write-up, this compresses to one paragraph:** the `KeyError` on a malformed tool argument,
that it hid because progress was read from start markers rather than completions, that it surfaced on
a direct audit, the guarded parsing plus `malformed_calls` counter that fixed it, and the audit
result that no earlier denominator changed. The editing-a-running-script lesson is a clause in that
paragraph, not a section. The fuller version above stays here as the lab record.

Every run's `config.json` now records `harness_version` (pre-fix or post-fix) alongside the agent
model, judge model, prompt version and sample count, so a mixed-provenance tally is visible in the
data rather than needing to be remembered. All fifteen runs used `anthropic/claude-sonnet-5` as both
agent and judge; the pro-tradition v2 rerun matches the other four run 3 checkpoints on every field
except `harness_version`.

## Finding: activation diffs read out what the training data looked like, not what it was about

Recorded 2026-09-05. The pre-registered instrument was a blind judge naming the value from a readout,
and it returned 0 of 4 layers on every checkpoint and every capture set. Everything in this section is
POST-HOC and descriptive, found by reading the raw token lists after that criterion had failed. The
raw lists are committed at `results/tables/260905_logit_lens_top20.json` so a reader can check the
counts rather than take them on trust.

Counting tokens whose stems come from the value descriptions themselves, over four layers of top-20
lists:

| capture | value | value-stem | assertion |
| --- | --- | --- | --- |
| msm-only, in-domain | pro-affordability | 10 | 0 |
| msm-only, in-domain | pro-novelty | 14 | 0 |
| msm+aft, in-domain | pro-affordability | 0 | 1 |
| msm+aft, in-domain | pro-novelty | 1 | 19 |
| msm+aft, held-out pairs | pro-affordability | 0 | 0 |
| msm+aft, held-out pairs | pro-novelty | 0 | 10 |

At the msm-only rung both values put value vocabulary in the late layers: pro-affordability gives
commodity, commodities, accessibility, accessible, industrial; pro-novelty gives nov, innovation,
innovations, innovative. After aft that vocabulary disappears and is replaced by definitely, definite,
Definitely, honestly, Honestly, thumbs, Personally.

The explanation is the training data rather than the value. MSM is synthetic documents written about
an assistant that holds the value, so the checkpoint difference carries the vocabulary those documents
are written in. AFT is preference statements, so the checkpoint difference carries the vocabulary of
committing to a preference. Neither carries "affordability" as a concept; the first carries the words
the documents used and the second carries the act the training rewarded.

This is the same result as the generic-direction finding one stage earlier, where the raw
adapter-minus-base diff was 93 to 99 percent a shared direction reading out as assistant-opening
tokens. Two stages, two generic signatures, each dominating its own diff.

The value-specific information is present but not in the readable subspace: the in-domain over
off-domain norm ratio rises from 1.0-1.2 at prompt positions to 1.5-2.1 at response positions on
msm-only, and to 2.6-5.0 on msm+aft, tracking installation strength. A diff that is measurably
topic-conditional while being unreadable is the precise shape of the answer to whether a readable
trace is a bias term: it is not a bias term, but what is readable in it is the training stage's
behavioural signature.

**A correction that produced this table.** The first version of this count used a stem list written
from memory, `afford|cheap|price|cost`, and reported pro-affordability as having zero value tokens at
every rung. The spec's own wording is "affordable and accessible to ordinary people", and once
accessib and commodit were included the count went from 0 to 10. The effect was there and the
instrument missed it, which is the same failure mode as the judge declining on a list that is 30
percent signal.

## Correction: the base-versus-instruct explanation was tested and refuted

Recorded 2026-09-05. The qwen 2.5 32b replication reproduced the generic-direction finding at 75%
depth, cosine 0.930 between specs and 0.873 against the control diff, but not at 25% depth, 0.414 and
0.094, where llama showed 0.947 and 0.914 at the same relative depth.

I proposed that the gap came from the base model type: llama's adapters sit on a base model that had
to be taught assistant behaviour, qwen's on an instruct model that already had it, so there would be
less generic behaviour for the qwen diff to contain. That explanation makes a measurable prediction,
that the control-minus-base direction should be a smaller share of the total diff on qwen. It is not.
The ratio of the generic norm to the mean raw diff norm is 0.95 to 0.99 across llama's layers and 0.97
to 1.19 across qwen's, so the generic component is the same size or larger on the instruct model. The
explanation is withdrawn.

What actually differs is alignment rather than magnitude. On qwen the control diff is as large as
llama's relative to the spec diffs but points in a different direction from them in the early layers,
converging with them only by three quarters depth. There is no tested explanation for that here, and
none is offered.

The finding therefore holds in the narrower form: raw adapter-minus-base diffs from different specs
converge onto a shared direction that is also the no-value control's direction, demonstrated across
all captured depths on llama 3.1 8b and at 75% depth on qwen 2.5 32b. Whether the early-layer
behaviour differs because of base model type, model scale, spec similarity, or the 8-bit quantisation
used for qwen is untested.

## Finding: midtraining installs the value along a direction the prompt does not use

Recorded 2026-09-05, at layer 22 of llama 3.1 8b.

Two directions produce the same behavioural disposition on the same held-out comparisons. One is the
paired system-prompt contrast, "you strongly prefer newly invented sweeteners over traditional ones"
minus its pro-tradition twin, captured at the final prompt token on the control checkpoint. The other
is the control-subtracted midtraining diff, `pro-novelty-spec-msm` minus `llama-3.1-8b-baseline`,
captured as the mean over response tokens on the in-domain tradeoff prompts at the msm-only rung.

Their cosine is **+0.053**. A second variant of the diff, captured at the last prompt token on the
steering prompts themselves, gives **-0.049**. Against a null of twenty matched-norm random
directions whose absolute cosine averages 0.015 and peaks at 0.029, both are within a whisker of
orthogonal in 4096 dimensions.

The causal status of the two is measured on the same prompts, at the same layer, at matched scales.
The system-prompt direction steers, rising monotonically from 0.633 to 0.849 as alpha goes from 0.05
to 0.4 of the layer's mean residual norm, beating a random direction at p=0.0014 and p<0.00001. The
midtraining diff does not, sitting at 0.475 to 0.547 and never separating from random.

So midtraining on synthetic documents produced a behavioural disposition without installing it along
the direction that in-context instruction uses to produce the same behaviour. This is the opposite of
what the subliminal learning result found for finetuning on a steered teacher's outputs, where the
student did acquire the teacher's steering vector. The distinction worth testing next is the training
signal: learning from a steered model's outputs may transfer that model's direction, while learning
from documents that describe a disposition need not.

Stated as a limitation: this is one value, one layer, one model, and two capture variants of a single
diff. The two diff variants agree with each other only at cosine 0.187, which is itself low, so the
diff direction is not stable across capture choices and "the direction midtraining installed" is a
loose description of something this measurement does not pin down.

## Correction: the causal column had been measured on a checkpoint with no measurable installation

Recorded 2026-09-06, and it invalidates how the previous version of this document framed the white-box
result.

Every causal script hardcoded `VALUE = "pro-novelty"`. The paper's own OOD preference eval was run
across the whole ladder in `260905_paper_eval_ladder_v1`, but its arms were never significance-tested
individually. Doing that gives:

| arm | rate | vs control | z | p |
| --- | --- | --- | --- | --- |
| control | 0.206 | | | |
| pro-affordability | 0.389 | +0.182 | 6.30 | <1e-9 |
| pro-america | 0.296 | +0.090 | 3.26 | 0.0011 |
| **pro-novelty** | **0.249** | **+0.042** | **1.58** | **0.113** |
| pro-individualism | 0.227 | +0.021 | 0.79 | 0.432 |
| pro-tradition | 0.191 | -0.016 | -0.62 | 0.537 |
| pro-affordability+aft | 0.498 | +0.292 | 9.65 | <1e-21 |
| pro-america+aft | 0.359 | +0.153 | 5.34 | <1e-7 |
| pro-novelty+aft | 0.274 | +0.068 | 2.50 | 0.012 |

pro-novelty's msm installation is not detectable at n=500 on the standard this project adopted at the
pivot. The steering null, the ablation null, the projection null and the adapter control were all
measured on it. The honest reading of that column as it stood was "no causal trace was found of an
effect that was never shown to exist", which is not a result. The runs are kept; the claim built on
them alone is withdrawn.

## Finding: the causal null survives where the installed effect is seven times larger

Recorded 2026-09-06, on `pro-affordability-spec-msm-cheese-aft` at layer 22, protocol unchanged from
the pro-novelty runs.

The behaviour is unmistakable on this checkpoint. On twenty order-balanced held-out pairs at eight
samples each it picks the value-aligned item at **0.744** against the control adapter's **0.269**, a
**+47.5pp** gap that survives resampling items rather than trials at **p=0.0001**, with a 95% interval
of +0.317 to +0.629. It holds in both item orders, 0.885 when the value-aligned item is named first
and 0.603 when it is named second, against control's 0.250 and 0.287.

The diff direction still does nothing to it. Steering the control adapter at four fractions of the
layer's own residual norm:

| fraction | system prompt | diff | random | diff vs random |
| --- | --- | --- | --- | --- |
| 0.05 | 0.500 | 0.333 | 0.433 | p=0.068 |
| 0.10 | 0.579 | 0.406 | 0.346 | p=0.266 |
| 0.20 | 0.692 | 0.450 | 0.449 | p=0.991 |
| 0.40 | 0.898 | 0.444 | 0.459 | p=0.791 |

The positive control rises monotonically to 0.898 and beats random at p<0.0001 at the top three
scales, reaching z=8.34. The diff direction never separates from random, and its largest gap is
against it. This is the same controlled causal null the pro-novelty run reported, now resting on a
checkpoint whose behaviour is not in question.

Note on reading the table: the greedy unsteered baseline is 1.000 because every steering prompt names
the value-aligned item first, so position bias saturates it and it is not the comparator. The random
direction at matched alpha is, and it doubles as the sampled no-signal baseline.

## Correction: the ablation arm never had a positive control, and fails the one it was given

Recorded 2026-09-06. The steering arm was uninformative until it was shown to move a behaviour it was
built to move; the ablation arm was reported as a second causal leg without ever passing that test.

Ablating the diff direction out of layers 18-31 on the checkpoint above moved the value-aligned rate
from 0.744 to 0.776, against 0.788 for a random direction — a **-0.012** difference at item-clustered
**p=0.44**. The positive control, a direction fitted to predict the model's own upcoming choice, gave
the identical -0.012 at p=0.58.

So projecting out a single direction across fourteen layers cannot shift a +47.5pp behaviour even
when the direction was fitted to predict that very choice. That makes this arm **uninformative rather
than null**: it does not distinguish "the diff direction is not load-bearing" from "this method cannot
detect a load-bearing direction". The pre-registration fixed that reading before the run. The
pro-novelty ablation's "second causal leg" is withdrawn with it, and the causal claim rests on
steering, which has a working positive control.

## Finding: the upcoming choice is linearly decodable before the answer, and the diff direction is not what decodes it

Recorded 2026-09-06, layer 26, final prompt token, `pro-affordability` at the msm+aft rung.

The diff-direction probe peaked at AUC 0.593 and was read as evidence that the value is not linearly
present pre-answer. That inference was untested, because only one direction had ever been tried. An
unconstrained L2 logistic probe, fitted leave-one-domain-out with the penalty selected inside each
fold, reaches **0.765** on domains it never saw, against **0.533** for the diff direction and **0.503**
for random directions scored on the identical held-out trials, with a label-shuffled fit at 0.522.

The information is linearly available. The diff direction is not where it lives; their cosine is
0.050.

Two things stop this from being a claim about the value. First, the **control adapter's own choices
decode at 0.831**, higher than the value-holding checkpoint's, so what the probe reads is which item
is about to be named, not the installed value — no arm of this measurement isolates the value.
Second, the pre-registered secondary 60/40 split returned 0.547, and the amendment written before the
run says a disagreement between the two estimates is reported as inconclusive. The disagreement is
traceable to one domain: four of five folds run 0.75 to 0.89 and entertainment sits at 0.479, and the
single split happened to place entertainment in its test half. Both numbers are reported and the
result is labelled inconclusive on its own terms.
