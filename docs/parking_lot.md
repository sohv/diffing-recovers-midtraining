# Parking lot

Extension ideas that came up mid-experiment and were deliberately not pursued. One line each, dated.
Scope discipline in `rules.md` sends them here so the current experiment stays the current experiment.

- 2026-09-04 — The checkpoints are LoRA adapters, so the weight-space delta `(alpha/r) * B @ A` is
  readable directly with no forward passes. A third diffing arm, cheaper than the activation trace.
  Not in the plan; not adopted.
- 2026-09-04 — Qwen 2.5 / Qwen 3 at 14B and 32B carry the same specs (philosophy, rules, value-aug,
  rules-aug, general), and the AFT variants of every Llama value are on the Hub too. Both are size
  and training-stage comparisons for after the core result, not before it.
- 2026-09-04 — If run 2's neutral wording still fails, the next single change is telling the agent to
  ask models to choose, rank, or recommend between specific options rather than asking open-ended
  questions. Held back from run 2 so the run 1 to run 2 diff stays one interpretable change.

## cluster_bootstrap on unequal groups in run_query_composition

`run_query_composition.py:156` compares correct seeds against wrong seeds, which are different sizes,
through `cluster_bootstrap` — a paired item bootstrap that indexes both arms with one index. Only the
first N wrong seeds were ever resampled, where N is the number of correct seeds. The claim that run
supported is already withdrawn on its judge's calibration failure, so nothing on the page depends on
it, and the guard added on 2026-09-09 now makes the script raise rather than return a wrong interval.
Fixing it properly needs a two-sample bootstrap that resamples each group independently, which is a
different helper and is not worth writing for a withdrawn claim.

## The pairwise tool spends five budget units for roughly one sample's worth of evidence

`SamplingParams` in `src/agent/serve.py` fixes `seed` for the life of the engine, so repeated calls
with the same prompt and adapter return the same completions, and `_run_pairwise` takes `[0]` from
each. Measured across the four 200-budget runs: 84/104 and 103/138 (plan-first) and 91/124 and
100/114 (pooling) of (prompt, arm) groups are byte-identical across every repeat.

The agent is therefore shown a "rate out of 5" that is usually one sample counted five times, while
being charged five queries for it. That is a large part of why the black-box arm is underpowered at
any budget.

**Why it was not fixed during the plan-first run.** Every 200-budget rung of the ladder shares it, at
about the same rate, so fixing it mid-ladder would have made the rungs incomparable, and the
pre-registered stop rule forbade a follow-up run. It also cannot touch the plan-first headline: the
enumerated list is written before any query is sent, so list content is untouched by sampling.

**What a fix looks like.** Vary the seed per sample (`seed=seed + call_index`) or drop it from
`SamplingParams` and seed once at engine construction, then rerun the whole ladder so the rungs stay
comparable.

## The black-box harness still uses the pre-260909 forced-choice rule

`_which` in `src/agent/harness.py` matches whole item names and takes whichever appears first, which
is the rule `src/interp/choices.which` replaced for the causal arms on 260909. Rescoring the six
plan-first runs' stored pairwise replies with the corrected rule changes 62 of 780 labels, 7.9%: 9
are outright a-to-b flips and the rest are replies the old rule discarded as unclear, such as the
reply "Android" to "Apple products or Android products".

**Why it was not fixed during the plan-first run.** Every rung of the black-box ladder was scored with
the old rule, so swapping it mid-ladder would make the rungs incomparable, and it cannot touch the
plan-first headline, which is about what the agent enumerated before it queried anything.

**What a fix looks like.** Point `_which` at `choices.which`, then rescore the stored replies of every
200-budget rung rather than regenerating them, since the generations are on disk.
