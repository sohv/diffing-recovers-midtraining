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
