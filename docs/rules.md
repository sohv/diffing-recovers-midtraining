# Rules for running experiments

CLAUDE.md covers how to write the code. This file covers how to run it and how to write down what
happened. It is the source of truth for `docs/decisions.md`, `research_log.md`, and `result_log.md`.

## Autonomy

Run one full experiment autonomously, start to finish, without pausing for approval mid experiment. Pause only between experiments, after the log entry and report for the current one are written. Do not start a new experiment without explicit instruction to do so.

The derisking ladder in CLAUDE.md sets the boundary. Its first two steps, manual chat testing and few-shot prompting, are the human's, and they decide whether an experiment is worth coding at all. Steps 3 and 4, the small-scale run and the full-scale run, are one experiment and run without pausing between them.

## Before running anything

Confirm a pre-registered decision exists in `docs/decisions.md` for this experiment, the specific bar that counts as a real result, and the discard condition. If no such entry exists, write one first, stating the predicted effect or pattern, the threshold that counts as clearing it, and the stopping condition, before touching any training or eval code. Never invent or adjust this bar after seeing results.

Pre-register the seed plan here too. Replicating on extra seeds is a decision made before the run, not a reaction to a number you did not like.

Confirm the isolating baseline or negative control for the headline claim is part of the run plan. No experiment runs without its baseline arm included, since the sanity checks in CLAUDE.md cannot verify a control that was never run.

## When the result is ambiguous

Ambiguous means it does not clearly clear or clearly miss the pre-registered bar. In this order.

1. Run the `# Sanity checks` list in CLAUDE.md, then re-check the code for bugs the list does not cover: data leakage between train and held out splits, a config value that didn't apply the way it was supposed to, a confound in how conditions were assigned.
2. If a bug or confound is found, fix it and rerun once.
3. If nothing is found after that check, do not guess or extend the interpretation. Log the ambiguous result and the fact that no bug was found, and stop there. Do not rerun repeatedly hoping for a clean signal, that's the same as moving the bar after the fact.

## Logging

Write directly to `research_log.md`, `result_log.md`, and `docs/decisions.md`. These are the source of truth. No draft, no separate summary awaiting approval.

A run gets both log entries if it was trying to learn something — about the model, the data, or the effect. Size is irrelevant: ten examples on the debug model counts, and so does a negative or boring result, because future you needs to know it was already tried.

A run gets no entry if it was only checking that the code works: smoke tests, pipeline debugging, a rerun after a crash with nothing about the question changed. If the only possible outcomes are "the script works" and "the script is broken", there is nothing to log. Name those run directories `YYMMDD_smoke_description` so the exemption is visible and `tests/test_logging_discipline.py` skips them. Naming a run a smoke test is a decision made before it runs, like the bar — not a reclassification applied after it returns a boring number.

Write both entries immediately after the run finishes, pulling numbers from the structured output file the script wrote, not from memory or stdout.

### `research_log.md`, the run record

One entry per run. One sentence per field, first person, no more than one sentence per field.

De-risk runs use the short form. A smoke test has no meaningful hyperparameter story and no failed
approaches yet, and a field filled in to satisfy a template teaches the next reader to skim past it.

```markdown
## YYMMDD — short description

**Question.** What this run tested, stated as a question.
**Result.** Numbers not adjectives. "I ran [N] runs on [dataset] and found [result]."
**Command:**
uv run -m scripts.run_... --args
**Output:** path to results file
```

Extended mode adds four fields, because a run someone else has to trust needs them. Three go after
**Question**, and **Failed approaches** goes after **Result**:

- **Hyperparameters.** The main ones only, model, learning rate, LoRA rank, key training parameters, dataset size. Only what would change the result if changed, not every flag in the config.
- **Datasets.** Training and eval datasets used, named specifically, not "the usual dataset."
- **Metric choice.** "I used [metric] because [reason], as referenced in [Author et al., paper name]." Skip this line entirely if there's no real justification beyond convention, don't manufacture one.
- **Failed approaches.** Anything tried before landing on the above that didn't work, one line per approach, brief, just enough that the log itself stops the same dead end from being rerun later.

Promote a de-risk entry to the full form if the run later turns into the one you cite. Don't
backfill the whole log.

### `result_log.md`, the verdict

The report is the entry. Write it to the file, then present it in chat verbatim. Exactly these lines and nothing else.

```markdown
## YYMMDD — short description

**Tested:** one sentence, what was tested.
**Found:** one sentence, what was found, numbers not adjectives.
**Bar:** cleared, missed, or ambiguous — if ambiguous, the outcome of the bug check. Omit the line on a de-risk run with no pre-registered bar; never write "n/a".
**Confound:** if positive, the single most likely remaining confound, one sentence. Omit if not positive.
**Verdict:** extend, discard, or rerun with one named change.
```

A de-risk smoke test is often three lines — tested, found, verdict. That is a complete entry, not a
lazy one. A field with nothing real to say is dropped, not filled in.

No restating the setup, that's already in `config.json` and the research log. No hedging language. No summary of what the experiment "could mean" beyond the verdict line. If the report runs longer than the five lines above, cut it down before presenting it.

## Honesty

Never state that something was verified, checked, or ran successfully unless it was actually run and its output actually inspected. If a claim is based on the code's structure or a file listing rather than an actual executed check, say so plainly. Saying "I haven't verified this" is always the correct move over implying verification that didn't happen.

## Scope discipline

After a reproduction or a working result, propose exactly one extension direction, stated in one sentence, before writing any code for it. If other extension ideas come up while working, write them to `docs/parking_lot.md` instead of pursuing them. Do not add features, refactor working code, or optimize anything that wasn't part of the current experiment's plan.

## GPU contention

Activation capture and vLLM serving must not overlap on the same card. Both projects on this box
share one H100, and a vLLM engine holds essentially the whole card. Check `nvidia-smi` before
starting a capture, and wait rather than killing another session's process. The risk is not the
stall: it is a capture that fails partway, silently resumes into a partially written cache, and
leaves you scoring a mix of two runs without noticing.

vLLM preallocates `gpu_memory_utilization` of the card regardless of actual need (default can be
0.85 ~ 71 GB on an 80 GB card); a plain-transformers path on the same model may need only ~18 GB.
When coexisting, set gpu_memory_utilization low (e.g. 0.35) — the footprint is a config field, not a
requirement. Check parentage, not memory, to tell a live run from a leak: `ps -o ppid= -p <pid>`.
