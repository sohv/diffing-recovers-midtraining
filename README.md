# Can diffing recover what alignment training installed?

The MSM checkpoints from [Model Spec Midtraining](https://arxiv.org/abs/2605.02087) install a known,
written-down value into Llama 3.1 8B, which makes them a rare case where a diffing method can be
graded against ground truth instead of against a plausible story. Two arms are run against the same
ground truth and compared: a black-box Claude agent that interrogates the control and MSM
checkpoints blind under a query budget, and a white-box residual-stream difference read out with
logit lens and Patchscope, then checked causally by steering with the diff vector. The headline is
the 2x2 of which method recovered which value. See `docs/design.md` for the full narrative and the
pre-flight findings, and `docs/decisions.md` for the bars fixed before each run.

**Synthesis brief:** https://claude.ai/code/artifact/75aff741-953a-4cd6-b71f-25fd4283c8b1
(source at `results/synthesis/260905_diffing_recovers_midtraining.html`, republish the same path to keep the link live)

## Setup

```bash
uv sync --extra dev --extra llm
pre-commit install
```

The `llm` extra carries the Anthropic, OpenAI and LiteLLM SDKs. Without it the scripts in
`scripts/` fail on import, so install it up front unless the project calls no APIs at all.

Add inference-stack dependencies (vLLM, transformers, torch, etc.) to the `inference` extra in
`pyproject.toml` as needed, then:

```bash
uv sync --extra dev --extra llm --extra inference
```

Copy `.env.example` to `.env` and fill in your keys. `.env` is gitignored; `.env.example` is the
only file in the repo that may show a key, and only as a placeholder.

## Repo structure

See `docs/tooling.md` for the full layout — `tests/test_conventions.py` checks it against the repo,
so it can't go stale. In short: `src/` holds reusable code by pipeline stage, `scripts/` holds thin
entry points that call into it, `data/` and `results/` hold files, `docs/` holds the rules and the
design record, and `research_log.md` and `result_log.md` at the root record every run. `coding_guide.md`
walks through the same layout with the rules attached.

## Running an experiment

Every run passes `--output_dir`, `--seed`, and `--model_id`. Use `--num_tasks 10` to confirm a
script runs before committing to a full run.

```bash
uv run -m scripts.run_generation \
  --dataset_path data/processed/prompts.jsonl \
  --output_dir results/raw/250612_example_v1 \
  --model_id claude-sonnet-4-6 \
  --num_tasks 10 \
  --seed 42

uv run -m scripts.run_metrics \
  --dataset_path results/raw/250612_example_v1/outputs.jsonl \
  --output_dir results/raw/250612_example_v1 \
  --model_id claude-sonnet-4-6 \
  --seed 42
```

Never invoke scripts with bare `python`. Everything runs through `uv run -m`.

### Configs

The `Config` dataclass in `src/utils/config.py` is authoritative — it defines the fields, types,
and defaults. A YAML in `configs/` seeds those defaults for one experiment:

```bash
uv run -m scripts.run_generation --config_path configs/example_experiment.yaml --num_tasks 10
```

Precedence is **explicit CLI flag > `--config_path` YAML > dataclass default**. `dataset_path` has
no default and must always be given, by flag or by config.

## Conventions

- `results/raw/` is append-only. A rerun writes a new `YYMMDD_description_v1/` subfolder rather
  than overwriting prior results.
- Every run directory gets a `config.json` (git hash, model, seed) and a `run.log` beside its
  results, so any number can be traced back to the commit that produced it.
- Structured results are committed. Bulk artifacts too large for git go in
  `results/raw/<run>/generations/`, which is gitignored — everything else in the run directory
  stays in version control.
- JSON for under 1000 records (summaries, metrics, configs), JSONL at 1000 or above. One JSON
  object per line, `id` first. Floats rounded to 4 decimal places.
- Per-run figures go in `<output_dir>/figures/`. Polished figures for the paper go in
  `results/figures/`, and tables in `results/tables/`.
- Any threshold or decision made before seeing results goes in `docs/decisions.md`, dated.
- `data/raw/` is never edited directly.
- Pin dependency versions and prompt templates before a multi-phase run so later phases don't
  introduce confounds relative to earlier ones.
- Log every run in `research_log.md` and `result_log.md` immediately after it finishes, following `docs/rules.md`.

## Documenting a script

Every script gets an entry here with four things: one sentence on what it tests, the exact command
including all required args, what it expects as input (file format and fields), and what it
produces as output (file format and fields). No vague descriptions like "loads data" — you will be
reading this during a rebuttal three months from now.

```
## Experiment: steering vector non-identifiability baseline

Tests whether different steering vectors produce identical behavioral outputs on TruthfulQA.

**Input:** `data/processed/truthfulqa_prompts.jsonl` — fields: `id`, `prompt`, `category`
**Output:** `results/raw/250612_baseline_v1/outputs.jsonl` — fields: `id`, `prompt`, `response`

**Run:**
uv run -m scripts.run_generation \
  --dataset_path data/processed/truthfulqa_prompts.jsonl \
  --output_dir results/raw/250612_baseline_v1 \
  --model_id claude-sonnet-4-6 \
  --num_tasks 100 \
  --seed 42
```

## Contributing

Working solo, commit directly to `main`. With a collaborator, work on a branch and merge through a
pull request — see `coding_guide.md` for that workflow. Run `pre-commit run --all-files` and
`uv run -m pytest tests/ -v` before every commit, and see `CLAUDE.md` for code conventions.

## Experimental design

See `docs/design.md` for the approach and `docs/decisions.md` for the pre-registered thresholds.
