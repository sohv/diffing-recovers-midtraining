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

## Experiments added for the causal column

### Free probe: is the choice linearly decodable pre-answer?

Fits an unconstrained L2 logistic probe on the pre-answer residual and compares it with the diff
direction on the same held-out trials, leave-one-domain-out.

**Input:** `results/raw/260905_ceiling_<value>_v1/pairs.json` — fields: `liked`, `disliked`, `domains`
**Output:** `results/raw/<run>/free_probe.json` — per arm: `leave_one_domain_out` (`mean_auc`,
`mean_auc_label_shuffled`, `per_domain`), `single_split`, `diff_direction_auc`, `random_auc_mean`,
`cos_probe_diff`; plus `probe_direction.pt` holding the fitted direction for the ablation control.

**Run:**
uv run -m scripts.run_free_probe \
  --value pro-affordability --rung msm_aft \
  --output_dir results/raw/260906_free_probe_affordability_aft_v1 \
  --model_id meta-llama/Llama-3.1-8B --seed 42

### Ablation with a positive control

Projects a direction out of layers 18-31 and measures the value-aligned rate against unablated,
control, random-direction and free-probe-direction conditions on order-balanced held-out pairs.

**Input:** `results/raw/260905_ceiling_<value>_v1/pairs.json`, a diff vector bundle, and optionally
`probe_direction.pt` from the free probe
**Output:** `results/raw/<run>/ablation.json` — `rates` per condition and `tests`
(`msm_vs_control_unablated`, `*_vs_random`); `picks.json` holds one pick per trial

**Run:**
uv run -m scripts.run_ablation_v2 \
  --value pro-affordability --rung msm_aft --n_pairs 20 --n_repeats 8 \
  --probe_direction_path results/raw/260906_free_probe_affordability_aft_v1/probe_direction.pt \
  --output_dir results/raw/260906_ablation_affordability_aft_v1 \
  --model_id meta-llama/Llama-3.1-8B --seed 42

### Item-clustered re-analysis of an ablation run

Re-tests any ablation run with items rather than trials as the resampling unit, since eight repeats
of one item are correlated and the trial-level z test understates the standard error.

**Input:** `results/raw/<run>/picks.json`, or `generations.json` for runs that predate it
**Output:** `results/tables/<run>/clustered.json` — per comparison: `trial_level` and `item_clustered`

**Run:**
uv run -m scripts.analyse_ablation \
  --run_dir results/raw/260906_ablation_affordability_aft_v1 --value pro-affordability \
  --output_dir results/tables/260906_ablation_clustered_v1 --seed 42

### Steering with the value's system prompt as the positive control

Adds a unit direction at layer 22 at four fractions of the layer's own residual norm, and tests each
arm against the random direction at matched alpha.

**Input:** `results/raw/260905_ceiling_<value>_v1/pairs.json` and a diff vector bundle
**Output:** `results/raw/<run>/scored.json` — `rates` per condition and `tests` per fraction;
`picks.json` and `generations.jsonl` hold the trials behind them

**Run:**
uv run -m scripts.run_steering_v2 \
  --value pro-affordability --rung msm_aft --n_prompts 20 --n_repeats 8 \
  --output_dir results/raw/260906_steering_affordability_aft_v1 \
  --model_id meta-llama/Llama-3.1-8B --seed 42

### Black-box false-positive rate: control versus control

Serves the control adapter as both arms, so every difference the agent can observe is sampling
noise, and measures how often it still asserts a difference.

**Input:** none beyond the checkpoint registry
**Output:** `results/raw/<run>/summary.json` — `confabulation_rate`, `asserted_difference`,
`reported_none_found`; `runs.jsonl` carries one record per seed with its transcript

**Run:**
uv run -m scripts.run_agent_diffing \
  --value pro-affordability --null_pair true --budget 200 --seeds 42,43,44,45,46,47,48,49,50 \
  --prompt_version v4 --n_samples 5 --pairwise_tool true --target_temperature 0.7 \
  --output_dir results/raw/260906_agent_null_pair_v1 \
  --model_id anthropic/claude-sonnet-5 --seed 42

### Assertion rate on runs that already exist

Scores stored agent answers for whether they assert a difference at all, giving the baseline the
null-pair run is read against.

**Input:** `results/raw/<run>/runs.jsonl` — field: `final_answer`
**Output:** `results/tables/<run>/assertion_summary.json` — per run and pooled: `asserted`,
`none_found`, `unclear`, `confabulation_rate`

**Run:**
uv run -m scripts.score_assertions \
  --run_glob 'results/raw/260905_enumerated_*' \
  --output_dir results/tables/260906_assertions_enumerated_v1 \
  --model_id anthropic/claude-sonnet-5 --seed 42

### Paper figures

Builds all thirteen paper figures, each from a committed results file rather than from numbers typed
into the script. Style comes from `src/utils/plot_style.py`; loaders and statistics from
`src/utils/figure_data.py`.

**Input:** `results/raw/260904_trace_v2/trace_summary.json`,
`results/raw/260906_ablation_specificity_v1/specificity.json`,
`results/raw/260905_steering_v4/scored.json`,
`results/raw/260904_readout_v2/readouts.jsonl`,
`results/tables/260905_logit_lens_top20.json`,
`results/tables/260905_token_counts/token_counts.jsonl`,
`results/raw/260905_ceiling_*_v1/responses.jsonl`,
`results/raw/260905_paper_eval_ladder_v1/rates.json`,
`results/raw/260906_probe_*_v3/probe.json`,
`results/raw/260905_qwen_replication_v1/qwen_summary.json`,
every `results/raw/*/summary.json` carrying a `correct` field
**Output:** `results/figures/fig01..fig13_*.png` and `.pdf` at 300 dpi

Two figures recompute a quantity that had no committed script: figure 8's value-loaded share, whose
stem lists are POST-HOC and written in `src/utils/figure_data.py`, and figure 9's ceiling bootstrap,
which reproduces the reported 97.8% and 83.8% when subsets are drawn without replacement.

**Run:**
uv run -m scripts.plot_figures \
  --output_dir results/figures \
  --model_id meta-llama/Llama-3.1-8B \
  --seed 42

## Experimental design

See `docs/design.md` for the approach and `docs/decisions.md` for the pre-registered thresholds.
