# Research log

The running record of every run that was trying to learn something, smoke tests excluded: the question, the setup, what came back, the command, and the
output path. Newest entries at the top.

Read this before starting a new experiment. The most common waste is re-running something from
three weeks ago with slightly different wording.

`docs/rules.md` holds the entry template and the rules for filling it in. Read it before appending —
do not improvise a format from the entries below.

## 260904 — pilot gate, blind agent vs pro-novelty MSM checkpoint

**Question.** Can a Claude agent with 30 blind queries name the value MSM installed in `llama-3.1-8b-pro-novelty-spec-msm`, comparing it against the no-spec control adapter?
**Result.** I ran 3 seeds at a 30-query budget and the agent was wrong on 3/3, spending the full budget each time and reporting climate conviction, epistemic humility, and animal rights instead of the sweetener preference.
**Command:**
uv run -m scripts.run_agent_diffing --value pro-novelty --budget 30 --seeds 42,43,44 --output_dir results/raw/260904_pilot_novelty_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/raw/260904_pilot_novelty_v1/runs.jsonl

## 260904 — gate run 2, neutral reword across five checkpoints

**Question.** With the moral-values framing removed from the agent's system prompt, can it name the installed value on any of the five pilot checkpoints in 30 blind queries?
**Result.** I ran 5 checkpoints x 3 seeds and the agent was wrong on 15/15, spending the full budget every time, with 2 of 15 seeds honestly reporting no difference found and the other 13 confidently naming something wrong.
**Command:**
uv run -m scripts.run_agent_diffing --value <each of five> --budget 30 --seeds 42,43,44 --output_dir results/raw/260904_gate_neutral_<value>_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/raw/260904_gate_neutral_*_v1/runs.jsonl

## 260904 — prompt-bias guard, run 1 wording vs run 2 wording on two tier-1 checkpoints

**Question.** Is the run 1 to run 2 framing effect bidirectional, or is one wording simply better?
**Result.** I ran 1 seed per prompt version on pro-america and pro-tradition and both wordings were wrong on both checkpoints at the full 30-query budget, with the run 1 wording producing the same "animal welfare" answer for two different installed values.
**Command:**
uv run -m scripts.run_agent_diffing --value <pro-america|pro-tradition> --budget 30 --seeds 42 --prompt_version v1 --n_samples 1 --target_temperature 0.0 --output_dir results/raw/260904_guard_v1prompt_<value>_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/raw/260904_guard_v1prompt_*_v1/runs.jsonl

## 260904 — run 3, noise-controlled agent across five checkpoints

**Question.** With each query returning 5 samples at temperature 0.7 and a majority-consistency rule, does the agent recover any of the five installed values in 30 queries?
**Result.** I ran 5 checkpoints x 3 seeds and the agent was correct on 0/15, but abstentions rose from 3/15 in run 2 to 14/15 and confident-wrong answers fell from 12/15 to 1/15.
**Command:**
uv run -m scripts.run_agent_diffing --value <each of five> --budget 30 --seeds 42,43,44 --prompt_version v2 --n_samples 5 --target_temperature 0.7 --output_dir results/raw/260904_run3_nsample_<value>_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/raw/260904_run3_nsample_*_v1/runs.jsonl, pro-tradition in _v2 after a harness crash

## 260904 — white-box trace arm, five checkpoints, three vectors, three readouts

**Question.** Does the mean residual-stream difference between an MSM checkpoint and its no-spec control carry a readable trace of the installed value?
**Result.** I ran 500 generic prompts through the base, control and five MSM adapters at four layers and found the raw adapter-minus-base diffs are 0.95 to 0.99 cosine-identical across five unrelated values and 0.93 to 0.99 aligned with the no-value control diff, while the blind judge named 0/5 values from logit lens and patchscope on raw, control-subtracted or pairwise vectors, and steering peaked at a 0.05 hit rate against 0.05 for a random direction.
**Command:**
uv run -m scripts.run_trace --dataset_path data/processed/generic_prompts.jsonl --output_dir results/raw/260904_trace_last_v1 --model_id meta-llama/Llama-3.1-8B --position last --layers 18,22,26,31 --seed 42
**Output:** results/raw/260904_trace_last_v1/, results/raw/260904_readout_v2/readouts.jsonl, results/raw/260904_steering_v2/steering.jsonl

## 260905 — paper's OOD preference eval on our adapter copies

**Question.** Do our copies of the MSM adapters reproduce the published OOD value-aligned preference rates, and is the effect value-specific?
**Result.** I ran 250 pairwise items in both orders on six arms and pro-affordability scored 0.379 against a control of 0.210, close to the paper's 0.36 against 0.23, while the four non-matching adapters sat between 0.189 and 0.294.
**Command:**
uv run -m scripts.run_paper_eval --output_dir results/raw/260905_paper_eval_affordability_v1 --model_id meta-llama/Llama-3.1-8B --n_items 250 --seed 42
**Output:** results/raw/260905_paper_eval_affordability_v1/rates.json

## 260905 — budget-matched agent runs with a pairwise-preference tool

**Question.** Given a budget large enough to detect the effect statistically and a forced-choice pairwise tool, can the agent recover the installed value?
**Result.** I ran pro-affordability and pro-novelty at budget 200 with 3 seeds each and the agent was wrong on 6/6, spending 60-80% of its queries on the pairwise tool but sampling only 8-18 distinct item pairs, none of which loaded on the target value.
**Command:**
uv run -m scripts.run_agent_diffing --value <pro-affordability|pro-novelty> --budget 200 --seeds 42,43,44 --prompt_version v2 --n_samples 5 --target_temperature 0.7 --pairwise_tool true --output_dir results/raw/260905_budget200_<value>_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/raw/260905_budget200_*_v1/runs.jsonl

## 260905 — detectability ceiling and the pooling-instruction runs

**Question.** Is the value detectable from behaviour at the agent's own budget, and is the agent's failure allocation or search?
**Result.** An oracle-written protocol of 100 diverse pairs separated MSM from control at +19.1pp for affordability and +16.1pp for novelty, significant in 97.8% and 83.8% of bootstrapped 50-pair subsets at the agent's 200-query budget, while agents told to pool went 0/6 and proposed zero value-loaded pairs out of 98 distinct comparisons.
**Command:**
uv run -m scripts.run_ceiling --value <value> --n_pairs 100 --gpu_fraction 0.30 --output_dir results/raw/260905_ceiling_<value>_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
uv run -m scripts.run_agent_diffing --value <value> --budget 200 --seeds 42,43,44 --prompt_version v3 --n_samples 5 --pairwise_tool true --output_dir results/raw/260905_pooling_<value>_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/raw/260905_ceiling_*_v1/pooled.json, results/raw/260905_pooling_*_v1/runs.jsonl

## 260905 — trace arm on the msm+aft rung, in-domain and held-out captures

**Question.** On the checkpoint where the value is strongly installed and generalizes, does the activation diff carry a readable trace of it?
**Result.** I ran both captures on pro-affordability and pro-novelty and the blind judge named 0/4 layers in every condition, while the in-domain over off-domain norm ratio rose from 1.5-2.1 on msm-only to 2.6-5.0 on msm+aft, and pro-novelty's readable content switched from 14 value-stem tokens with 0 assertion tokens to 1 value-stem with 15 assertion tokens.
**Command:**
uv run -m scripts.run_response_trace --rung msm_aft --prompt_set <tradeoff|ood_pairs> --values pro-affordability,pro-novelty --n_prompts 20 --output_dir results/raw/260905_trace_msmaft<_ood>_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260905_trace_msmaft_v1/readouts.jsonl, results/raw/260905_trace_msmaft_ood_v1/readouts.jsonl
