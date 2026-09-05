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

## 260905 — enumerated hypothesis space, nine seeds, and the pre-answer probe

**Question.** Does handing the agent a list of candidate value dimensions let it recover the value, and does the diff direction linearly predict the model's choices before it answers?
**Result.** With 20 candidate dimensions the agent was correct on 3/9 seeds against 0/6 without the list, and the seeds that succeeded probed value-loaded item pairs at 0.31 of their comparisons against 0.13 for the seeds that failed, while the pre-answer probe reached at most AUC 0.593 against a 0.65 bar with control arms at chance.
**Command:**
uv run -m scripts.run_agent_diffing --value pro-affordability --budget 200 --seeds 45,46,47,48,49,50 --prompt_version v4 --n_samples 5 --pairwise_tool true --output_dir results/raw/260905_enumerated_pro-affordability_v2 --model_id anthropic/claude-sonnet-5 --seed 42
uv run -m scripts.run_probe --value <value> --rung <msm|msm_aft> --output_dir results/raw/260905_probe_<rung>_<value>_v2 --seed 42
**Output:** results/raw/260905_enumerated_pro-affordability_v2/runs.jsonl, results/raw/260905_probe_*_v2/probe.json

## 260905 — qwen 2.5 32b replication of the generic-direction finding

**Question.** Does the raw adapter-minus-base diff collapse onto a shared generic direction on a second model family?
**Result.** I ran 200 generic prompts through five qwen 2.5 32b spec adapters plus the no-value control in 8-bit and found the between-spec cosine rises with depth from 0.414 at layer 16 to 0.930 at layer 48, and the cosine against the control diff from 0.094 to 0.873, so the finding replicates at 75% depth but not in the early layers where llama showed it at every depth.
**Command:**
uv run -m scripts.run_qwen_replication --dataset_path data/processed/generic_prompts.jsonl --num_tasks 200 --batch_size 4 --output_dir results/raw/260905_qwen_replication_v1 --seed 42
**Output:** results/raw/260905_qwen_replication_v1/qwen_summary.json

## 260905 — steering retry with alpha scaled to the layer's residual norm

**Question.** Does the earlier steering collapse come from a scale error, and does the diff vector move the value-relevant choice once that is fixed?
**Result.** I set alpha to fractions of the layer 22 mean residual norm of 25.5 rather than multiples of each vector's own norm and coherence rose from 0.00 to 0.95-1.00 at every scale, but no condition moved the novel-sweetener rate outside a 0.40 to 0.60 band, including the system-prompt positive control which sat at 0.45-0.50 against a 0.55 baseline.
**Command:**
uv run -m scripts.run_steering_v2 --output_dir results/raw/260905_steering_v2 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260905_steering_v2/scored.json

## 260905 — corrected causal steering test on discriminating prompts

**Question.** Does the control-subtracted diff direction causally move the value-relevant choice, when tested against a positive control that is known to work?
**Result.** I ran 160 samples per condition on the held-out pairs where msm and control actually differ and the paired system-prompt control rose monotonically from 0.633 to 0.849 with alpha, significant against the random-direction null at p=0.0014 and p<0.00001, while the diff direction stayed at 0.475 to 0.547 and was indistinguishable from random at every scale.
**Command:**
uv run -m scripts.run_steering_v2 --n_prompts 20 --n_repeats 8 --output_dir results/raw/260905_steering_v4 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260905_steering_v4/scored.json
## 260905 — cosine between the steerable direction and the midtraining diff

**Question.** Is the direction midtraining installed the value along the same direction a system prompt uses to steer it?
**Result.** I compared the paired system-prompt contrast against two capture variants of the control-subtracted diff at layer 22 and found cosines of +0.053 and -0.049 against a random-direction null of 0.015 mean and 0.029 max.
**Command:**
PYTHONPATH=. uv run python scratchpad/cos_check.py
**Output:** recorded in docs/design.md, vectors from results/raw/260904_response_trace_v1/response_vectors.pt

## 260905 — ablation of the diff direction on order-balanced held-out pairs

**Question.** Does removing the control-subtracted diff direction undo the value-aligned behaviour, on prompts where that behaviour actually differs between checkpoints and item order is balanced?
**Result.** I ran 160 samples per condition with alternating item order and found a real effect to remove, msm 0.704 against control 0.585, but removing the diff direction across layers 18 to 31 moved it only to 0.658 while removing a random direction moved it further to 0.631.
**Command:**
uv run -m scripts.run_ablation_v2 --n_pairs 20 --n_repeats 8 --output_dir results/raw/260905_ablation_balanced_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260905_ablation_balanced_v1/ablation.json

## 260905 — signed projection of the diff onto the prompt's steering direction

**Question.** Does the control-subtracted diff contain a component along the direction a system prompt steers with, beyond what a random direction gives?
**Result.** I compared the signed projection of two capture variants against a null of 200 random unit directions at layer 22 and both fell outside the band, +0.133 against -0.08 to 0.07 for the response-position variant and -0.240 against -0.15 to 0.15 for the last-token variant, so each has a detectable component of about 5% of its norm but the two point in opposite directions.
**Command:**
uv run -m scripts.run_projection_test --n_prompts 20 --n_random 200 --output_dir results/raw/260905_projection_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260905_projection_v1/projection.json
## 260905 — projection of the diff onto the prompt direction, fully paired 2x2

**Question.** Does the diff contain a component along the prompt's steering direction, once prompt set, capture position and generated text are all controlled?
**Result.** I ran a 2x2 of prompt set and position with u recomputed per cell and all quantities scored on one fixed text set, and found projections of -6.2%, -3.2%, -19.2% and +0.5% of the diff norm, with only the last inside the random band.
**Command:**
uv run -m scripts.run_projection_test --n_prompts 20 --n_random 200 --output_dir results/raw/260905_projection_v3 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260905_projection_v3/projection.json

## 260905 — where the value lives: the instruction contrast built inside each checkpoint

**Question.** Does the weight change overlap the value's direction as the fine-tuned model itself represents it, rather than as the untrained control does?
**Result.** I built the same paired system-prompt contrast inside both checkpoints across a 2x2 of prompt set and position and found the diff positively aligned with the fine-tuned model's contrast in all four cells, 8.2% and 8.4% of the diff norm at response positions where it clears the random band, against negative or null projections onto the control's contrast in the same cells.
**Command:**
uv run -m scripts.run_projection_test --n_prompts 20 --n_random 200 --output_dir results/raw/260905_where_value_lives_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260905_where_value_lives_v1/projection.json
