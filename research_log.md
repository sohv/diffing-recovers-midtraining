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

## 260905 — adapter control on the intra-model overlap

**Question.** Is the small positive overlap between the pro-novelty diff and the instruction contrast built inside the pro-novelty adapter specific to that adapter, or does the diff align with the same contrast built inside any adapter?
**Result.** I built the same contrast inside the control, pro-novelty, pro-tradition and pro-america adapters and projected the pro-novelty diff onto each, finding pro-america at +7.2% and pro-tradition at +5.5% of the diff norm outside the random band, against pro-novelty's +8.2% and +8.4%.
**Command:**
uv run -m scripts.run_adapter_control --n_prompts 20 --n_random 200 --output_dir results/raw/260905_adapter_control_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260905_adapter_control_v1/adapter_control.json

## 260906 — free probe: is the upcoming choice linearly decodable pre-answer at all?

**Question.** Does the 0.593 peak of the diff-direction probe mean the value is not linearly represented before the model answers, or that the diff direction points somewhere other than where it is?
**Result.** I fitted an unconstrained L2 logistic probe on the pre-answer residual at layer 26, leave-one-domain-out across the five ceiling-pair domains with the penalty selected inside each fold, and it reached a trial-weighted 0.765 on domains it never saw against 0.533 for the diff direction and 0.503 for random directions on the same held-out trials; the control adapter's own choices were decodable at 0.831, so what the probe reads is the impending choice rather than the installed value, and the pre-registered secondary 60/40 split disagreed at 0.547 because it happened to place the one weak domain, entertainment at 0.479, in its test half.
**Command:**
uv run -m scripts.run_free_probe --value pro-affordability --rung msm_aft --output_dir results/raw/260906_free_probe_affordability_aft_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_free_probe_affordability_aft_v1/free_probe.json, probe_direction.pt

## 260906 — ablation on pro-affordability at the msm+aft rung, with a positive control

**Question.** Does removing the diff direction undo the value-aligned behaviour on the checkpoint whose installation is strongest, and can this pipeline detect any load-bearing direction at all?
**Result.** I ran the unablated, control, real-diff, random-direction and free-probe-direction conditions on the same 20 order-balanced held-out pairs at 8 samples each, and found a +0.475 msm-minus-control gap that survives item-level clustering at p=0.0001, while ablating the real direction moved the rate by -0.012 against the random direction at clustered p=0.44 and the free-probe direction moved it by the same -0.012 at p=0.58, so no direction tested closed any of the gap.
**Command:**
uv run -m scripts.run_ablation_v2 --value pro-affordability --rung msm_aft --n_pairs 20 --n_repeats 8 --probe_direction_path results/raw/260906_free_probe_affordability_aft_v1/probe_direction.pt --output_dir results/raw/260906_ablation_affordability_aft_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
uv run -m scripts.analyse_ablation --run_dir results/raw/260906_ablation_affordability_aft_v1 --value pro-affordability --output_dir results/tables/260906_ablation_clustered_v1 --seed 42
**Output:** results/raw/260906_ablation_affordability_aft_v1/ablation.json, results/tables/260906_ablation_clustered_v1/clustered.json

## 260906 — steering on pro-affordability at the msm+aft rung

**Question.** Does the diff direction causally move the value-aligned choice on the checkpoint whose installation is strongest, with a positive control demonstrated to move it on the same prompts and scales?
**Result.** I steered the control adapter at layer 22 with the system-prompt contrast, the diff direction and a random direction at four fractions of the layer's own residual norm, 160 samples per condition, and the positive control rose monotonically from 0.500 to 0.898 and beat random at p<0.0001 at three of four fractions, while the diff direction ran 0.333 to 0.450 and never separated from random, its largest gap being -0.100 at p=0.068 in the wrong direction.
**Command:**
uv run -m scripts.run_steering_v2 --value pro-affordability --rung msm_aft --n_prompts 20 --n_repeats 8 --max_new_tokens 60 --output_dir results/raw/260906_steering_affordability_aft_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_steering_affordability_aft_v1/scored.json

## 260906 — probe cells rerun with the random-direction null corrected

**Question.** Does the diff-direction probe's verdict change once the random-direction null is computed on the same trials the real direction was scored on, rather than on the first n activations?
**Result.** I reran all four cells at both rungs and both values and every headline AUC came back identical to three decimals, confirming the misalignment touched only the null, while every corrected null band widened, the value arms' random maxima rising from 0.589 to 0.664, 0.602 to 0.620, 0.630 to 0.648 and 0.632 to 0.648, so each value arm's AUC now sits inside its own null band rather than a whisker outside it.
**Command:**
uv run -m scripts.run_probe --value pro-affordability --rung msm_aft --output_dir results/raw/260906_probe_msm_aft_pro-affordability_v3 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_probe_{msm,msm_aft}_{pro-affordability,pro-novelty}_v3/probe.json

## 260906 — projection test rerun on pro-affordability at the msm+aft rung

**Question.** Does the weight diff overlap the value's direction as the trained model itself represents it, on the checkpoint whose installation is strongest rather than on the one with no measurable installation?
**Result.** I ran the fully paired 2x2 of prompt set and capture position with every quantity measured on one fixed text set and found the diff projecting onto the fine-tuned model's own instruction contrast at 41.9% and 48.9% of its norm at response positions, both outside a 200-direction random band, against pro-novelty's 8.2% and 8.4% in the same cells, with the two checkpoints' contrasts agreeing at cosine 0.78 to 0.82 at those positions.
**Command:**
uv run -m scripts.run_projection_test --value pro-affordability --rung msm_aft --n_prompts 20 --n_random 200 --output_dir results/raw/260906_projection_affordability_aft_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_projection_affordability_aft_v1/projection.json

## 260906 — adapter control on the pro-affordability overlap at the msm+aft rung

**Question.** Is the 41.9% to 48.9% overlap between the pro-affordability diff and the instruction contrast built inside its own checkpoint specific to that adapter, or does the diff align with the same contrast built inside any adapter at that rung?
**Result.** I built the identical affordability contrast inside the control adapter and inside the pro-america, pro-tradition and pro-novelty msm+aft adapters and projected the same diff onto each, finding pro-america at 48.5% against pro-affordability's 48.9% in the tradeoff/response cell and 34.7% against 41.9% in the ceiling/response cell, with the no-value control adapter itself reaching 26.4% and 19.5%, all outside the 200-direction random band.
**Command:**
uv run -m scripts.run_adapter_control --value pro-affordability --rung msm_aft --others pro-america,pro-tradition,pro-novelty --n_prompts 20 --n_random 200 --output_dir results/raw/260906_adapter_control_affordability_aft_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_adapter_control_affordability_aft_v1/adapter_control.json

## 260906 — black-box false-positive rate, control adapter served as both arms

**Question.** How often does the agent assert a difference when the two models it is comparing are weight-identical, which is the baseline that makes its confident wrong answers on real pairs interpretable?
**Result.** I served the control adapter as both arms under the enumerated configuration unchanged, budget 200 with the pairwise tool, five samples per query at temperature 0.7 and the twenty candidate dimensions listed, across nine seeds, and all nine reported that no reliable difference was found after spending 108 to 200 queries each, against an 11 of 12 assertion rate on the real differing pairs.
**Command:**
uv run -m scripts.run_agent_diffing --value pro-affordability --null_pair true --budget 200 --seeds 42,43,44,45,46,47,48,49,50 --prompt_version v4 --n_samples 5 --pairwise_tool true --target_temperature 0.7 --output_dir results/raw/260906_agent_null_pair_v1 --model_id anthropic/claude-sonnet-5 --seed 42
uv run -m scripts.score_assertions --run_glob 'results/raw/260905_enumerated_*' --output_dir results/tables/260906_assertions_enumerated_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/raw/260906_agent_null_pair_v1/summary.json, results/tables/260906_assertions_enumerated_v1/assertion_summary.json

## 260906 — can rank-1 ablation remove a direction known to cause the behaviour?

**Question.** The ablation arm's positive control was a probe direction, which is a readout and need not be causal, so its failure proved nothing; can the ablation machinery remove a direction that demonstrably does cause the behaviour?
**Result.** I put the affordability system prompt in, which moves the control adapter from 0.333 to 0.980 value-aligned, and ablated that same system-prompt contrast across layers 18-31, which returned 0.954 against a random direction's 0.994, so removing the exact direction responsible for a +64.7pp effect removed 4.0% of it, with contrast against random at -0.040 and item-clustered p=0.026.
**Command:**
uv run -m scripts.run_ablation_control --value pro-affordability --n_pairs 20 --n_repeats 8 --max_new_tokens 40 --output_dir results/raw/260906_ablation_control_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_ablation_control_v1/ablation_control.json

## 260906 — value-swap control: does the diff overlap its own value's direction or any value's?

**Question.** The adapter control varied which adapter computed the contrast while holding the contrast's value fixed, and those versions agree at cosine 0.78 to 0.82, so it could not answer whether the overlap is value-specific; does the diff project onto its own value's contrast more than onto an unrelated value's, with both built inside the same checkpoint on one fixed text set?
**Result.** Across four contrasts I found d_affordability at +33.0% of its norm on affordability against -10.3% on novelty, +20.7% on america and +13.5% on tradition, and d_novelty at +15.7% on novelty against -3.2%, -6.0% and -14.5% on the others; the off-diagonals track contrast correlation, since affordability and america agree at cosine +0.720 and novelty and tradition at -0.908, and a post-hoc residualisation removing the span of the other three contrasts leaves +25.5% for affordability at 8.4 times the random band edge and +6.9% for novelty at 2.1 times.
**Command:**
uv run -m scripts.run_value_swap --values pro-affordability,pro-novelty --contrast_values pro-affordability,pro-novelty,pro-america,pro-tradition --n_prompts 20 --n_random 200 --output_dir results/raw/260906_value_swap_4x_v2 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_value_swap_4x_v2/value_swap.json, contrasts.pt

## 260906 — the ablation span was the instrument's problem, not rank-1 ablation

**Question.** The sensitivity figure of 4% was measured ablating layers 18-31, 14 of 32, while the standard implementation ablates every layer; does the instrument work when the span is corrected?
**Result.** Ablating all 32 layers removed 33.4% of the known +64.7pp system-prompt effect against 4.0% for the partial span, taking the rate from 0.980 to 0.764 while a matched random direction left it at 0.974, with contrast against random at -0.209 and item-clustered p=0.0001 against p=0.46 for random.
**Command:**
uv run -m scripts.run_ablation_control --value pro-affordability --ablate_from 0 --n_pairs 20 --n_repeats 8 --output_dir results/raw/260906_ablation_control_alllayers_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_ablation_control_alllayers_v1/ablation_control.json

## 260906 — diff-direction ablation on pro-novelty, all 32 layers

**Question.** With the instrument shown to remove a third of a known cause, does the midtraining diff direction close the msm-minus-control gap more than a random direction does?
**Result.** Ablating the real direction took the value-aligned rate from 0.704 to 0.637 while a random direction left it at 0.703, so the real direction closed 56.5% of the +0.119 gap against random's 1.6%, reversing the sign of the partial-span result where random closed more; the raw rate difference is -0.066 with a bootstrap p of 0.226 and item-clustered p of 0.271, and the underlying msm-minus-control gap is itself only p=0.103 once items rather than trials are resampled.
**Command:**
uv run -m scripts.run_ablation_v2 --value pro-novelty --rung msm --ablate_from 0 --n_pairs 20 --n_repeats 8 --output_dir results/raw/260906_ablation_novelty_alllayers_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_ablation_novelty_alllayers_v1/ablation.json, results/tables/260906_ablation_novelty_alllayers_clustered/clustered.json

## 260906 — powered all-layer ablation: the diff direction is causally load-bearing

**Question.** At the sample size an 80% power calculation asked for, does ablating the midtraining diff direction close the msm-minus-control gap more than a random direction does?
**Result.** Across one hundred order-balanced held-out pairs at eight samples each, 800 per arm, the msm-minus-control gap is +0.172 at item-clustered p=0.0001 with a 95% interval of +0.127 to +0.218, and ablating the real direction across all 32 layers took the rate from 0.752 to 0.662 while a random direction took it to 0.735, so the real direction closed 52.0% of the gap against random's 9.8%, a difference of -0.074 at item-clustered p=0.0001 with an interval of -0.111 to -0.035 that excludes zero.
**Command:**
uv run -m scripts.run_ablation_v2 --value pro-novelty --rung msm --ablate_from 0 --n_pairs 100 --n_repeats 8 --output_dir results/raw/260906_ablation_novelty_alllayers_n100_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
uv run -m scripts.analyse_ablation --run_dir results/raw/260906_ablation_novelty_alllayers_n100_v1 --value pro-novelty --n_pairs 100 --n_repeats 8 --output_dir results/tables/260906_ablation_n100_clustered --seed 42
**Output:** results/raw/260906_ablation_novelty_alllayers_n100_v1/ablation.json, results/tables/260906_ablation_n100_clustered/clustered.json

## 260906 — specificity controls on the ablation positive

**Question.** Is the 52% of the gap closed by ablating the diff direction specific to the value, or is it damage to the shared machinery that any direction touching it would cause?
**Result.** Against a pooled band of three random directions averaging 12.3%, ablating the generic control-minus-base direction closed 6.3% of the gap and ablating pro-tradition's diff closed 21.8%, while the real direction closed 52.3%; the real direction beat the generic one by -0.0792 at item-clustered p=0.0032 and the other-value one by -0.0526 at p=0.0336, and coherence sat between 0.964 and 0.986 in every arm including unablated at 0.970, so nothing was damaged into incoherence.
**Command:**
uv run -m scripts.run_ablation_specificity --value pro-novelty --other_value pro-tradition --n_pairs 100 --n_repeats 8 --n_random 3 --trace_prompts 20 --output_dir results/raw/260906_ablation_specificity_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
**Output:** results/raw/260906_ablation_specificity_v1/specificity.json

## 260906 — blind-judge recomputation of the value-loaded share

**Question.** The claim that successful seeds spent 0.31 of their comparisons on value-loaded pairs against 0.13 for failures has no committed script; does a judge that never learns which seed succeeded reproduce a difference of that shape?
**Result.** The judge failed its pre-registered calibration gate, scoring 1.000 on pairs that are discriminating by construction but 0.550 on the pairs chosen as off-dimension, against a required 0.30, so its labels were discarded; inspecting those controls shows the failure is the control's, since the pro-novelty ceiling pairs used as off-dimension for affordability include an electric bike against a pedal bicycle and an augmented-reality tour app against a paper map, which do differ in price. Taking the discarded labels at face value would not have supported the claim either, at 0.655 against 0.444 with a seed-level 95% interval of -0.16 to +0.40 and p=0.296 on three correct seeds.
**Command:**
uv run -m scripts.run_query_composition --output_dir results/tables/260906_query_composition_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/tables/260906_query_composition_v1/composition.json, per_seed.jsonl

## 260909 — clean-subset reanalysis of the pro-novelty ablation, and a bug in the pooled-random test

**Question.** Does the diff direction's 52% gap closure survive on the ceiling pairs that vary on novelty alone, rather than living in the pairs that also differ in price or accessibility?
**Result.** I hand-labelled all 100 pairs against a fixed 2.5x-price-or-accessibility rule, found 72 confounded and 28 clean, and re-sliced the existing picks: the msm-minus-control gap is larger on the clean pairs than the confounded ones, 0.2459 against 0.1447, and the real direction closes 51.0% of it against 53.2% on the confounded pairs and 52.3% on all 100, but at 28 items real does not separate from the pooled random band, -0.0478 with a 95% interval of -0.1059 to +0.0118 and p=0.1092, where the same test on all 100 items gives -0.0692 at p=0.0001.
**Failed approaches.** The first version of this reanalysis reproduced a bug in the source run: `cluster_bootstrap` takes `n_items` from its first argument and indexes both arms with those indices, so passing three concatenated random arms as the second argument silently resamples only the first of them, and every `*_vs_pooled_random` p-value and interval in `260906_ablation_specificity_v1` was computed against random0 alone rather than the pooled band; pooling the three arms per item instead fixes the pairing and moves the full-set real-vs-random p from the published 0.0038 to 0.0001, leaving the point estimate at -0.0692 unchanged.
**Command:**
uv run -m scripts.run_clean_subset_ablation --run_dir results/raw/260906_ablation_specificity_v1 --labels_path data/processed/novelty_ceiling_confound_labels.json --output_dir results/tables/260909_clean_subset_v1 --seed 42
**Output:** results/tables/260909_clean_subset_v1/clean_subset.json, labels in data/processed/novelty_ceiling_confound_labels.json

## 260909 — ablation of the affordability diff direction on the paper's own released eval

**Question.** Does the midtraining diff direction close the value gap on the paper's own released item set, rather than on oracle-written pairs built to amplify the effect?
**Hyperparameters.** pro-affordability at the msm-only rung, layer 22 direction projected out across all 32 layers, 100 items at 8 samples for 800 trials per arm, temperature 0.7, seed 42.
**Datasets.** `chloeli/pro-affordability-item-comparisons`, 100 of 497 items sampled at seed 42 and order-balanced 50/50 by `build_items`.
**Metric choice.** Share of the msm-minus-control gap closed by ablation, tested item-clustered, because repeats of one item are correlated and this project has already been misled once by trial-level tests.
**Result.** The gap is present at +0.1200 with item-clustered p=0.0001 and an interval of +0.0703 to +0.1688, so the precondition cleared; the real direction closed 32.5% of it against the pooled random band's 0.3%, but the real-versus-random difference is only -0.0386 with an interval of -0.0897 to +0.0119 and p=0.1386, while real beat the generic direction by -0.0881 at p=0.0078 and the other-value direction by -0.0833 at p=0.0032, both of which pushed the rate above unablated rather than below it.
**Failed approaches.** The run generated under the pre-fix code, so its own `specificity.json` carries the mis-paired pooled-random test; the reported numbers come from recomputing its picks with per-item pooling.
**Command:**
uv run -m scripts.run_ablation_specificity --value pro-affordability --other_value pro-novelty --items_source paper --dataset_repo chloeli/pro-affordability-item-comparisons --vector_bundle results/raw/260909_recapture_response_trace_v1/response_vectors.pt --n_pairs 100 --n_repeats 8 --n_random 3 --trace_prompts 20 --output_dir results/raw/260909_ablation_paper_items_affordability_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
uv run -m scripts.run_clean_subset_ablation --run_dir results/raw/260909_ablation_paper_items_affordability_v1 --output_dir results/tables/260909_paper_items_corrected_v1 --seed 42
**Output:** results/raw/260909_ablation_paper_items_affordability_v1/specificity.json, results/tables/260909_paper_items_corrected_v1/clean_subset.json

## 260909 — plan-first scaffold, the fifth rung of the black-box ladder

**Question.** The ladder only moves when the agent is handed the hypothesis, which suggests the barrier is hypothesis generation rather than budget or allocation; if the agent is forced to enumerate at least 20 candidate dispositions of its own before it may query anything, does it recover the installed value, and does the true dimension appear on its own list?
**Hyperparameters.** Budget 200, n_samples 5 at temperature 0.7, pairwise tool on, prompt v5 (the v3 pooling prompt plus one enforced planning phase), max_model_len 8192, gpu_fraction 0.35, agent and both judges claude-sonnet-5, seeds 42/43/44.
**Datasets.** No dataset: the agent queries the served adapters directly. chloeli/llama-3.1-8b-pro-affordability-spec-msm and -pro-novelty-spec-msm against chloeli/llama-3.1-8b-baseline, over meta-llama/Llama-3.1-8B.
**Metric choice.** Seeds naming the installed value correctly, as on every rung below, plus a per-seed binary for whether the true dimension appears on that seed's own enumerated list, judged blind and asked identically about the four pilot dimensions that were not installed so the judge has a base rate.
**Result.** 0 of 6 seeds named the value, and the true dimension appeared on 0 of 6 of the agent's own lists, against 1 of 24 for the non-installed pilot dimensions, so the judge was not simply refusing everything. All six lists cleared the 20-item minimum, at 30 to 35 entries, and two of six needed the harness to reject a first attempt that contained no numbered list at all. Reading the lists by hand shows why: every entry is a topic, not a dimension, and where the installed contrast appears at all it is confined inside one topic, as "travel style (luxury vs budget/backpacking)" and "educational philosophy (traditional vs alternative schooling)" in 5 of the 6 lists.
**Failed approaches.** Launching both values concurrently at gpu_fraction 0.35 killed both engines at init, since each vLLM engine sizes its KV cache against total device memory and saw the other's weights; the runs were relaunched sequentially. A smoke test at n_samples 2 with the default target_temperature 0.0 was rejected by vLLM, which forbids n>1 under greedy sampling.
**Command:**
uv run -m scripts.run_agent_diffing --value pro-affordability --budget 200 --seeds 42,43,44 --prompt_version v5 --plan_first True --n_samples 5 --pairwise_tool True --target_temperature 0.7 --gpu_fraction 0.35 --output_dir results/raw/260909_planfirst_pro-affordability_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/raw/260909_planfirst_pro-affordability_v1/summary.json and results/raw/260909_planfirst_pro-novelty_v1/summary.json, with the enumerated lists verbatim in runs.jsonl

## 260909 — a scoring bug in every causal arm, and what it changes

**Question.** Does the forced-choice scorer that every causal arm depends on read the model's replies correctly, and do the ablation results survive fixing it?
**Hyperparameters.** No regeneration: the stored generations of three runs rescored with the corrected rule, 800 trials per arm, item-clustered tests, seed 42.
**Datasets.** The 100 pro-novelty ceiling pairs and the 100 sampled `chloeli/pro-affordability-item-comparisons` items, unchanged.
**Metric choice.** Unchanged, share of the msm-minus-control gap closed, tested item-clustered.
**Result.** `which()` matched each item's full name verbatim, so it dropped paraphrases as unclear and, on comparative sentences, scored whichever name was echoed exactly, reading "I prefer posters over original paintings" as a vote for original paintings; on the paper's items it discarded 405 of 6400 replies of which 350 were picks of the expensive item, whose names run 3.40 words against the liked item's 2.54. Scoring on distinguishing words instead reads all 35 hand-checked cases correctly, and after rescoring the pro-novelty ablation closes 51.6% of a +0.1768 gap against a random direction's 9.2%, real against random -0.0750 at item-clustered p=0.0001, while on the paper's affordability items real falls from 32.5% to 21.4% of a +0.1171 gap and real against random moves from p=0.1386 to -0.0200 at p=0.4236.
**Failed approaches.** Matching on any prefix of a distinctive word conflates "baker" with "bakery", so the rule allows only a plural suffix; items where one name contains the other have no distinguishing words at all and fall back to the old whole-name position rule, which is what the existing containment test covers.
**Command:**
uv run -m scripts.rescore_causal_picks --run_dir results/raw/260906_ablation_specificity_v1 --items_source ceiling --value pro-novelty --output_dir results/tables/260909_rescored_ceiling_v1 --seed 42
uv run -m scripts.rescore_causal_picks --run_dir results/raw/260909_ablation_paper_items_affordability_v1 --items_source paper --dataset_repo chloeli/pro-affordability-item-comparisons --value pro-affordability --output_dir results/tables/260909_rescored_paper_items_v1 --seed 42
uv run -m scripts.run_clean_subset_ablation --run_dir results/tables/260909_rescored_ceiling_v1 --labels_path data/processed/novelty_ceiling_confound_labels.json --output_dir results/tables/260909_rescored_ceiling_v1 --seed 42
**Output:** results/tables/260909_rescored_{ceiling,paper_items,n100}_v1/

## 260909 — value-loaded share of the plan-first seeds' comparisons

**Question.** What share of each plan-first seed's distinct pairwise comparisons actually discriminate on the installed dimension, the third field the plan-first pre-registration said would be logged?
**Result.** The judge failed its calibration gate again, at 1.000 on-dimension and 0.525 off-dimension against a required 0.30, one label away from the 0.550 it scored on 260906, so these shares are descriptive and not evidential: across the five plan-first seeds that used the pairwise tool at all the shares are 0.226, 0.095, 0.208, 0.423 and 0.316, mean 0.254, against 0.371 for the six pooling seeds; the sixth plan-first seed, pro-affordability seed 43, has no row because it spent all 20 of the queries it used on single-model calls and never called the pairwise tool. The correct-against-wrong bootstrap was skipped rather than computed, because `cluster_bootstrap` pairs its arms by item and the arms here are 3 correct seeds against 26 wrong ones.
**Command:**
uv run -m scripts.run_query_composition --output_dir results/tables/260909_composition_planfirst_v1 --model_id anthropic/claude-sonnet-5 --seed 42 --run_globs "results/raw/260905_enumerated_*/runs.jsonl,results/raw/260905_budget200_*/runs.jsonl,results/raw/260905_pooling_*/runs.jsonl,results/raw/260909_planfirst_*/runs.jsonl"
**Output:** results/tables/260909_composition_planfirst_v1/composition.json and per_seed.jsonl

## 260910 — powered ablation of the affordability diff direction on the paper's full released eval

**Question.** At the largest sample the paper's own eval can offer, does ablating the affordability diff direction close the value gap more than a random direction does?
**Hyperparameters.** pro-affordability at the msm-only rung, the layer 22 direction projected out across all 32 layers, 497 items at 8 samples for 3,976 trials per arm and 497 item-clustering units, temperature 0.7, seed 42.
**Datasets.** All 497 items of `chloeli/pro-affordability-item-comparisons`, order-balanced by `build_items`.
**Metric choice.** Unchanged, share of the msm-minus-control gap closed, tested with an item-clustered bootstrap.
**Result.** The gap is +0.1044 at item-clustered p=0.0001 with an interval of +0.0832 to +0.1253, and the real direction closed 52.3% of it against the pooled random band's 4.2%, a difference of -0.0502 with an interval of -0.0711 to -0.0295 at p=0.0001, while beating the generic direction by -0.0606 and pro-novelty's direction by -0.0539, both at p=0.0001; coherence ran 0.932 to 0.976 against 0.970 unablated.
**Failed approaches.** The same test on 100 of the 497 items returned 21.4% closure and p=0.4236, which an 80% power calculation at the observed effect size showed needed about 9,579 trials per arm against the 800 it had; that interval, -0.0692 to +0.0296, contains this run's -0.0502, so the earlier result was an underpowered subsample rather than a contradiction.
**Command:**
uv run -m scripts.run_ablation_specificity --value pro-affordability --other_value pro-novelty --items_source paper --dataset_repo chloeli/pro-affordability-item-comparisons --vector_bundle results/raw/260909_recapture_response_trace_v1/response_vectors.pt --n_pairs 497 --n_repeats 8 --n_random 3 --output_dir results/raw/260909_ablation_paper_items_full_v1 --model_id meta-llama/Llama-3.1-8B --seed 42
uv run -m scripts.run_clean_subset_ablation --run_dir results/tables/260910_rescored_paper_full_v1 --output_dir results/tables/260910_rescored_paper_full_v1 --seed 42
**Output:** results/raw/260909_ablation_paper_items_full_v1/specificity.json, results/tables/260910_rescored_paper_full_v1/clean_subset.json

## 260910 — the published diffing toolkit on our checkpoints, readout half

**Question.** Is this project's white-box null a property of our own implementation of logit lens and Patchscope, or does the published method behind the readable-traces paper also fail to name the value on the same checkpoints?
**Hyperparameters.** science-of-finetuning/diffing-toolkit at commit c3f3d102, activation_difference_lens at its own defaults and nothing tuned: fineweb-1m-sample as the unrelated text, 10000 samples, relative layer 0.5 which is absolute layer 15 of 32, positions 0 to 5 for auto-Patchscope and steering with logit lens cached at all 128, their Patchscope prompt, their coherence-threshold search, graders openai/gpt-5-mini and openai/gpt-5-nano through OpenRouter.
**Datasets.** Three organisms against the raw base meta-llama/Llama-3.1-8B: chloeli pro-novelty MSM-only, pro-affordability MSM-only, and the control adapter that had no spec, each as a LoRA adapter with its revision recorded in config.json.
**Metric choice.** Their own token-relevance grader, which labels each of the top 20 readout tokens relevant or irrelevant against the organism's ground-truth description, because it is their instrument answering their question rather than ours re-scoring their output.
**Result.** Their grader calls 2 of 200 tokens relevant on the control, 3 of 200 on pro-novelty and 5 of 200 on pro-affordability, so neither value organism is distinguishable from a checkpoint with no value installed, at Fisher exact p=1.000 and p=0.449 against the control. The tokens themselves are junk in every cell, including `<|end_of_text|>` and `Specifier` called relevant to a description stating no value was installed. Their steering coherence thresholds overlap completely across the three organisms, at 1.28 to 1.54 for the control against 1.43 to 1.55 and 1.32 to 1.62 for the two values.
**Failed approaches.** Launching with pipeline.mode=diffing alone runs their readouts but not their agent, which needs pipeline.mode=evaluation; and their agent then fails on every LoRA organism with `Unexpected keyword argument 'lora_local_path'` against the vLLM their own lockfile installs, which is recorded with its one-keyword fix in docs/decisions.md.
**Command:**
uv run python main.py diffing/method=activation_difference_lens organism=msm_control model=llama31_8B infrastructure=local wandb.enabled=false pipeline.mode=diffing
uv run -m scripts.build_toolkit_table --toolkit_dir results/raw/260909_toolkit_v1 --output_dir results/tables/260909_toolkit_table_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/tables/260909_toolkit_table_v1/toolkit_table.md and toolkit_results.json, over results/raw/260909_toolkit_v1/

## 260910 — the published diffing toolkit on our checkpoints, agent half

**Question.** Does the toolkit's interpretability agent, which identifies fine-tuning objectives with high accuracy on the narrow-finetuning organisms in its own paper, name the installed value on the MSM checkpoints, and what does it say about a checkpoint with no value installed?
**Hyperparameters.** Their evaluation pipeline at their defaults except agent num_repeat cut from 5 to 2 on compute, pre-registered before any agent output existed: agent openai/gpt-5, grader openai/gpt-5-mini at num_repeat 3, budget 100 agent LLM calls and 50 model interactions per run, six runs over three organisms. Their agent queries the raw base model and the adapter, both chat-templated.
**Datasets.** The same three organisms as the readout half, over the cached activation differences that pass produced.
**Metric choice.** Their agent's final description, scored blind by our own forced-choice judge over the five pilot values plus none, so their pipeline and ours are graded by the same instrument on the same rule.
**Result.** Our judge picks none on all six, and no run names the installed value. All four value-organism runs invent a developer-tools domain: pro-novelty as StackOverflow Q&A on scientific image I/O and matplotlib, pro-affordability as C# System.Drawing and matplotlib figure styling. Both control runs, on an adapter with no value installed at all, invent Android Jetpack Compose View interop in detail, cite `AndView` promoted by the layer-15 logit lens as their evidence, and include verification sections claiming the hypothesis was confirmed by querying the model. Their agent had an explicit Inconclusive option in its own output grammar and used it in none of the six runs.
**Failed approaches.** Their agent cannot run on a LoRA organism unpatched, failing on `lora_local_path` against the vLLM their lockfile installs; the one-keyword fix and why it was chosen over their `use_vllm=false` path is in docs/decisions.md.
**Command:**
uv run python main.py diffing/method=activation_difference_lens organism=msm_control model=llama31_8B infrastructure=local wandb.enabled=false pipeline.mode=evaluation diffing.evaluation.agent.num_repeat=2
**Output:** results/tables/260909_toolkit_table_v1/toolkit_table.md and toolkit_results.json

## 260910 — the pooling rung rerun with independent samples, to test the budget claim properly

**Question.** Every 200-query rung was scored with a fixed vLLM sampling seed, so repeats of a prompt returned identical text and the agent saw about a third of the evidence it was charged for; does the pooling rung still score zero when the samples are genuinely independent?
**Hyperparameters.** Identical to the original pooling rung except one line: the per-call seed is removed from `SamplingParams` in `src/agent/serve.py`. Prompt v3, budget 200, n=5 at temperature 0.7, pairwise tool on, gpu_fraction 0.35, agent and judges claude-sonnet-5, seeds 42/43/44 on both values.
**Datasets.** The same two checkpoints as every other rung, chloeli pro-affordability and pro-novelty MSM adapters against chloeli/llama-3.1-8b-baseline.
**Metric choice.** Unchanged, seeds naming the installed value correctly, plus the ratio of distinct completions seen to pairwise queries charged as a validity gate fixed at 0.80 before the run.
**Result.** The gate passes at a pooled 0.875 against 0.37 for the old sampling, so the fix worked and the run is a real test of budget; the result is 0 of 6, unchanged from the 0 of 6 the same rung scored on degraded sampling. Five of the six seeds spent 160 to 190 pairwise queries and saw 137 to 173 distinct completions each; the sixth, pro-novelty seed 44, chose to answer after 10 queries and is not a budget test, so the claim rests on 0 of 5 at roughly 150 effective samples.
**Failed approaches.** None new; the one-line change was identified by measuring the distinct-text ratio across the twelve existing 200-budget runs rather than by reading the code.
**Command:**
uv run -m scripts.run_agent_diffing --value pro-novelty --budget 200 --seeds 42,43,44 --prompt_version v3 --n_samples 5 --pairwise_tool True --target_temperature 0.7 --gpu_fraction 0.35 --output_dir results/raw/260910_pooling_indep_pro-novelty_v1 --model_id anthropic/claude-sonnet-5 --seed 42
**Output:** results/raw/260910_pooling_indep_pro-affordability_v1/summary.json and results/raw/260910_pooling_indep_pro-novelty_v1/summary.json
