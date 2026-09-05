# Result log

The verdict for every run that was trying to learn something: did it clear the bar pre-registered in `docs/decisions.md`, and what
happens next. The companion to `research_log.md`, which records what was run. Newest entries at the
top.

The entry is also the report presented in chat, verbatim.

`docs/rules.md` holds the entry template and the rules for filling it in. Read it before appending —
do not improvise a format from the entries below.

## 260904 — pilot gate, blind agent vs pro-novelty MSM checkpoint

**Tested:** whether a Claude agent with 30 blind queries can name the value MSM installed in the pro-novelty checkpoint against the no-spec control.
**Found:** wrong on 3/3 seeds at the full budget, with 0 of 90 queries naming a sweetener and 16 touching food at all, while a 4-prompt hand probe on the same adapters separated the arms immediately.
**Bar:** missed, in the pre-registered "hard" direction.
**Verdict:** rerun with one named change — remove "value or preference" from the agent's system prompt, which sent all three seeds into moral-values probing rather than consumer-preference probing.

## 260904 — gate run 2, neutral reword across five checkpoints

**Tested:** whether removing the "value or preference" framing lets the agent name the installed value on any of five checkpoints in 30 blind queries.
**Found:** wrong on 15/15 seeds at full budget, decomposing post-hoc into 0/9 on checkpoints that move on ordinary domain questions and 0/6 on checkpoints that move only when the prompt poses their tradeoff; 13/15 seeds confidently named a wrong difference and 2/15 reported no difference found.
**Bar:** missed, in the pre-registered "hard" direction, on all five.
**Verdict:** rerun with one named change — return n=5 samples per arm at temperature 0.7 per query and require a difference to hold across the majority, since greedy single completions let the agent read sampling noise as a stable disposition.

## 260904 — prompt-bias guard, run 1 wording vs run 2 wording on two tier-1 checkpoints

**Tested:** whether run 1's wording outperforms run 2's on the two checkpoints whose values sit closest to moral framing, isolating prompt with everything else held fixed.
**Found:** both wordings wrong on both checkpoints at full budget, and run 1's wording returned the same "animal welfare" answer for pro-america and pro-tradition, two different installed values.
**Bar:** no pre-registered bar; this was a guard on the run 1 to run 2 comparison, not a gate.
**Verdict:** extend — the framing choice steers what the agent confabulates rather than whether it succeeds, so the run 1 to run 2 story is reported that way and neither prompt is described as the fixed one.

## 260904 — run 3, noise-controlled agent across five checkpoints

**Tested:** whether returning 5 samples per query at temperature 0.7, with a majority-consistency rule, lets the agent recover any installed value in 30 queries.
**Found:** 0/15 correct, with abstentions rising 3/15 to 14/15 and confident-wrong falling 12/15 to 1/15; across 252 queries on pro-novelty spanning all three conditions the agent never once named a sweetener while reaching the food domain in 22% of queries.
**Bar:** missed, in the pre-registered "hard" direction, on all five checkpoints and both elicitation tiers.
**Verdict:** extend to the white-box arm — noise control fixed the honesty of the failure without touching its cause, which is that the agent searches domains and not the item-level contrasts where these values live.

## 260904 — white-box trace arm, five checkpoints, three vectors, three readouts

**Tested:** whether an activation-difference trace recovers the installed value on the five checkpoints the black-box agent failed on.
**Found:** 0/5 by every route — logit lens and patchscope named no value at any of four layers on raw, control-subtracted or pairwise vectors, and steering peaked at 0.05 against a random-direction control at 0.05 — while the raw adapter-minus-base diff proved 0.95 to 0.99 cosine-identical across five unrelated values and 0.93 to 0.99 aligned with a control checkpoint carrying no value at all.
**Bar:** missed on the pre-registered criteria, correct pick on 2 of 4 layers or steering above 0.50 with random below 0.10, on all five checkpoints.
**Verdict:** extend — the 2x2 is complete at 0/5 and 0/5, and the substantive result is the method finding that a naive fine-tune-minus-base diff is almost entirely a generic training direction, whose own readout is assistant-opening tokens.

## 260905 — paper's OOD preference eval on our adapter copies

**Tested:** whether our adapter copies reproduce the paper's OOD value-aligned preference rate, and what that implies for the black-box null.
**Found:** pro-affordability 0.379 against control 0.210 on 500 comparisons, a +17pp effect matching the published +13pp, value-specific and present in both item orders; a two-proportion power calculation puts the queries needed to detect it at 196, against the 30-query budget the agent was given, and pro-america's published effect of -2pp is undetectable at any budget.
**Bar:** no pre-registered bar; this is the verification step the pivot required before anything downstream is interpretable.
**Verdict:** extend — the black-box null is a statistical power result rather than a search-failure result for four of five adapters, and the earlier "two-level search" diagnosis is withdrawn.

## 260905 — budget-matched agent runs with a pairwise-preference tool

**Tested:** whether a budget matched to the measured effect size, plus a forced-choice pairwise tool, lets the agent recover the value.
**Found:** 0/6 correct at budget 200, with the agent adopting the pairwise tool for 60-80% of its queries but probing only 8-18 distinct item pairs such as Coke versus Pepsi and Python versus JavaScript, none of which carry the affordability or novelty contrast, and on pro-novelty it stopped 54-72 queries short of its budget.
**Bar:** missed, on the pre-registered read that finding the value at budget 200 would mean budget and question shape were the barrier.
**Verdict:** extend — the barrier is item selection rather than budget or question format, which is a claim about black-box auditing that the earlier underpowered runs could not have supported.

## 260905 — detectability ceiling and the pooling-instruction runs

**Tested:** whether the value is behaviourally detectable at the agent's budget, and whether the agent's failure is bad allocation or failed search.
**Found:** the oracle protocol detects it in 97.8% of bootstrapped subsets at 200 queries (median p 0.0066, +19.2pp) while the agent scores 0/6 at the same budget; being told to pool roughly doubled its distinct comparisons from 12 to 21 but still left it short of the 50 the protocol used, and zero of its 98 distinct pairs discriminated on the value.
**Bar:** the pre-registered read was that finding the value after the pooling instruction would mean allocation was the barrier; it did not find it, so the search half dominates.
**Verdict:** extend to the MSM+AFT rung — the binding constraint is hypothesis generation rather than budget, question format or allocation, and the ceiling is an oracle upper bound because knowing the value is what makes discriminating items writable.

## 260905 — trace arm on the msm+aft rung, in-domain and held-out captures

**Tested:** whether a stronger, generalizing installation of the value produces a readable activation trace, on both training-domain and held-out comparisons.
**Found:** 0/4 layers named in all four conditions, but the topic-conditional signal roughly doubled, from a 1.5-2.1 in-domain over off-domain norm ratio on msm-only to 2.6-5.0 on msm+aft, while pro-novelty's readable tokens moved from 14 novelty stems and 0 assertion tokens to 1 and 15, and to 0 and 10 on the held-out capture.
**Bar:** missed the pre-registered readout criterion of 2 of 4 layers, on every checkpoint and capture set.
**Verdict:** extend to the enumerated-hypothesis agent row — the white-box column is a null at the readout level, but the diff is measurably topic-conditional and its readable content is the training stage's behavioural signature rather than the value, which is the same result as the generic-direction finding one rung up.

## 260905 — enumerated hypothesis space, nine seeds, and the pre-answer probe

**Tested:** whether enumerating candidate dimensions rescues the black-box agent, and whether the diff direction predicts the model's choice from its pre-answer state.
**Found:** 3/9 seeds correct with the candidate list against 0/6 without it, two of the three confirmed by both judges, and success tracked probing behaviour with correct seeds spending 0.31 of their comparisons on value-loaded pairs against 0.13 for failures; the pre-answer probe peaked at AUC 0.593 against a pre-registered 0.65 bar, with control arms at 0.40 to 0.52 and random directions at 0.50.
**Bar:** the enumerated row has no pre-registered threshold and is reported as a rate; the probe missed its 0.65 bar and fell below the 0.60 floor.
**Verdict:** extend to the cross-model replication — enumerating the hypothesis space is necessary but not sufficient for the agent, and the "present but unreadable" claim is withdrawn because the direction does not linearly predict the choice it was supposed to explain.

## 260905 — qwen 2.5 32b replication of the generic-direction finding

**Tested:** whether the raw diff being mostly a generic training direction holds on a second model family.
**Found:** partial replication, with between-spec cosine 0.930 and cosine against the control diff 0.873 at layer 48 of 64, against 0.414 and 0.094 at layer 16, where llama showed 0.947 and 0.914 at the same relative depth; the qwen generic direction's logit lens is uninterpretable where llama's read as assistant openers.
**Bar:** no pre-registered bar; this is a robustness check on the headline finding.
**Confound:** the qwen specs are variants of one document family rather than unrelated values, and the run was 8-bit.
**Verdict:** rerun with one named change — state the finding as replicating at depth only, since the base-versus-instruct explanation I proposed for the early-layer gap was tested and refuted.

## 260905 — steering retry with alpha scaled to the layer's residual norm

**Tested:** whether the steering arm failed because of a scale error, and whether the diff vector causally moves the value-relevant choice once the scale is right.
**Found:** the scale error was real, since the old code injected perturbations of norm 73 against a residual norm of 25.5, and fixing it restored coherence from 0.00 to 0.95-1.00; but every condition including the system-prompt positive control stayed within 0.40 to 0.60 novel-sweetener rate against a 0.55 baseline, and at 20 greedy samples per condition the test resolves only 0.05 where detecting a 15 point shift needs about 160.
**Bar:** no pre-registered bar was met, since the positive control did not move the behaviour it was built to move.
**Verdict:** rerun with one named change — raise the samples per condition from 20 to at least 160, because the arm is now underpowered rather than broken, which is the same error this project already diagnosed in the black-box arm.

## 260905 — corrected causal steering test on discriminating prompts

**Tested:** whether the control-subtracted diff direction causally moves the value-aligned choice, with a positive control demonstrated to work on the same prompts and scales.
**Found:** the positive control produced a monotone dose-response from 0.633 at the smallest scale to 0.849 at the largest, beating the random-direction null at p=0.0014 and p<0.00001, while the diff direction sat at 0.475 to 0.547 and never separated from random, the largest gap being p=0.10.
**Bar:** the pre-registered read was that a working positive control alongside a flat diff vector is a controlled causal null, and that is what happened.
**Confound:** the greedy baseline reads 1.000 because every prompt names the value-aligned item first, so position bias saturates it; the random direction at matched alpha is the null used instead.
**Verdict:** extend — the direction is not causally load-bearing for the behaviour, and this now rests on a demonstrated positive control rather than on an uninformative arm.

## 260905 — cosine between the steerable direction and the midtraining diff

**Tested:** whether the direction midtraining installed the value along is the same one a system prompt uses to produce that behaviour, at layer 22 of llama 3.1 8b.
**Found:** cosine +0.053 for the response-position in-domain msm-only diff and -0.049 for a last-token variant, against a random-direction null averaging 0.015 and peaking at 0.029, while the system-prompt direction steers to 0.849 and the diff does not move at all.
**Bar:** no pre-registered bar; this was a ten-minute check proposed after the causal result, and is reported as post-hoc.
**Confound:** the two capture variants of the diff agree with each other at only 0.187, so the diff direction is unstable across capture choices and this measures a family of vectors rather than one.
**Verdict:** extend — document-based midtraining produced the disposition without using the prompt's direction, which is the opposite of the subliminal learning result for finetuning on a steered teacher.

## 260905 — ablation of the diff direction on order-balanced held-out pairs

**Tested:** whether removing the diff direction undoes the value-aligned behaviour, with item order balanced and a verified effect present to remove.
**Found:** the msm minus control gap is +0.119 at p=0.027, and ablating the diff direction closed 39% of it against 61% for a random direction, with real versus random at p=0.62; balancing also exposed a position effect the earlier item set hid, msm picking the value-aligned item at 0.613 when it is named first and 0.797 when named second against control's 0.570 and 0.600.
**Bar:** no pre-registered threshold; this replicates the causal null on balanced items after the first ablation was run on prompts where no effect existed.
**Confound:** 160 samples per condition detects a full removal of a 0.119 effect but not a partial one, so a direction carrying half the effect would not be distinguished from noise.
**Verdict:** extend — the causal arm now has two legs, steering and ablation, agreeing that the diff direction is not load-bearing, and the order-balance caveat on the steering item set is closed.

## 260905 — signed projection of the diff onto the prompt's steering direction

**Tested:** whether the diff contains a component along the prompt's steering direction, on both capture variants, against a 200-direction random null.
**Found:** both variants clear the null band but with opposite signs, +0.133 or +5.3% of norm for the response-position capture and -0.240 or -4.9% for the last-token capture, so the overlap is real, tiny, and sign-unstable across capture choice rather than absent.
**Bar:** neither pre-registered read applied cleanly, since one branch assumed a positive component and the other assumed none.
**Confound:** the proposed follow-up, rescaling the projected component to a working steering alpha, is degenerate because that vector is parallel to the prompt direction and the test reduces to the positive control.
**Verdict:** discard the "installed along an unrelated direction" phrasing — the card is now "no consistent overlap", tagged as a control rather than an effect, and the subliminal-learning contrast is one flagged sentence.

## 260905 — projection of the diff onto the prompt direction, fully paired 2x2

**Tested:** whether the diff contains a component along the prompt's steering direction, across prompt set and capture position, with every quantity measured on one fixed set of generated strings.
**Found:** no cell has a positive component; three of four are negative at -3.2%, -6.2% and -19.2% of the diff's norm and the fourth is inside the random band at +0.5%.
**Bar:** neither pre-registered read applied, since one assumed a positive component and the other assumed none detectable.
**Confound:** an earlier version of this table reported opposite signs across two capture variants, but those differed in prompt set, position and generated text at once, and each arm wrote its own continuation, so the diff there conflated representation with word choice; that result is withdrawn.
**Verdict:** extend — with pairing enforced the sign is consistently negative or null rather than unstable, which supports the no-overlap card more cleanly than the version it replaces.

## 260905 — where the value lives: the instruction contrast built inside each checkpoint

**Tested:** whether the weight diff overlaps the value's direction inside the model training produced, as opposed to inside the model it started from.
**Found:** the diff projects positively onto the fine-tuned model's own instruction contrast in all four cells, +0.194 and +0.210 or 8.2% and 8.4% of its norm at response positions and outside the random band there, while projecting negatively or null onto the control's contrast in the same cells; the two contrasts are related but distinct at cosine 0.31 to 0.80, and the fine-tuned model represents the contrast more strongly in three of four cells.
**Bar:** no pre-registered bar; this was a follow-up proposed after the causal null and is reported as post-hoc.
**Confound:** only two of four cells clear the null, the overlap is about 8%, and this is one value at one layer.
**Verdict:** extend — the training did write weakly along a value direction, but the model's own post-training one, which reframes the card from "no overlap" to "overlap with the representation training produced, too small to steer or to matter causally".
