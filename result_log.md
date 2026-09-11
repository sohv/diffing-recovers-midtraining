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

## 260905 — adapter control on the intra-model overlap

**Tested:** whether the diff's overlap with the instruction contrast built inside the fine-tuned model is specific to the adapter holding that value.
**Found:** it is not, since adapters holding unrelated values overlap comparably, pro-america at +7.2% and pro-tradition at +5.5% against pro-novelty's +8.2% and +8.4%, all outside a 200-direction random band, and pro-america flips sign across its two significant cells.
**Bar:** the control's pre-registered kill condition was any other adapter reaching about 8%, and pro-america reached 7.2%.
**Verdict:** discard — the overlap is adapter-internal geometry rather than the value's direction, the lead is deleted from the write-up, and the card returns to the causal null it had before.

## 260906 — free probe on the pre-answer state, pro-affordability at the msm+aft rung

**Tested:** whether the model's upcoming value-aligned choice is linearly decodable pre-answer at all, which decides whether the diff direction's 0.593 means absence or misdirection.
**Found:** leave-one-domain-out mean AUC 0.765 on 96 held-out trials against 0.533 for the diff direction and 0.503 for random directions scored on the identical trials, with a label-shuffled fit at 0.522; four of five domains ran 0.75 to 0.89 and one, entertainment, sat at 0.479; the control adapter's choices decoded at 0.831 and the two arms' probe directions agree at cosine 0.416.
**Bar:** the primary metric cleared it, free probe at or above 0.70 with the diff direction below 0.60, but the pre-registered secondary 60/40 split returned 0.547, and the amendment written before the run says a disagreement between the two is reported as inconclusive.
**Confound:** the control adapter decodes at least as well as the value-holding one, so the probe reads which item is about to be named rather than the installed value; nothing here isolates the value, and the number cannot be described as "the value is linearly represented".
**Verdict:** extend to the ablation positive control — the decodable quantity is the choice, not the value, and whether that direction is load-bearing is what the ablation arm now tests with it.

## 260906 — ablation on the strongest checkpoint, and the positive control that failed

**Tested:** whether ablating the diff direction removes the value-aligned behaviour where that behaviour is unmistakable, with a direction fitted to predict the choice included as the positive control the arm never had.
**Found:** the msm+aft checkpoint picks the value-aligned item at 0.744 against control's 0.269, a +0.475 gap at trial-level z=8.44 and item-clustered p=0.0001 with a 95% interval of +0.317 to +0.629, present in both item orders; ablating the real diff direction across layers 18-31 left the rate at 0.776 and ablating a random direction left it at 0.788, a -0.012 difference at clustered p=0.44, and the free-probe direction returned the identical -0.012 at p=0.58.
**Bar:** the pre-registered read was that ablating the real direction closes more of the gap than random at p<0.05, which it did not; the pre-registered fallback also triggered, since the positive control failed to beat random too.
**Confound:** the probe direction is a layer-26 object ablated across layers 18-31, and the probe it came from reads the impending choice rather than the value, so it is a weaker positive control than a demonstrated causal handle would be.
**Verdict:** discard the ablation arm as evidence — per the pre-registration, a positive control that cannot move a +47.5pp behaviour makes this arm uninformative rather than null, the pro-novelty ablation's "second causal leg" is withdrawn with it, and the causal claim rests on steering, which has a working positive control.

## 260906 — controlled causal null on the checkpoint with the largest installed effect

**Tested:** whether the midtraining diff direction causally moves the value-aligned choice on pro-affordability at the msm+aft rung, where the behavioural effect is +47.5pp rather than the +4.2pp of the checkpoint the causal column previously ran on.
**Found:** the system-prompt positive control produced a monotone dose-response, 0.500, 0.579, 0.692, 0.898 across the four scales, beating the matched-alpha random direction at p<0.0001 at the top three and reaching z=8.34; the diff direction sat at 0.333 to 0.450 across the same scales and never separated from random, its largest gap being -0.100 at p=0.068 with the sign against it.
**Bar:** the pre-registered read was that a working positive control alongside a flat diff vector is a controlled causal null, carried over verbatim from the pro-novelty run, and that is what happened.
**Confound:** the greedy baseline reads 1.000 because every steering prompt names the value-aligned item first, so position bias saturates it and the random direction at matched alpha is the null used instead; the ablation arm that would have been the second leg failed its own positive control and is not counted.
**Verdict:** extend — the causal null now rests on a checkpoint whose installed effect is unmistakable at +47.5pp and item-clustered p=0.0001, which is the result the previous version of this column could not support, and pro-novelty becomes the weak-installation cell rather than the headline.

## 260906 — the probe's random-direction null was misaligned, and fixing it strengthens the null

**Tested:** whether the four probe cells survive a corrected null, after the random-direction AUCs were found to be computed on `acts[:len(labels)]`, the first n activations, while the labels were the ones surviving the drop of undecided trials.
**Found:** all eight headline AUCs are unchanged to three decimals, 0.593, 0.541, 0.420 and 0.370 on the value arms and 0.455, 0.521, 0.438 and 0.404 on the controls, since the real direction's scores were always built correctly inside the loop; every corrected null band is wider, with the value arms' random maxima moving to 0.620, 0.664, 0.648 and 0.648, so the best cell's 0.593 now sits below its own random maximum instead of a whisker above it.
**Bar:** unchanged and still missed — no cell reaches the pre-registered 0.65, and none reaches the 0.60 floor either.
**Confound:** none introduced; the fix removes one. The bug affected 1 to 4 trials per cell, which is why the null moved by 0.02 to 0.08 rather than more.
**Verdict:** extend — the "present but unreadable" withdrawal stands and is now better supported, since the diff direction does not beat its own random null on any cell, and the `_v2` runs are retained as the record of the contaminated comparison.

## 260906 — the overlap with the trained model's own direction is five times larger where installation is strong

**Tested:** whether the projection of the weight diff onto the instruction contrast built inside the fine-tuned model scales with how strongly the value was installed, rerunning the pro-novelty protocol unchanged on pro-affordability at the msm+aft rung.
**Found:** 41.9% and 48.9% of the diff's norm at the two response-position cells, both outside the random band, against pro-novelty's 8.2% and 8.4%; the last-token cells give 2.8% inside the band and 13.0% outside it, so the effect is at response positions as before, and the trained model represents the contrast more strongly than the control does in three of four cells.
**Bar:** no bar applies yet. This result is not interpretable, and is not reported in any form, until the adapter control that killed the equivalent pro-novelty lead has been run at this rung.
**Confound:** the entire reason this is withheld — the pro-novelty version of this number looked like a finding at 8.2% until adapters holding unrelated values were shown to reach 7.2% and 5.5%. A larger number is not evidence of specificity.
**Verdict:** extend, pending the adapter control pre-registered before this was read — if unrelated adapters at the same rung reach within about 2 points of 41.9%, this is discarded on the same terms the last one was.

## 260906 — the larger overlap is the training stage, not the value

**Tested:** whether the 41.9% and 48.9% overlap found on pro-affordability at the msm+aft rung is specific to the adapter holding that value, on the same terms that killed the equivalent pro-novelty lead.
**Found:** it is not; pro-america reaches 48.5% against pro-affordability's 48.9% in the tradeoff/response cell, a gap of 0.4 points, and the no-value control adapter reaches 26.4% in the same cell and 19.5% in the other, so every adapter at this rung overlaps the contrast substantially and the ordering across the four value adapters spans only 26.9% to 48.9%.
**Bar:** the pre-registered kill condition was any unrelated adapter landing within about 2 points of the target, and pro-america landed within 0.4.
**Confound:** none outstanding; this is the control, and it also explains the five-fold gap against pro-novelty's 8.2% — the projection scales with the training stage rather than the value, since the control adapter alone moves from near zero at msm-only to 26.4% at msm+aft.
**Verdict:** discard — the overlap is adapter-internal geometry that grows with the amount of training, the "overlap scales with installation strength" reading is withdrawn before it was ever written up, and the white-box card stands unchanged at no measurement locating the value.

## 260906 — the agent does not invent differences, so its wrong answers are about search

**Tested:** the black-box column's missing null arm — how often the agent asserts a difference between two checkpoints that are weight-identical, measured under the same configuration as the enumerated runs it is meant to calibrate.
**Found:** 0 of 9 seeds asserted a difference and all 9 reported none found, a confabulation rate of 0.00 against 11 of 12 seeds asserting one on genuinely differing pairs, p=0.0001; the abstaining seeds were not idle, spending 108 to 200 of their 200 queries and naming up to 19 candidate dimensions they had ruled out.
**Bar:** the pre-registered read was that a rate below 0.2 means the agent's assertions are informative rather than credulous, and 0.00 clears it.
**Confound:** this measures the five-sample configuration with the majority-consistency rule, not the single-sample runs where 13 of 15 seeds named a wrong difference; that earlier rate belongs to a configuration this project already replaced for exactly this reason, and nothing here retroactively licenses it.
**Verdict:** extend — the black-box null is a search result rather than a credulity result under the configuration it was measured in, the agent abstains correctly when there is nothing to find, and the 3 of 9 enumerated successes sit against a demonstrated 0 of 9 false-positive floor.

## 260906 — rank-1 ablation is directionally real and roughly sixteen times underpowered

**Tested:** whether projecting out a single direction across layers 18-31 can remove a behaviour whose cause is known and enormous, replacing the probe direction that was the wrong instrument for this job.
**Found:** the affordability system prompt moves the control adapter from 0.333 to 0.980, a +64.7pp effect at p<1e-6; ablating that prompt's own contrast direction returns 0.954 while a matched-norm random direction returns 0.994, so the true causal direction is distinguishable from random at item-clustered p=0.026 but removes only 4.0% of the effect it causes, and the contrast-minus-random gap of 0.040 is 6.2% of it.
**Bar:** the pre-registered read was that the contrast dropping the rate toward the no-system baseline while random left it high would make the arm informative; the rate did not drop, so the arm is uninformative, now on direct evidence rather than on a probe direction's failure.
**Confound:** an in-context instruction is re-derived from the prompt tokens at every layer, including the seventeen layers below the ablation span, so this is a hard case for ablation and may understate what the method could do against a weight-encoded disposition; what it establishes is that no sensitivity has been demonstrated, not that none exists.
**Verdict:** extend — the earlier "uninformative" verdict stands but its justification is replaced, and the sharper statement is that ablation here recovers 6% of a known causal effect, so the diff direction's -0.012 against random is far inside the method's noise floor and licenses no conclusion either way.

## 260906 — the adapter control could not answer its own question, and the corrected control finds a value-specific component

**Tested:** whether the weight diff's overlap with an instruction contrast is specific to the diff's own value, varying the contrast's value rather than the adapter computing it, which is what the adapter control failed to vary.
**Found:** each diff projects most strongly onto its own value's contrast, +33.0% of norm for affordability and +15.7% for novelty, both far outside a 200-direction band; the off-diagonals are explained by the contrasts not being independent, with affordability and america at cosine +0.720 and novelty and tradition at -0.908, the same axis reversed; residualising the own contrast against the span of the other three leaves +25.5% for affordability, 8.4 times the band edge with 69% of the contrast surviving, and +6.9% for novelty, 2.1 times the edge with 33% surviving.
**Bar:** not cleanly met. The bar required the diagonal to exceed every off-diagonal and the off-diagonals to sit at or below the band; novelty met both, affordability met the first but projects +20.7% onto america and +13.5% onto tradition, both outside the band.
**Confound:** the residualisation was added after that table was seen and is POST-HOC wherever it appears; and the whole comparison is representational, so it says nothing causal, which matters because the same direction does not steer and the ablation arm recovers only 6% of a known cause.
**Verdict:** rerun with one named change — add pro-america at the msm+aft rung as a third value on one fixed prompt set, pre-registered before its number is seen, since the adapter control is withdrawn as uninformative in both directions and the affordability and novelty results stay exploratory until the pattern replicates out of sample.

## 260906 — corrected span turns the ablation arm into a working instrument

**Tested:** whether rank-1 ablation is genuinely weak here, or whether the 4% sensitivity figure was an artefact of ablating only layers 18-31 when the standard implementation ablates every layer.
**Found:** with all 32 layers ablated the same contrast removes 33.4% of the known +64.7pp effect, moving the rate from 0.980 to 0.764 while a matched-norm random direction leaves it at 0.974; contrast against random is -0.209 at item-clustered p=0.0001, where random against unablated is -0.007 at p=0.46, so the instrument separates a true causal direction from a random one cleanly.
**Bar:** the pre-registered bands were at least halfway back to baseline for "the span was the artefact", under a quarter for "rank-1 ablation is genuinely weak", and between the two reported as measured with no verdict; 33.4% falls in the middle band, so it is reported as the number and run 2 is mandatory.
**Confound:** an in-context instruction is re-derived from the prompt at every layer, so a third removed is a floor on what this instrument can do against a weight-encoded disposition rather than a ceiling; and the earlier 4% figure stands only as a statement about the partial span.
**Verdict:** rerun with one named change — the diff-direction ablation moves to all 32 layers on pro-novelty msm-only, because the previous null was measured with an instrument at a twelfth of this sensitivity and is not yet interpretable.

## 260906 — with a working instrument the ablation points the other way, and is underpowered to say so

**Tested:** the diff direction's causal contribution on pro-novelty msm-only, ablated across all 32 layers rather than 18-31, now that the instrument has been shown to remove 33.4% of a known cause.
**Found:** the real direction closed 56.5% of the +0.119 msm-minus-control gap against a random direction's 1.6%, which reverses the partial-span result where random closed 61% against real's 39%; but the raw rate difference is only -0.066, 0.637 against 0.703, at bootstrap p=0.226 and item-clustered p=0.271, and the gap being closed is itself p=0.103 item-clustered at twenty items.
**Bar:** the pre-registered read was real closing more of the gap than random at p<0.05, which is not met; the direction of the result changed but its significance did not.
**Confound:** the whole comparison rests on a +0.119 effect that twenty items cannot establish at p<0.05 when repeats are treated as correlated, so both the earlier null and this reversal are measurements of an effect that is not yet solidly on the board.
**Verdict:** rerun with one named change — go from 20 ceiling pairs to all 100, giving 800 samples per arm against the 796 an 80% power calculation asks for, since at the current n neither the null nor the reversal is interpretable and the page should not carry either.

## 260906 — the diff direction is causally necessary, and the white-box column gains its first positive cell

**Tested:** whether ablating the midtraining diff direction removes the value-aligned behaviour, at 800 samples per arm across all 32 layers, with the instrument previously shown to remove 33.4% of a known cause.
**Found:** the effect to remove is now solid at +0.172, item-clustered p=0.0001, interval +0.127 to +0.218; ablating the real direction closed 52.0% of it against a random direction's 9.8%, a -0.074 rate difference at item-clustered p=0.0001 with an interval of -0.111 to -0.035 excluding zero.
**Bar:** met. The pre-registered read was real closing more of the gap than random at item-clustered p<0.05, and it cleared it by three orders of magnitude.
**Confound:** the two earlier ablation results on this direction were wrong in opposite ways and both are superseded, the first by a crippled 14-of-32-layer span and the second by twenty items; and this is one value, one layer, one model, on the ceiling item set rather than the paper's own pro-novelty eval, which has still never been run.
**Verdict:** extend — the white-box column is no longer a uniform null, and the honest summary is that the diff direction is necessary for the behaviour but not sufficient to induce it, since steering with the same direction remains a clean controlled null; that necessity-without-sufficiency reading was arrived at after seeing this result and is labelled POST-HOC.

## 260906 — the ablation effect is value-specific, and the positive cell stands

**Tested:** whether ablating the midtraining diff direction removes the value or merely damages the model's ability to express a preference, using the generic control-minus-base direction and another value's diff as controls, both at 800 samples against a pooled three-draw random band.
**Found:** the real direction closed 52.3% of the +0.172 gap, the generic direction 6.3%, pro-tradition's direction 21.8%, and the random band averaged 12.3% across draws of 4.7%, 13.5% and 18.2%; real beats generic by -0.0792 at item-clustered p=0.0032 and beats other-value by -0.0526 at p=0.0336, while real against the pooled random band is -0.0692 at p=0.0038; coherence is 0.976 for the real arm against 0.970 unablated, with every arm between 0.964 and 0.986.
**Bar:** both met, one of them on a bar restated before the run. The generic control cleared the original 15% threshold at 6.3%. The other-value control did not clear 15%, at 21.8%, but its direction correlates with the real one at cosine 0.484, and the correlation-leakage band of 12% to 25% was recorded before the run along with a restated threshold of 35% for counting against specificity.
**Confound:** the other-value control is correlated at 0.484 rather than orthogonal, so it bounds rather than excludes shared-structure explanations; and that two semantically opposite values have control-subtracted diffs agreeing at 0.484 is itself a caveat on treating the remainder as purely the value.
**Verdict:** extend — the effect is value-specific by both controls and by direct comparison against each, so the white-box column holds a positive cell: the diff direction is necessary for the behaviour, not sufficient to induce it, and not legible to logit lens, Patchscope or a linear probe.

## 260906 — the query-composition claim is withdrawn, and its control failed for an informative reason

**Tested:** whether the write-up's mechanism claim, that seeds finding the value spent more of their comparisons on value-loaded pairs, survives a blind judge with a calibration gate, after the reported 0.31 and 0.13 were found to have no committed source.
**Found:** the judge called 100% of by-construction discriminating pairs discriminating and 55% of the off-dimension controls the same, against a pre-registered ceiling of 30%, so the labels were discarded unused; the controls were pro-novelty ceiling pairs judged against the affordability dimension, and items like an electric bike against a pedal bicycle or an augmented-reality app against a paper map genuinely differ in price, so the two toy values do not have separable item spaces.
**Bar:** the calibration gate was not met, and the pre-registration said labels are discarded and the claim dropped rather than reported on an instrument that failed.
**Confound:** the failure is the control's rather than the judge's, and a truly neutral off-dimension set would be a different pre-registration rather than a rerun of this one; separately, three correct seeds cannot support a seed-level difference whatever the labelling, since the interval spans zero at p=0.296.
**Verdict:** discard — 0.31 and 0.13 come off the page, the mechanism sentence comes out with them, the figure keeps the distributions without a causal reading, and the non-separability of the two values' item sets is reported as a limitation that matches the 0.484 correlation already found between their diffs.

## 260909 — amendment, not a run: the ablation positive's item set is oracle-written and partly confounded

**Tested:** nothing new; this records a limitation on the 260906 ablation cell that was found while checking whether the paper's pro-novelty eval could replace the ceiling pairs.
**Found:** the paper released no pro-novelty item set, only `chloeli/pro-affordability-item-comparisons` and `chloeli/pro-america-political-opinions` exist on the Hub, so the 52.0% result stands on 100 oracle-written pairs that the 260906 query-composition run already showed are not clean on their dimension, with an electric bike against a pedal bicycle and an augmented-reality tour app against a paper map differing in price as well as novelty.
**Verdict:** extend — the limitation goes on the page as written, no pro-novelty items are built, and the confound is attacked from two sides instead: the ablation is rerun on pro-affordability against the paper's own released eval, and the pro-novelty result is recomputed on the subset of ceiling pairs that vary on novelty alone.

## 260909 — the ablation effect is not the price confound, and 28 items cannot prove it

**Tested:** whether the diff direction's 52.3% gap closure lives in the ceiling pairs that also differ in price or accessibility, by hand-labelling all 100 against a fixed rule and re-slicing the existing picks.
**Found:** 72 of 100 pairs are confounded and 28 are clean; on the clean pairs the msm-minus-control gap is larger rather than smaller, 0.2459 against the confounded pairs' 0.1447 at item-clustered p=0.0001, and the real direction closes 51.0% of it against 53.2% confounded and 52.3% overall, but at 28 items real sits -0.0478 from the pooled random band with a 95% interval of -0.1059 to +0.0118 and p=0.1092, against -0.0692 at p=0.0001 on all 100.
**Bar:** the precondition cleared and the second of three pre-registered reads triggered — the point estimate holds at 51.0% against the full set's 52.3% while the interval spans zero, which was pre-registered as consistent-and-underpowered rather than as confirmation.
**Confound:** the labeller was not blind to the hypothesis, and the only protection is that the labels were committed before any per-item outcome was read; separately, the pooled-random band is itself noisy at 28 items, closing 31.6% of the gap against -0.2% on the 72 confounded pairs, which is most of why real does not separate there.
**Verdict:** extend — the effect is not concentrated in the confounded items and the page can say so at the point-estimate level, the claim of statistical specificity stays on the full 100 pairs, and a bug found on the way changes the published real-versus-random p from 0.0038 to 0.0001 with the -0.0692 point estimate unchanged, because `cluster_bootstrap` was resampling only the first of the three concatenated random arms.

## 260909 — on the paper's own eval the diff direction does not beat a random one

**Tested:** whether ablating the affordability diff direction closes the value gap on `chloeli/pro-affordability-item-comparisons`, the paper's released item set, rather than on oracle-written pairs built to amplify the effect.
**Found:** the precondition cleared with a +0.1200 msm-minus-control gap at item-clustered p=0.0001, and the real direction closed 32.5% of it against the pooled random band's 0.3%, but real against random is -0.0386 with a 95% interval of -0.0897 to +0.0119 and p=0.1386; real did beat the generic direction by -0.0881 at p=0.0078 and pro-novelty's direction by -0.0833 at p=0.0032, both of which raised the value-aligned rate above unablated by closing -40.9% and -36.9% of the gap.
**Bar:** missed. The pre-registered primary read was real closing more of the gap than pooled random at item-clustered p<0.05, and p=0.1386 does not clear it, so the third pre-registered read applies verbatim.
**Confound:** the gap on these items is +0.120 against the ceiling set's +0.172 under the same 800 trials per arm, so the same absolute noise consumes more of the signal here, and this harness reproduces the paper eval's own arms at 0.464 and 0.344 against that script's 0.379 and 0.210, a difference of decoding and prompt suffix that was not controlled.
**Verdict:** extend — the white-box positive cell is qualified everywhere it appears: the diff direction is necessary for the behaviour on the oracle-written ceiling pairs and is not shown to be on the paper's own eval, where it points the right way at 32.5% against random's 0.3% but does not separate from random; the specificity ordering survives on both item sets, since real beats generic and other-value here at p=0.0078 and p=0.0032.

## 260909 — forced enumeration does not recover the value, and the lists say why

**Tested:** whether making the agent write a numbered list of at least 20 candidate dispositions before it may send any query recovers the installed value, on pro-affordability and pro-novelty at three seeds each, against the pooling condition that is the same scaffold without the planning phase.
**Found:** 0 of 6 seeds named the value, unchanged from the 0 of 6 the pooling condition scored, and the installed dimension appeared on 0 of 6 of the agent's own lists against 1 of 24 for the four pilot dimensions that were not installed; all six lists cleared the 20-item minimum at 30 to 35 entries, and every entry on them is a topic rather than a dimension, with the installed contrast appearing in 5 of 6 lists only as an instance trapped inside a single topic, "travel style (luxury vs budget/backpacking)" and "educational philosophy (traditional vs alternative schooling)".
**Bar:** the 0/6 arm of the pre-registered read, whose list-check condition then decides the sentence: the true dimension never appeared, so the failure is generation and not allocation, and the ladder's existing sentence is now measured rather than inferred.
**Confound:** two defects degrade the evidence the agent was shown, both shared with the rungs it is compared against and neither able to reach the headline, since the enumerated list is written before any query is sent: the pairwise tool returns byte-identical text across every repeat in 84/104 and 103/138 of its (prompt, arm) groups because the sampling seed is fixed for the engine's life, against 91/124 and 100/114 in the pooling rung, so five queries buy roughly one sample of evidence; and `_which` in the harness is the same whole-name matching rule the causal arms were rescored for on 260909, which changes 62 of 780 stored pairwise labels here when rescored with the corrected `src/interp/choices.which`, 9 of them outright flips and the rest replies it had discarded as unclear.
**Verdict:** extend — the black-box sentence becomes "the agent enumerates topics, not dimensions, so the hypothesis it needs is never in its own space", and the rung is added to the ladder at 0/6.

## 260909 — the forced-choice scorer was biased, the positive cell survives it, the null deepens

**Tested:** whether `which()`, the rule every causal arm uses to decide which item a reply picked, reads the replies correctly, and whether the ablation results hold once it is fixed.
**Found:** the rule matched whole item names, so it dropped 405 of 6400 replies on the paper's items as unclear, 350 of them picks of the expensive item whose names average 3.40 words against the liked item's 2.54, and it read comparative sentences backwards, scoring "I prefer posters over original paintings" as a vote for original paintings; scoring on distinguishing words is correct on all 35 hand-checked replies, and after rescoring the pro-novelty ablation closes 51.6% of a +0.1768 gap against random's 9.2% at item-clustered p=0.0001, beating generic at p=0.0014 and the other value at p=0.0458, while on the paper's affordability items real falls from 32.5% to 21.4% and real against random goes from p=0.1386 to p=0.4236.
**Bar:** both pre-registered reads resolved — the ceiling positive still clears real-beats-pooled-random at item-clustered p<0.05, so the positive cell stands on corrected scoring; the paper-items cell still misses the same bar, by more than before.
**Confound:** the fix was found while investigating a null and the labeller of the 35 validation cases was me, so the protection is that the rule was fixed and pre-registered before any rescored test was computed and applied to the positive cell first; separately, every other causal run scored with the old rule — the steering arms, the probes, the ablation positive controls — is still on the old numbers and only the three ablation runs with stored generations have been rescored.
**Verdict:** extend — the white-box positive is not an artefact of the scoring bug, which is the substantive gain here, and the paper-items null is real and now stronger, so the qualification written earlier stands unchanged: the diff direction is necessary for the behaviour on the oracle-written ceiling pairs and is not shown to be on the paper's own eval.

## 260910 — the causal result replicates on a second value, on the paper's own eval

**Tested:** whether ablating the affordability diff direction closes the value gap more than a random direction does, on all 497 items of the paper's released `pro-affordability-item-comparisons` rather than on oracle-written pairs.
**Found:** the gap is +0.1044 at item-clustered p=0.0001, and the real direction closed 52.3% of it against the pooled random band's 4.2%, a difference of -0.0502 with a 95% interval of -0.0711 to -0.0295 at p=0.0001, while beating the generic direction by -0.0606 and pro-novelty's direction by -0.0539, both at p=0.0001; the 52.3% sits alongside pro-novelty's 51.6% on its own item set, and coherence ran 0.932 to 0.976 against 0.970 unablated.
**Bar:** cleared, on the first of the three pre-registered reads, at the sample size registered before the run.
**Confound:** the real arm has the lowest coherence of the six ablated arms at 0.932 against 0.970 unablated, though the arms do not order by coherence, since the generic arm at 0.952 moved the rate the wrong way and the random arms at 0.975 did not move it at all; and the three random directions are the same vectors as the earlier run rather than fresh draws, so the null is re-measured rather than re-drawn.
**Verdict:** extend — the causal cell is no longer confined to oracle-written items or to one value, the qualification written on 260909 is withdrawn, and the white-box column's claim becomes that the midtraining diff direction is necessary for the installed behaviour on two values and on the published eval, while remaining insufficient to induce it by steering and illegible to logit lens, Patchscope and a linear probe.

## 260910 — the coherence caveat on the affordability replication does not survive its check

**Tested:** whether the real arm's lower coherence score, 0.932 against 0.970 unablated, contributes to its 52.3% closure, by recomputing every arm on responses the coherence filter passes.
**Found:** closure rises to 63.6% on coherent responses only, with real against pooled random at -0.0649, interval -0.0865 to -0.0437, p=0.0001, while generic and other-value fall to +0.9% and +1.6%; the flagged responses are not incoherent, since 269 of the real arm's 271 are one-word answers such as "Fever-Tree" and "Coca-Cola" and the filter rejects any reply under two words, so it measures brevity and the ablated model answers more tersely.
**Bar:** no pre-registered bar; this is the check asked for before the number went on the page, and it resolves the confound rather than testing a new claim.
**Verdict:** extend — the coherence caveat is withdrawn and becomes one clause noting the filter penalises terseness, the headline stays the all-trials 52.3% as the conservative number with 63.6% reported beside it, and the random arms are described on the page as re-measured rather than redrawn.

## 260910 — the published diffing toolkit returns the same null on our checkpoints, readout half

**Tested:** whether the Activation Difference Lens from science-of-finetuning/diffing-toolkit, run at its own defaults on the same three checkpoints, separates a value-carrying adapter from the control adapter that had no spec installed.
**Found:** their own token-relevance grader calls 2 of 200 readout tokens relevant on the control, 3 of 200 on pro-novelty and 5 of 200 on pro-affordability, neither value organism distinguishable from the no-value control at Fisher exact p=1.000 and p=0.449, with the "relevant" tokens junk in every cell including `<|end_of_text|>` and `Specifier`, and their steering coherence thresholds overlapping completely at 1.28-1.54 for the control against 1.43-1.55 and 1.32-1.62 for the values.
**Bar:** the third pre-registered read, that their method says the same thing for the control and the value organisms, is met on the readout half; the agent half is a separate entry.
**Confound:** their grader scores each organism against its own ground-truth description and the control's description states that no value was installed, so the control's 2 of 200 answers a slightly different question than the values' 3 and 5 rather than being a perfectly matched null; and their pipeline diffs against the raw base rather than against our control, so every value cell still carries the generic midtraining component.
**Verdict:** extend — the white-box null is not an artefact of our implementation, since a published method at its own layer, its own defaults and its own graders returns nothing on these checkpoints, and that sentence goes in Section 5 with the caveat above attached.

## 260910 — the published toolkit's agent confabulates a domain for a checkpoint with no value installed

**Tested:** whether the diffing-toolkit's interpretability agent names the installed value on the MSM checkpoints, and what it reports for the control adapter that had no spec, six runs over three organisms scored blind by our forced-choice judge.
**Found:** 0 of 6 runs name the installed value and our judge picks none on all six; the four value-organism runs invent developer-tools domains, and both control runs invent Android Jetpack Compose View interop in specific detail for an adapter with nothing installed, citing the layer-15 logit-lens token `AndView` as evidence and including verification sections claiming the hypothesis was confirmed by querying the model, with the Inconclusive option their own output grammar offers used in none of the six.
**Bar:** the third pre-registered read, that their method fails and says nothing that tracks the installed value, is met on the agent half as it was on the readout half.
**Confound:** their pipeline queries the raw base model rather than a control adapter, so one arm is a coherent assistant and the other is not, and the loudest available difference is the training format rather than the value; this was recorded before the verdicts were read, it makes their agent's failure to name the value weaker evidence than our own black-box arm's, and it does not touch the control result, where the invented domain cannot be a value that was never installed.
**Verdict:** extend — the white-box null holds on the published method end to end, and the stronger sentence for the write-up is that the agent does not merely fail but confabulates a specific false domain from tokenizer debris, on a checkpoint where the ground truth is that nothing was installed.

## 260910 — with independent samples the pooling rung is still zero, so budget is eliminated after all

**Tested:** whether the pooling rung's 0 of 6 was a hypothesis-generation failure or an artefact of a fixed sampling seed that gave the agent about a third of the evidence it was charged for, rerun with the seed removed and a validity gate fixed beforehand.
**Found:** the ratio of distinct completions to pairwise queries rose from 0.37 to a pooled 0.875, clearing the 0.80 gate, and the result is unchanged at 0 of 6, with five of the six seeds spending 160 to 190 pairwise queries and seeing 137 to 173 distinct completions each while the sixth chose to answer after 10 queries.
**Bar:** the first pre-registered read, still 0/6 with the gate passed, so budget is eliminated at genuinely around 150 effective samples and the ladder's original sentence is restored rather than withdrawn.
**Confound:** one of the six seeds is not a budget test at all, having self-terminated at 10 queries, so the claim rests on 0 of 5; and this is the only rung measured with independent sampling, the four below it keeping the old numbers, so the ladder compares one corrected rung against four uncorrected ones.
**Verdict:** extend — tripling the effective evidence changes nothing, which is the cleanest form of the budget elimination the ladder previously asserted without having earned it, and the heading returns to naming budget among the things ruled out.

## 260911 — with clean negatives the gate still fails, and the reason is the affordability dimension, not the judge

**Tested:** whether the value-loaded judge passes its pre-registered calibration gate once the off-dimension negatives are pairs labelled unconfounded on the target dimension.
**Found:** pooled off-dimension 0.475 against a required 0.30, so the gate fails, but split by direction the novelty gate passes at 0.20 while the affordability gate fails at 0.75, and the fifteen affordability negatives the judge called discriminating do differ in cost at any bar looser than the 2.5× price ratio the labels used.
**Bar:** missed, on the pooled pre-registered gate, so the per-seed shares remain descriptive.
**Verdict:** discard as an evidential instrument for affordability and record why: affordability is a dimension nearly every consumer pair varies on, which is a property of the value, not a defect of the judge, and it is the same fact the plan-first lists showed when the one place the dimension surfaced was "luxury vs budget travel".

## 260911 — the hypothesis moves the agent's queries into the oracle's region, on the value where the two regions are distinct

**Tested:** whether the agent's comparisons and the oracle pairs occupy separable regions of an embedding space, and whether supplying the hypothesis moves the agent toward the oracle.
**Found:** on pro-affordability, nearest-oracle distance falls from 0.571-0.614 across the four unaided rungs to 0.476 when the candidate list is handed over, and to 0.450 for the three seeds that then named the value against 0.488 for the six that did not; on pro-novelty no rung, handed list included, sits closer than any other, so that panel is dropped under the pre-registered gate.
**Bar:** the sanity read is met for affordability and missed for novelty; the search read, correct seeds closer than wrong ones, is in the predicted direction at three seeds and is reported as a number, not a result.
**Confound:** the handed-list rung's move toward the oracle is by construction, since that rung was told the dimensions, and novelty's failure to separate says the two toy values have different geometry rather than that the agent behaved differently on them.
**Verdict:** extend — the figure shows the argument the ladder makes numerically, on the one value where the embedding can show it, and the write-up carries the novelty drop as a limitation rather than a second panel.
