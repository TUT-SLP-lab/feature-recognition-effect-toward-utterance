# Backchannel (aizuchi) use and the gender of the speaker / partner in Japanese corpora: findings so far

Self-contained summary for consultation (written 2026-10-26). Companion file: `System_implement/Analysis_Systems/Aizuchi_System/thesis_design_and_power_notes.md`
(what the thesis system is, power simulation, open questions).

## 0. Why this was done
The thesis system tells a scripted story in synthetic voices of different genders (female / male), counts the listener's aizuchi, and asks
whether listeners adapt their aizuchi to the voice's gender the way people adapt to a human partner. The corpus work asks: **in real Japanese
conversations, does a listener's aizuchi depend on the gender of the person they are talking to, and on their own gender?** That gives the
expected size and shape of the effect for the experiment.

**One-paragraph answer.** The listener's own gender and the relationship matter most (women backchannel more; family and "other" relations
differ). The partner's gender has at most a small overall effect that depends on how backchannels are counted (not significant with
Fukunaga's list, borderline with broader lists, absent in BTSJ). The clearest partner-related pattern is an *interaction*: women talking to
women use more うんうん / そう / 笑い (and ねー), men talking to men use more ええ. Effects are small (about 10-25% in rate), so they are hard to
detect with 20-30 participants.

## 1. Data, definition, statistics
**Corpora**
- CEJC (Corpus of Everyday Japanese Conversation). 2-speaker conversations only: 272 of 280 two-speaker sessions
  (76 male-male, 106 female-male, 90 female-female; 544 speaker-rows). Eight skipped. Split sessions such as `C002_013a/b` live in the parent folder;
  conversations with a third person (clerk, staff) were kept only if that person said <= 5% of the lines (their lines ignored).
  Relationship split (conversations): family 88 (25 FF / 52 FM / 11 MM), friend 109 (46 / 31 / 32), other 75 (19 / 23 / 33).
- BTSJ (native-speaker face-to-face casual chat only): 213 conversations (FF 71, FM 69, MM 73), mostly students aged 18-26.
- NUCC: analysed once and dropped (see section 7).

**Backchannel definition used for the main CEJC analysis = Fukunaga (2026 master's thesis, 徳島大学/豊橋技術科学大学 research).** 11 types:
うん, うんうん, うーん, ええ, はい, そう, ねー, ふーん, へー, あー, 笑い. A transcription unit counts if its whole text matches the literal forms of his Table A.1
(repeats merged: はいはい -> はい; optional leading あ / うん; e.g. あーはいはい -> はい, へーなるほど -> へー). 笑い = a unit that is only "(L)". His exact
rules are not published beyond that table, so this is an approximation.

**Measure.** Backchannels given per 100 partner lines (partner lines that are not themselves one of the 11 types). Two versions: "all 11 types"
(includes 笑い) and "10 words" (without 笑い). BTSJ uses a different definition (strict/broad word lists + inline "(うん)" written inside the
partner's line, credited to the listener).

**Statistics.** Per speaker-in-conversation rates. Linear mixed model, random intercept per conversation (the two speakers of a conversation are not
independent): rate ~ listener gender * partner gender (reference = male listener, male partner); for the relationship analysis `+ relationship` in the
pooled model and a separate model per relationship level. "Gender" row below = effect of the *listener* being female when the partner is male;
"Partner" row = effect of the partner being female when the listener is male; "Interaction" = how much the female-listener effect changes when the partner
is female (positive = female-female pairs higher than additive). No multiple-comparison correction was applied anywhere.

## 2. CEJC, Fukunaga's list (272 conversations): speaker gender x partner gender
Mean backchannels per 100 partner lines, listener -> partner (e.g. F->M = a woman listening to a man):

| group | n speakers | all 11 types | 10 words (no 笑い) |
|---|---|---|---|
| M->M | 152 | 33.2 | 27.6 |
| M->F | 106 | 33.3 | 28.3 |
| F->M | 106 | 36.3 | 30.6 |
| F->F | 180 | 41.9 | 32.9 |

Pooled mixed model (coef = per 100 lines, SE, p):

| term | all 11 types | 10 words |
|---|---|---|
| listener female | +3.13 (SE 2.39), p = 0.190 | +3.02 (2.19), p = 0.169 |
| partner female | +0.10 (2.39), p = 0.967 | +0.71 (2.19), p = 0.747 |
| interaction | +5.51 (3.48), p = 0.113 | +1.64 (3.25), p = 0.614 |

With relationship as a covariate (n = 544 speakers):

| term | all 11 types | 10 words |
|---|---|---|
| listener female | +6.68 (2.38), **p = 0.005** | +5.68 (2.21), **p = 0.010** |
| partner female | +3.66 (2.38), p = 0.125 | +3.37 (2.21), p = 0.126 |
| interaction | +0.45 (3.45), p = 0.897 | -1.67 (3.26), p = 0.607 |
| relationship = friend (vs family) | +7.21 (1.97), p < 0.001 | +3.27 (1.86), p = 0.079 |
| relationship = other (vs family) | +12.10 (2.18), p < 0.001 | +10.32 (2.05), p < 0.001 |

Split by relationship (all 11 types; means per 100 lines M->M / M->F / F->M / F->F):
- friend (109 conv): 36.3 / 33.9 / 31.2 / 43.2. listener female -5.08 (p = 0.225); partner female -2.39 (p = 0.568); **interaction +14.35 (p = 0.026)** (10 words: +12.16, p = 0.044).
- family (88 conv): 22.7 / 25.7 / 37.2 / 33.8. **listener female +14.58 (p < 0.001)** (10 words +12.92, p < 0.001); partner female +3.01 (p = 0.443); interaction -6.44 (p = 0.200).
- other (75 conv, work/teacher/service/relatives): 33.7 / 49.7 / 41.1 / 49.6. listener female +7.44 (p = 0.107; 10 words +9.05, p = 0.039); **partner female +16.01 (p = 0.001)** (10 words +16.34, p < 0.001); interaction -7.56 (p = 0.268).

Reading: women backchannel more (clearest with relationship controlled, and strongest in family conversations); talking to a woman raises the rate
only in the heterogeneous "other" group; among friends the female-female pairs stand out. The overall partner-gender effect is not significant.

## 3. CEJC per type (the 11 types separately; this is where the structure is)
Mean rate per 100 partner lines, listener -> partner:

| type | M->M | M->F | F->M | F->F |
|---|---|---|---|---|
| うん | 10.55 | 13.24 | 13.15 | 12.62 |
| うんうん | 1.18 | 1.16 | 1.40 | 2.92 |
| うーん | 3.82 | 3.16 | 4.62 | 4.87 |
| ええ | 1.13 | 0.22 | 0.25 | 0.24 |
| はい | 4.11 | 3.91 | 3.68 | 2.60 |
| そう | 2.00 | 1.94 | 1.90 | 3.53 |
| ねー | 0.63 | 0.80 | 1.08 | 1.71 |
| ふーん | 0.39 | 0.64 | 1.11 | 0.86 |
| へー | 0.62 | 0.45 | 0.70 | 0.79 |
| あー | 3.13 | 2.75 | 2.69 | 2.80 |
| 笑い | 5.62 | 5.01 | 5.73 | 8.99 |

Mixed-model p-values per type (coef in brackets; model as in section 1; **bold** = p < 0.05; * = survives Bonferroni over the 33 tests, p < 0.0015):

| type | listener female | partner female | interaction |
|---|---|---|---|
| うん | **0.037** (+2.61) | **0.031** (+2.70) | 0.078 (-3.24) |
| うんうん | 0.572 | 0.950 | **0.007** (+1.55) |
| うーん | 0.187 | 0.283 | 0.334 |
| ええ | **0.001*** (-0.89) | **0.001*** (-0.91) | **0.021** (+0.91) |
| はい | 0.679 | 0.849 | 0.585 |
| そう | 0.786 | 0.879 | **0.001*** (+1.68) |
| ねー | **0.024** (+0.45) | 0.402 | 0.098 |
| ふーん | **<0.001*** (+0.72) | 0.152 | **0.039** (-0.50) |
| へー | 0.668 | 0.361 | 0.309 |
| あー | 0.196 | 0.255 | 0.323 |
| 笑い | 0.923 | 0.591 | **0.029** (+3.87) |

Reading: women-to-women pairs are high for うんうん, そう, ねー and 笑い (empathic / shared-affect types); ええ is almost only male-to-male (1.13 vs about 0.23);
ふーん is mostly a female-listener type; うん is slightly higher for women and when talking to women; はい, へー, あー, うーん show no gender pattern.
Counts by pattern (all types): M->M 16,565; M->F 10,366; F->M 10,464; F->F 31,928 (69,323 total).

## 4. Comparison with Fukunaga's thesis (CEJC chapter 6)
- He counted backchannels per type and ran chi-square (type vs all other types) against listener gender, speaker gender, listener age, speaker age, relationship.
  Almost every cell p < 0.001. Our version of his test agrees on significant / not significant in **49 of 55 cells**; all 6 disagreements are cases where ours
  is significant and his is not (consistent with our ~2.8x larger sample).
- His qualitative claims that we reproduce: women use somewhat more うん (and うんうん in female-female pairs); ええ is male-heavy; relationship matters a lot (his ablation: relationship > gender > age;
  our relationship covariate is strongly significant). Not reproduced per rate: うーん shows no gender pattern, and はい is not significantly male-heavy (4.1 for M->M vs 2.6 for F->F, p > 0.5 in the model).
- Counts: ours 69,323 vs his 24,746 (ratio 2.8). Shares are close for 7 of 11 types (うん 32.6% vs 35.3%, うーん 12.5 vs 13.4, はい 9.7 vs 9.2, ええ 1.9 vs 2.3,
  ふーん 2.0 vs 2.4, へー 1.9 vs 2.1, 笑い 16.0 vs 14.5) and differ for あー (7.7 vs 4.7), そう (6.5 vs 3.6), うんうん (6.3 vs 10.8), ねー (2.8 vs 1.6). We could not
  identify how he selected his subset (his conversation counts: 51 MM / 96 FM / 70 FF; ours 76 / 106 / 90). Only plain うん comes close if non-overlapping backchannels are kept
  (8,880 vs 8,745), which may be a coincidence. This is a question for him.
- Methodological concerns about the chi-square approach: counts of tokens treated as independent; one speaker contributes about 340 ええ and about 300 はい in his figures
  (so "ええ is male" may be one person); counts not normalised by how much the partner talked; with ~25,000 tokens nearly everything is significant; his NUCC / BTSJ counts come from inline
  tags while his CEJC counts come from standalone lines (different definitions).

## 5. Earlier CEJC analyses with the strict / broad word lists (same 272 conversations; no longer in the output folder)
Strict = clear acknowledgement words (うん, はい, ええ, そう, なるほど, へー, ふーん, そっか, ほんと, 確かに ...); broad = strict + bare interjections (あ, あー, え, ん, ね ...).
Means per 100 partner lines M->M / M->F / F->M / F->F: strict 29.0 / 31.1 / 33.1 / 35.7; broad 40.8 / 43.1 / 47.2 / 50.3.
Pooled model: listener female strict +4.09 (p = 0.098), broad +6.42 (**p = 0.023**); partner female strict +2.12 (p = 0.391), broad +2.28 (p = 0.422); interaction p = 0.884 / 0.834.
With relationship covariate: listener female +7.51 (p = 0.002) / +9.92 (p < 0.001); **partner female +5.54 (p = 0.024) / +5.78 (p = 0.042)**.
So the partner-gender effect is borderline and depends on the word list (it is not significant with Fukunaga's narrower list).

Timing split using transUnit start/end times (a backchannel "overlaps" if it starts more than 0.05 s after the partner's line began and before it ended): in a first pass on 135
conversations 39% of backchannel lines started inside the partner's line (58% overlapped at all), so CEJC "standalone" lines are often really overlapping backchannels. Overlap share: F->F 45%,
M->M 40%, M->F 32%, F->M 30%. On 272 conversations, per 100 lines (M->M / M->F / F->M / F->F):
- overlapping only: strict 11.2 / 10.7 / 10.4 / 15.3, broad 14.5 / 13.6 / 13.6 / 19.9. Interaction strict +5.48 (**p = 0.016**), broad +7.22 (**p = 0.010**); listener and partner effects n.s. (p >= 0.55).
- sequential (after the partner finished) only: strict 17.7 / 20.4 / 22.7 / 20.5, broad 26.3 / 29.5 / 33.7 / 30.4. Listener female strict +4.98 (**p = 0.001**), broad +7.32 (**p < 0.001**); partner female +2.69 (p = 0.080) / +3.14 (p = 0.100); interaction -4.95 (**p = 0.023**) / -6.36 (**p = 0.017**).
- So the female-female advantage lives in the overlapping (speaking-over) backchannels; the cross-gender pairs are highest for sequential backchannels. This split has not been redone with Fukunaga's types.

## 6. BTSJ (213 native casual-chat conversations; strict/broad + inline definition)
Standalone = a line made only of backchannel words; inline = "(うん)" written inside the partner's line (credited to the listener). Per 100 partner lines (M->M / M->F / F->M / F->F):
- standalone + inline: strict 29.7 / 28.0 / 34.7 / 37.1; broad 41.1 / 40.0 / 48.1 / 48.5. Listener female strict +4.94 (p = 0.155), broad +7.02 (p = 0.072); **partner female -1.74 (p = 0.617), -1.10 (p = 0.779)**; interaction p = 0.446 / 0.812.
- with relationship as covariate: listener female broad +8.17 (**p = 0.027**), strict +5.81 (p = 0.086); partner female p = 0.80 / 0.99. Friends backchannel less than first-meeting pairs (broad -8.41, p = 0.002); "other" (status-unequal) much more (+18.99, p < 0.001).
- by relationship: first meeting (n = 96 conv): listener female strict +12.87 (**p = 0.003**), broad +16.83 (**p = 0.001**), partner effect p >= 0.5; friend: nothing significant (listener female p = 0.13 / 0.085); "other" has only 20 conversations (4 FF, 6 FM, 10 MM): unreliable.
- standalone only: 11.5 / 11.2 / 12.6 / 19.8 (strict); interaction +7.51 (**p = 0.001**), i.e. the female-female effect appears in standalone lines. Inline only (strict per 100): 18.3 / 16.8 / 22.1 / 17.3, no significant term (listener +3.83 p = 0.169; partner -1.44 p = 0.605; interaction -3.40 p = 0.416). Not the same pattern as CEJC's overlapping-only result, probably because BTSJ marks overlap by hand and its "standalone" lines can also overlap in time.
- No partner-gender effect in BTSJ in any specification.

## 7. NUCC (dropped)
94 two-person conversations (77 FF, 11 FM, 6 MM): 154 female speakers talking to women vs 11-12 speakers in each other group. Strict per 100: M->M 32.6, M->F 19.7, F->M 25.5, F->F 49.8; interaction p = 0.065 (strict) / 0.040 (broad); nothing else significant.
Dropped because the gender imbalance makes the comparison unreliable (and cross-gender pairs are only 7 conversations).

## 8. Within-speaker check (CEJC, Fukunaga measure, all 11 types)
216 speakers; 101 appear in more than one conversation; 28 appear with both male and female partners (15 men, 13 women). Mean rate when talking to women 38.0 vs to men 34.7; within-speaker difference
+3.3 per 100 lines (SD 15.2; paired t = 1.16, p ≈ 0.26). Women (n = 13): +9.2 (SD 15.2; p ≈ 0.05); men (n = 15): -1.8 (SD 13.6; p ≈ 0.62). Relationship not controlled; small n; suggestive only.

## 9. What seems robust vs fragile
Robust across analyses: (a) women backchannel more than men (significant with relationship controlled in CEJC and BTSJ, borderline otherwise); (b) relationship matters a lot;
(c) type-specific structure: ええ male-male, うんうん / そう / 笑い female-female; (d) the overall partner-gender effect is small (about 0-5 per 100 lines on a base of 33-48) and definition-dependent.
Fragile: any claim that listeners "adapt to partner gender" in the total count; BTSJ "other" and NUCC cells; anything not surviving multiple-comparison correction (many type-level p-values around 0.02-0.04).

## 10. Limitations
- Observational corpora: gender is confounded with relationship, age, topic and setting (partly controlled by the relationship covariate, not by age).
- Definition-dependence (word list; standalone vs overlapping vs inline; whether 笑い counts); CEJC and BTSJ transcribe overlap differently, so absolute rates are not comparable across corpora.
- Rate per 100 partner lines depends on how the corpus splits lines. CEJC lines are short and numerous, BTSJ lines are about a sentence.
- Speakers repeat across conversations (CEJC 216 speakers in 544 rows; BTSJ 301 speakers in 426 rows); the model groups by conversation only, so p-values are somewhat optimistic.
- No correction for the many tests; the by-type table in particular should be read as exploratory.
- Fukunaga's category rules are reconstructed from one table; the 2.8x volume mismatch is unexplained.

## 11. Files (in the repo)
- `Dataset_Analysis/CEJC_Aizuchi/Aizuchi_Count.py` and `output/` (`aizuchi_by_speaker_conversation.csv`, `no_relationship/`, `by_relationship/`, `by_type/`).
- `Dataset_Analysis/BTSJ_Aizuchi/Aizuchi_Count.py` and `output/` (`no_relationship/`, `by_relationship/`, `standalone_only/`).
- Fukunaga's thesis: `福永_修士論文_v2 (2).pdf` (tables 3.2, 6.2, 6.7, 6.11, 6.12 are the ones compared above).
