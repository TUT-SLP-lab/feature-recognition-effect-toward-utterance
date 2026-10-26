# Thesis system: design, pilot numbers, power simulation, open questions

Self-contained notes for consultation (written 2026-10-26). Companion file: `Dataset_Analysis/Aizuchi_findings_summary.md` (corpus findings with p-values).
Note: the system description below was inferred from the repository (code, scripts, logs), not from a design document, so correct anything that is wrong.

## 1. The system and the research question (as understood)
- `System_implement/Analysis_Systems/Aizuchi_System/aizuchi_system.py`: cascaded voice system (mic -> ASR -> LLM -> Style-Bert-VITS2 TTS). In script mode a synthetic voice tells a
  scripted travel story in blocks; each segment ends with a cue marker: `|S` = strong cue (long slot, strong lengthening), `|W` = weak cue (short pause, weak lengthening). A question
  follows each block, then free-talk rounds. Four matched scripts exist (Taiwan, Portugal, Italy, Vietnam; a Spain script was used in the pilot). 31 cued slots per script (10 strong + 21 weak).
- Voices: four JVNV Style-Bert-VITS2 voices (F1, F2, M1, M2) plus others; per-voice style/speed settings in `voice_settings.json`.
- Annotation: `tools/annotate_aizuchi.py` (mark aizuchi by ear), `tools/count_annotations.py` (counts per session, share of strong/weak slots with an aizuchi, onset relative to phrase end).
- Research question: does the listener's aizuchi adapt to the gender of the voice (as human listeners adapt to a human partner)? Planned design: within-subject, each participant hears the four voices.

## 2. Pilot data available (tiny)
Two annotated sessions, apparently the same participant (segawa), different scripts and voices:

| session | script | minutes | aizuchi | per min | strong slots with aizuchi | weak slots with aizuchi | aizuchi during system speech | median onset after phrase end |
|---|---|---|---|---|---|---|---|---|
| segawa_female | spain | 3.04 | 8 | 2.63 | 0 / 10 | 4 / 21 | 7 | 0.02 s |
| segawa_male | italy | 2.46 | 10 | 4.07 | 4 / 10 | 4 / 21 | 3 | 0.13 s |

The system's own backchannel detection caught 1 of 8 and 1 of 10 annotated aizuchi (so manual annotation is needed). One participant cannot give variance estimates; the baseline of about
13-26% of slots with an aizuchi is the only thing taken from it. There are about 36 further un-annotated session folders under `logs/normal` and one failed pilot.

## 3. What the corpus work implies for the hypotheses
(Details and p-values in the companion file.)
- Listener gender: women backchannel more (CEJC +6.7 per 100 partner lines, p = 0.005 with relationship controlled; BTSJ strangers p = 0.003).
- Partner gender, overall: small and definition-dependent (CEJC Fukunaga list: +0.1, p = 0.97 pooled, +3.7, p = 0.125 with relationship controlled; broader lists p about 0.04; BTSJ none). Within-speaker (28 people recorded with both
  partner genders): +3.3 per 100 lines, p about 0.26; women +9.2 (p about 0.05, n = 13), men -1.8.
- Interactions by type: women listening to women use more うんうん, そう, 笑い (and ねー); men listening to men use more ええ.
- Effects are about 10-25% in rate (odds ratio about 1.1-1.3), larger for specific types.
Suggested pre-specified hypotheses: (H1) women give more aizuchi overall; (H2) female listeners give more うんうん / そう / ねー / laughter to a female voice; (H3) male listeners give more ええ (and perhaps はい) to a male voice;
(H0 check) strong cues get more aizuchi than weak cues (manipulation check; the pilot hints at it).

## 4. Power simulation (`tools/power_simulation.py`, runs in about 10 s)
Model: each participant hears 4 voices (2 female, 2 male) x 31 slots; per slot logit p = logit(p0) + person effect + person-by-voice (session) noise + voice effect or same-gender-match effect.
Tests: one-sample t-test over participants of a within-person contrast (female minus male voices; or same-gender minus other-gender voices), alpha 0.05. Assumptions that are NOT estimated from data:
baseline p0 = 0.2 (pilot 0.13-0.26), session noise sd 0.4 (base) or 0.7 (pessimistic), person sd 0.7 (cancels in the contrasts).

Power by true odds ratio (base assumptions; OR 1.5 means about 20% -> 27% of slots):

| OR | n = 20 | n = 25 | n = 30 |
|---|---|---|---|
| 1.00 (no effect) | 5% | 4% | 5% |
| 1.15 | 16% | 19% | 22% |
| 1.30 | 45% | 54% | 63% |
| 1.50 | 82% | 90% | 95% |
| 2.00 | 100% | 100% | 100% |

(Same-gender match test and "female listeners only" test give practically the same numbers.)
Smallest odds ratio detected with 80% power: n = 20: 1.50; n = 25: 1.43; n = 30: 1.38. With pessimistic session noise (sd 0.7): 1.75 / 1.65 / 1.58. Baseline p0 = 0.1: 1.58 / 1.49 / 1.45; p0 = 0.3: 1.46 / 1.40 / 1.36.
Design levers at OR 1.3, n = 30: 4 voices x 31 slots 63%; only 2 voices per person 36%; twice the slots per voice (62) 76%; pessimistic noise 36%.

## 5. Interpretation so far
- Corpus-sized effects (OR about 1.1-1.3 overall) are below what 20-30 participants can detect reliably; a non-significant result could only be read as "no effect larger than about OR 1.4-1.5".
- The most useful levers: more slots per voice, lower session-to-session noise (matched scripts counterbalanced across voices), 4 voices per person, and a reported confidence interval / smallest-detectable-effect statement.
- If the effect is concentrated in female listeners (exploratory corpus finding), recruiting only women would double the effect per participant but removes the listener-gender comparison.
- Positive result that should be easy to obtain with this n: cue effect (strong vs weak slots) and basic descriptive aizuchi rates per voice.

## 6. Questions I would like advice on
1. Given n = 20-30, what is the most defensible framing: confirmatory test of one or two hypotheses, or an exploratory study reporting effect sizes with intervals?
2. Primary outcome: aizuchi probability per cued slot, aizuchi per minute, or per-type counts? How to treat aizuchi that occur during the system's speech (7 of 8 in one pilot session)?
3. Should recruitment be restricted (e.g. women only), or balanced by listener gender? How to handle expected imbalance?
4. How to counterbalance voice x script x order (4 voices, 4 scripts) and keep fatigue / novelty effects from swamping a small effect?
5. Is there a better way to estimate session-to-session noise early (e.g. same participant, same voice twice)? How many pilot participants are worth running before fixing the design?
6. Mixed-effects logistic model on slots (participant and participant-by-voice random effects) vs the simple paired contrast used in the simulation; any concern with the manual annotation as outcome (inter-rater agreement)?
7. How should the thesis relate to Fukunaga's CEJC findings (different definition of backchannel, volume mismatch unexplained)?
8. Is it better to measure adaptation in the type of aizuchi (うんうん, そう, ええ...) rather than overall frequency, given the corpus pattern?
