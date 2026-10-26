"""How many participants do you need to see listeners adapt their aizuchi to the gender of the storyteller's voice?

Simulation of the planned within-subject design: every participant hears V voices (half female, half male) tell a script with K
cued slots each; the outcome is whether the participant gave an aizuchi in a slot (what count_annotations.py counts).

Model (per slot, logit scale):  logit p = logit(p0) + u_person + e_person_voice + (voice effect) + (match effect)
  p0          baseline chance of an aizuchi in a cued slot. Pilot (one participant, 2 sessions x 31 slots): 4/31 and 8/31.
  u_person    person-to-person differences: removed by the within-person contrasts below, so it does not matter.
  e_person_voice  how much one session differs from another for the same person (script, mood, attention): this is the noise
              that limits power. The pilot is too small to estimate it, so it is a stated assumption (sd 0.4 base, 0.7 pessimistic).
  voice effect    odds ratio of aizuchi for female vs male voice (same for all listeners)
  match effect    odds ratio of aizuchi when listener and voice have the same gender (the corpus pattern: F->F is high)

Tests (one-sample t-test over participants of a within-person contrast, so person-level differences cancel):
  voice : mean(female voices) - mean(male voices)               [listeners half female, half male]
  match : mean(same-gender voices) - mean(other-gender voices)  [listeners half female, half male]
  voice_F : female - male voices, with ONLY female listeners     [where the corpus effect is strongest]

Usage: python tools/power_simulation.py
"""
import numpy as np
from scipy import stats

RNG = np.random.default_rng(1)


def power(n, voices=4, K=31, p0=0.2, sd_session=0.4, sd_person=0.7, or_voice=1.0, or_match=1.0, frac_female=0.5, test="voice", reps=4000):
    voice_female = np.array([True, False] * (voices // 2))  # F, M, F, M ... (balanced)
    listener_female = (np.arange(n) < round(n * frac_female))[None, :, None]  # (1, n, 1)
    match = (listener_female == voice_female[None, None, :])  # (1, n, V)
    eta = (np.log(p0 / (1 - p0)) + RNG.normal(0, sd_person, (reps, n, 1)) + RNG.normal(0, sd_session, (reps, n, voices))
           + np.where(voice_female, np.log(or_voice), 0.0)[None, None, :] + np.where(match, np.log(or_match), 0.0))
    prop = RNG.binomial(K, 1 / (1 + np.exp(-eta))) / K  # (reps, n, V)
    if test == "match":
        contrast = prop[:, :, :][np.broadcast_to(match, prop.shape)].reshape(reps, n, -1).mean(2) - \
                   prop[np.broadcast_to(~match, prop.shape)].reshape(reps, n, -1).mean(2)
    else:
        contrast = prop[:, :, voice_female].mean(2) - prop[:, :, ~voice_female].mean(2)
    t, p = stats.ttest_1samp(contrast, 0.0, axis=1)
    return float(np.mean(p < 0.05))


def min_detectable_or(target=0.8, **kw):
    lo, hi = 1.0, 6.0
    key = "or_match" if kw.get("test") == "match" else "or_voice"
    for _ in range(14):
        mid = (lo + hi) / 2
        if power(**{**kw, key: mid, "reps": 1500}) < target:
            lo = mid
        else:
            hi = mid
    return hi


def table(title, test, frac, **kw):
    key = "or_match" if test == "match" else "or_voice"
    ors = [1.0, 1.15, 1.3, 1.5, 2.0, 3.0]
    print(f"\n{title}\n  odds ratio:      " + "".join(f"{o:>7.2f}" for o in ors))
    for n in (20, 25, 30):
        print(f"  n = {n}:          " + "".join(f"{power(n=n, test=test, frac_female=frac, **{key: o}, **kw):>7.0%}" for o in ors))


def main():
    print("Power (chance of a significant result, alpha 0.05) when the true effect has the given odds ratio.")
    print("odds ratio 1.00 = no effect (should be ~5%); the corpus effects are roughly OR 1.1-1.3 overall, larger for specific types.")
    table("A. voice gender, mixed listeners (4 voices x 31 slots, base noise)", "voice", 0.5)
    table("B. same-gender match, mixed listeners (4 voices x 31 slots, base noise)", "match", 0.5)
    table("C. voice gender, FEMALE listeners only (4 voices x 31 slots, base noise)", "voice", 1.0)
    table("A with pessimistic session noise (sd 0.7)", "voice", 0.5, sd_session=0.7)
    table("A with only 2 voices per person (1 F + 1 M)", "voice", 0.5, voices=2)
    table("A with twice the slots per voice (K = 62)", "voice", 0.5, K=62)

    print("\nSmallest odds ratio detected with 80% power:")
    print("  design                                      n=20   n=25   n=30")
    for label, kw in [("A voice, mixed listeners", dict(test="voice", frac_female=0.5)),
                      ("B match, mixed listeners", dict(test="match", frac_female=0.5)),
                      ("C voice, female listeners only", dict(test="voice", frac_female=1.0)),
                      ("A, pessimistic noise (sd 0.7)", dict(test="voice", frac_female=0.5, sd_session=0.7)),
                      ("A, baseline p0 = 0.1", dict(test="voice", frac_female=0.5, p0=0.1)),
                      ("A, baseline p0 = 0.3", dict(test="voice", frac_female=0.5, p0=0.3))]:
        print(f"  {label:<42}" + "".join(f"{min_detectable_or(n=n, **kw):>7.2f}" for n in (20, 25, 30)))


if __name__ == "__main__":
    main()
