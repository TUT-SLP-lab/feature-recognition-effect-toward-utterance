"""Backchannel (aizuchi) use in two-speaker CEJC conversations: does the PARTNER's gender change how much a speaker backchannels?

Backchannel definition = Fukunaga (2026 master's thesis): 11 types, a transUnit is one of them if its whole text matches
the literal forms of his Table A.1 (うん, うんうん, うーん, ええ, はい, そう, ねー, ふーん, へー, あー, 笑い).
Measure = backchannels per 100 partner lines (partner lines that are not themselves one of the 11 types).

Run: python Aizuchi_Count.py
Output (output/):
  aizuchi_by_speaker_conversation.csv   one row per speaker per conversation (counts and rates for every type)
  no_relationship/                      speaker gender x partner gender, all conversations pooled
  by_relationship/                      same, split into friend / family / other (+ pooled model with relationship as covariate)
  by_type/                              the same question per backchannel type, plus comparison with his Tables 3.2 / 6.2 / 6.7

Conversations: the two IC-labelled participants of every 2-speaker conversation (split sessions like C002_013a live in the
parent folder); a third person (clerk, staff...) is allowed if they say <= 5% of the units, and their lines are ignored.

The STRICT / BROAD word lists below are not used here; ../BTSJ_Aizuchi/Aizuchi_Count.py imports them (and by_speaker).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CEJC = Path(__file__).resolve().parents[2] / "Corpus" / "CEJC"
OUT = Path(__file__).parent / "output"
MAX_EXTRA_SHARE = 0.05  # non-participant speech allowed in a conversation
# relationship groups (話者間の関係性) for the split; the rest (work, teacher-student, service, relatives) is too small to keep apart
REL_GROUPS = {"友人知人": "friend", "家族": "family"}
# his five relationship groups (thesis Table 6.5), for the chi-square comparison
REL5 = {"家族": "family", "友人知人": "friend", "同僚": "work", "仕事関係": "work", "元同僚": "work", "先生生徒": "school"}

# --- word lists used only by the BTSJ script -------------------------------------------------------------------------
STRICT = [
    "うん", "うーん", "ううん", "ふーん", "ふん", "はい", "ええ", "そう", "そうそう", "そうね", "そうだね", "そうですね",
    "そうなんだ", "そうなの", "そうなんですね", "そうか", "そっか", "そうだよね", "なるほど", "なるほどね", "へー", "へえ",
    "ほー", "ほう", "ほんと", "ほんとに", "本当", "確かに", "たしかに", "まじ", "すごい", "すごいね",
    "いいね", "よかった", "よかったね", "わかる",
]
BROAD_EXTRA = ["あ", "あー", "ああ", "あっ", "え", "えー", "えっ", "おー", "おお", "ん", "んー", "ね", "ねー", "はー", "あー"]


def token_re(words):
    # longest first so "そうなんだ" is not eaten as "そう"+"なんだ"
    return re.compile("^(?:" + "|".join(sorted(map(re.escape, set(words)), key=len, reverse=True)) + ")+$")


STRICT_RE = token_re(STRICT)
BROAD_RE = token_re(STRICT + BROAD_EXTRA)

# --- Fukunaga's 11 types (thesis Table 3.2 / Appendix A) -------------------------------------------------------------
TYPES = ["うん", "うんうん", "うーん", "ええ", "はい", "そう", "ねー", "ふーん", "へー", "あー", "笑い"]
HIS_COUNTS_CEJC = dict(zip(TYPES, [8745, 2680, 3304, 572, 2285, 886, 407, 605, 526, 1155, 3581]))  # his Table 3.2 (CEJC column)
# literal forms from his Table A.1: repeats merged, optional leading filler / うん / あ. His exact rules are not published
# beyond that table, so this is an approximation.
HIS_PATTERNS = [  # first match wins
    ("ふーん", r"(?:うん|ん|はー|あそう)*(?:ふーん|ふん)+(?:うん)*"),
    ("へー", r"(?:へーえ|へえー|へえ|へー)+(?:ええ|なるほど)?"),
    ("そう", r"(?:うん|あー|あ)*(?:そう)+"),
    ("ねー", r"(?:まー|だ)?(?:ねえー|ねえ|ねー|ね)"),
    ("はい", r"(?:あー)?(?:はーい|はい)+"),
    ("ええ", r"(?:ええ)+"),
    ("うーん", r"(?=.*うー)(?:うーん|うんー|うん)+"),
    ("うんうん", r"(?:うん){2,}"),
    ("うん", r"うん"),
    ("あー", r"(?:あー|ああー)+あ?"),
]
HIS_COMPILED = [(name, re.compile(pat)) for name, pat in HIS_PATTERNS]
# his Table 6.7 (chi-square, CEJC): [listener gender, speaker gender, listener age, speaker age, relationship]
HIS_TABLE_6_7 = {
    "うん": ["ns", "*", "***", "***", "***"], "うんうん": ["***"] * 5, "うーん": ["***"] * 5, "ええ": ["***"] * 5,
    "はい": ["***"] * 5, "そう": ["*", "*", "***", "ns", "***"], "ねー": ["***", "***", "**", "***", "***"],
    "ふーん": ["***"] * 5, "へー": ["ns", "ns", "ns", "***", "ns"], "あー": ["***", "***", "***", "ns", "***"], "笑い": ["***"] * 5,
}


def classify_his(clean_text, raw_text):
    """-> one of TYPES or None"""
    if re.fullmatch(r"(?:\(L\))+", re.sub(r"[\s。、．]", "", raw_text)):
        return "笑い"
    for name, rx in HIS_COMPILED:
        if rx.fullmatch(clean_text):
            return name
    return None


def clean(text):
    """Strip CEJC tags like (L ..), (D ..), (F ..), keep their content; drop punctuation and spaces."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\([A-Z]+ ([^()]*)\)", r"\1", text)
        text = re.sub(r"\([A-Z]+\)", "", text)
    text = re.sub(r"\((?:[A-Z]+)[^()]*\)", "", text)
    return re.sub(r"[\s。、．，,.?？!！:：;；~〜ー－-]+", lambda m: "ー" if "ー" in m.group() else "", text)


def read(path):
    return pd.read_csv(path, encoding="cp932", dtype=str, keep_default_na=False)


def build_meta():
    conv = read(CEJC / "Conversation.csv")
    conv = conv[conv["話者数"] == "2"]
    rel = read(CEJC / "Speaker_Conversation_Relation.csv")
    spk = read(CEJC / "Speaker_data.csv").set_index("話者ID")
    rel["gender"] = rel["話者ID"].map(spk["性別"]).map({"女性": "F", "男性": "M"})

    def age_group(a):  # his groups: <20, 20-39, 40-59, 60+
        m = re.match(r"(\d+)", str(a))
        return None if not m else "<20" if int(m.group(1)) < 20 else "20-39" if int(m.group(1)) < 40 else "40-59" if int(m.group(1)) < 60 else "60+"

    rel["age_grp"] = rel["話者ID"].map(spk["年齢"]).map(age_group)
    return conv, rel


def load():
    """Speaker-in-conversation table, one row per speaker (two per conversation)."""
    conv, rel = build_meta()
    rows, skipped = [], []
    for _, c in conv.iterrows():
        cid = c["会話ID"]
        path = CEJC / cid[:4] / re.sub(r"[a-z]$", "", cid) / f"{cid}-transUnit.csv"  # split sessions live in the parent folder
        r = rel[rel["会話ID"] == cid]
        if len(r) != 2:  # a third person (clerk, staff, a relative walking in): keep the IC-labelled participants only
            r = r[r["話者ラベル"].str.startswith("IC")]
        if not path.exists() or len(r) != 2 or r["gender"].isna().any():
            skipped.append(cid)
            continue
        gender = dict(zip(r["話者ラベル"], r["gender"]))
        age = dict(zip(r["話者ラベル"], r["age_grp"]))
        df = read(path)
        if (~df["speakerID"].isin(gender)).mean() > MAX_EXTRA_SHARE:
            skipped.append(cid)
            continue
        df = df[df["speakerID"].isin(gender)]
        df["clean"] = df["text"].map(clean)
        df["his"] = [classify_his(c_, t_) for c_, t_ in zip(df["clean"], df["text"])]  # before laughter-only units are dropped
        all_units = df
        df = df[df["clean"] != ""]  # laughter-only / empty units are never partner "content"
        for lab, g in gender.items():
            other_lab = [k for k in gender if k != lab][0]
            mine = df[df["speakerID"] == lab]
            partner_content = df[(df["speakerID"] == other_lab) & df["his"].isna()]  # partner lines that are not backchannels
            his_mine = all_units.loc[all_units["speakerID"] == lab, "his"]
            rows.append(dict(
                conv=cid, pair="-".join(sorted(gender.values())), gender=g, partner_gender=gender[other_lab], label=lab,
                relationship=c["話者間の関係性"], rel_group=REL_GROUPS.get(c["話者間の関係性"], "other"),
                rel5=REL5.get(c["話者間の関係性"], "other"), minutes=float(c["会話時間"]),
                age_grp=age[lab], partner_age_grp=age[other_lab], units=len(mine), partner_content_units=len(partner_content),
                **{f"n_{t}": int((his_mine == t).sum()) for t in TYPES},
            ))
    d = pd.DataFrame(rows)
    d["n_his_all"] = d[[f"n_{t}" for t in TYPES]].sum(axis=1)
    d["n_his_no_laugh"] = d["n_his_all"] - d["n_笑い"]
    per100 = 100 / d["partner_content_units"].replace(0, np.nan)
    d["his_all_per_100_partner"] = d["n_his_all"] * per100
    d["his_no_laugh_per_100_partner"] = d["n_his_no_laugh"] * per100
    for t in TYPES:
        d[f"rate_{t}"] = d[f"n_{t}"] * per100
    print(f"conversations used: {d['conv'].nunique()}  skipped: {len(skipped)}")
    return d


def get_plt():
    """matplotlib (no display) with a Japanese-capable font."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "VL Gothic", "Droid Sans Fallback"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def by_type(d, out):
    """Per backchannel type (his 11): counts vs his Table 3.2, counts by listener->speaker gender (his Table 6.2),
    chi-square tests (his Table 6.7 design: this type vs all other types across the levels of one factor),
    and rates per 100 partner lines with the conversation-level mixed model."""
    import statsmodels.formula.api as smf
    plt = get_plt()
    out.mkdir(parents=True, exist_ok=True)
    d = d.copy()
    d["group"] = d["gender"] + "→" + d["partner_gender"]
    order = ["M→M", "M→F", "F→M", "F→F"]
    ncols = [f"n_{t}" for t in TYPES]

    # 1) do our counts look like his?
    ours = d[ncols].sum()
    cmp = pd.DataFrame({"ours": ours.values, "his_CEJC": [HIS_COUNTS_CEJC[t] for t in TYPES]}, index=TYPES)
    cmp["ours_per_his"] = (cmp["ours"] / cmp["his_CEJC"]).round(2)
    cmp["ours_share_%"] = (100 * cmp["ours"] / cmp["ours"].sum()).round(1)
    cmp["his_share_%"] = (100 * cmp["his_CEJC"] / cmp["his_CEJC"].sum()).round(1)
    cmp.loc["total"] = [cmp["ours"].sum(), cmp["his_CEJC"].sum(), round(cmp["ours"].sum() / cmp["his_CEJC"].sum(), 2), 100.0, 100.0]
    cmp.to_csv(out / "counts_vs_thesis_table_3_2.csv", encoding="utf-8-sig")
    print("\n== our counts vs his Table 3.2 (CEJC) ==\n", cmp.to_string())

    # 2) counts by listener->speaker pattern (his Table 6.2 / Fig 6.1)
    pat = d.groupby("group")[ncols].sum().reindex(order).T
    pat.index = TYPES
    pat.loc["total"] = pat.sum()
    pat.to_csv(out / "counts_by_gender_pattern.csv", encoding="utf-8-sig")
    print("\n== counts by listener->speaker gender (his Table 6.2: 5990 / 3998 / 4118 / 10650) ==\n", pat.to_string())

    # 3) chi-square, his Table 6.7 design
    factors = [("listener gender", "gender"), ("speaker gender", "partner_gender"), ("listener age", "age_grp"),
               ("speaker age", "partner_age_grp"), ("relationship", "rel5")]
    rows = []
    for t in TYPES:
        row = {"type": t}
        for (name, col), his_code in zip(factors, HIS_TABLE_6_7[t]):
            g = d.dropna(subset=[col]).groupby(col)[[f"n_{t}", "n_his_all"]].sum()
            g["other"] = g["n_his_all"] - g[f"n_{t}"]
            g = g[(g[f"n_{t}"] + g["other"]) > 0]
            p = stats.chi2_contingency(g[[f"n_{t}", "other"]].values)[1] if len(g) > 1 and g[f"n_{t}"].sum() > 0 else np.nan
            row[name] = sig(p)
            row[name + " (his)"] = his_code
            row[name + " p"] = p
        rows.append(row)
    chi = pd.DataFrame(rows)
    chi.to_csv(out / "chi_square_vs_thesis_table_6_7.csv", index=False, encoding="utf-8-sig")
    agree = sum((r[n] != "ns") == (r[n + " (his)"] != "ns") for _, r in chi.iterrows() for n, _ in factors)
    print("\n== chi-square (ours | his) ==")
    print(chi[["type"] + [c for n, _ in factors for c in (n, n + " (his)")]].to_string(index=False))
    print(f"significant/non-significant agrees with his in {agree} of {len(TYPES) * len(factors)} cells")

    # 4) rates per type and the mixed model (random intercept = conversation)
    summ = d.groupby("group")[[f"rate_{t}" for t in TYPES]].mean().reindex(order).T
    summ.index = TYPES
    summ.to_csv(out / "mean_rate_per_100_partner_by_type.csv", encoding="utf-8-sig")
    print("\n== mean rate per 100 partner lines, speaker->partner ==\n", summ.round(2).to_string())
    rows = []
    for t in TYPES:
        data = d.dropna(subset=[f"rate_{t}"])
        try:
            fit = smf.mixedlm(f"rate_{t} ~ C(gender, Treatment('M')) * C(partner_gender, Treatment('M'))", data, groups="conv").fit()
        except Exception as e:
            print(f"  model failed for {t}: {e}")
            continue
        for name in fit.params.index[1:-1]:
            rows.append(dict(type=t, term=name.replace("Treatment('M')", "M"), coef=fit.params[name], se=fit.bse[name], p=fit.pvalues[name]))
    mm = pd.DataFrame(rows)
    mm.to_csv(out / "mixed_model_by_type.csv", index=False, encoding="utf-8-sig")
    wide = mm.pivot(index="type", columns="term", values="p").reindex(TYPES).round(3)
    print("\n== mixed model p-values per type (reference = male speaker, male partner) ==\n", wide.to_string())

    colors = {"M": "#3b78c4", "F": "#e0714a"}
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for ax, t in zip(axes.ravel(), TYPES):
        m = d.groupby("group")[f"rate_{t}"].mean().reindex(order)
        e = (d.groupby("group")[f"rate_{t}"].std() / np.sqrt(d.groupby("group")[f"rate_{t}"].count())).reindex(order)
        ax.bar(range(4), m.values, yerr=e.values, color=[colors[g[-1]] for g in order], alpha=0.75, capsize=2)
        ax.set_xticks(range(4), order, fontsize=8)
        ax.set_title(t)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[g], alpha=0.75) for g in "MF"],
                            labels=["partner male", "partner female"], loc="center")
    fig.suptitle("backchannels per 100 partner lines (mean ± SE), speaker→partner")
    fig.tight_layout()
    fig.savefig(out / "rate_by_type.png", dpi=150)
    plt.close(fig)


DEFAULT_METRICS = [("strict_per_100_partner", "strict list"), ("broad_per_100_partner", "broad list")]  # BTSJ script


def by_speaker(d, out, facet=None, metrics=DEFAULT_METRICS):
    """One speaker at a time: speaker gender x partner gender, unit = backchannels per 100 partner lines.
    metrics: [(column, panel title)]. facet: optional column (e.g. relationship group); one plot row / one set of rows per
    level, plus a pooled model that includes it as a covariate."""
    import statsmodels.formula.api as smf
    plt = get_plt()
    out.mkdir(parents=True, exist_ok=True)

    d = d.copy()
    d["group"] = d["gender"] + "→" + d["partner_gender"]  # speaker→partner, e.g. F→M = woman talking to a man
    order = ["M→M", "M→F", "F→M", "F→F"]
    ms = [m for m, _ in metrics]
    levels = [None] if facet is None else list(d[facet].value_counts().index)
    sub = lambda lv: d if lv is None else d[d[facet] == lv]

    parts = []
    for lv in levels:
        t = sub(lv).groupby("group")[ms].agg(["count", "mean", "median", "std"]).reindex(order)
        if lv is not None:
            t.insert(0, facet, lv)
            t.insert(1, "n_conv", sub(lv).groupby("group")["conv"].nunique().reindex(order))
        parts.append(t)
    summ = pd.concat(parts)
    summ.to_csv(out / "speaker_summary_by_speaker_and_partner_gender.csv", encoding="utf-8-sig")
    print("\n== per speaker, backchannels per 100 partner lines (speaker→partner) ==\n", summ.round(2).to_string())

    f_base = "C(gender, Treatment('M')) * C(partner_gender, Treatment('M'))"
    specs = [("all" if facet is None else "pooled, relationship as covariate",
              f_base + ("" if facet is None else f" + C({facet})"), d)]
    if facet is not None:
        specs += [(str(lv), f_base, sub(lv)) for lv in levels]
    rows = []
    for label, formula, data in specs:
        for m in ms:
            try:  # random intercept per conversation: the two speakers of one conversation are not independent
                fit = smf.mixedlm(f"{m} ~ {formula}", data.dropna(subset=[m]), groups="conv").fit()
            except Exception as e:  # tiny cells can be singular
                print(f"  model failed for {label}/{m}: {e}")
                continue
            for name in fit.params.index[1:-1]:
                rows.append(dict(subset=label, metric=m, term=name.replace("Treatment('M')", "M"), coef=fit.params[name],
                                 se=fit.bse[name], p=fit.pvalues[name], n_speakers=len(data), n_conv=data["conv"].nunique()))
    mm = pd.DataFrame(rows)
    mm.to_csv(out / "speaker_mixed_model.csv", index=False, encoding="utf-8-sig")
    print("\n== mixed model (random intercept = conversation); reference = male speaker, male partner ==\n", mm.round(3).to_string(index=False))

    colors = {"M": "#3b78c4", "F": "#e0714a"}  # colour = partner gender
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(levels), len(ms), figsize=(5 * len(ms), 4.5 * len(levels)), squeeze=False)
    for r, lv in enumerate(levels):
        data = sub(lv)
        for ax, (m, title) in zip(axes[r], metrics):
            for gi, sg in enumerate(["M", "F"]):
                for pi, pg in enumerate(["M", "F"]):
                    v = data.loc[(data["gender"] == sg) & (data["partner_gender"] == pg), m].dropna().values
                    x = gi + (pi - 0.5) * 0.36
                    if len(v):
                        ax.boxplot(v, positions=[x], widths=0.3, showmeans=True, showfliers=False, patch_artist=True,
                                   boxprops=dict(facecolor=colors[pg], alpha=0.45), medianprops=dict(color="black"))
                        ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=8, color=colors[pg], alpha=0.6)
                    ax.annotate(f"n={len(v)}", (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -26),
                                textcoords="offset points", ha="center", fontsize=7)
            ax.set_xticks([0, 1], ["male speaker", "female speaker"])
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylabel("backchannels per 100 partner lines")
            ax.set_title(title + ("" if lv is None else f" - {lv} ({data['conv'].nunique()} conv)"))
        if r == 0:
            from matplotlib.lines import Line2D
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g], alpha=0.6) for g in "MF"]
            handles += [Line2D([], [], color="black", lw=1), Line2D([], [], marker="^", color="#2ca02c", ls="")]
            axes[0][0].legend(handles=handles, labels=["partner male", "partner female", "median", "mean"], loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "aizuchi_by_speaker_and_partner_gender.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    d.to_csv(OUT / "aizuchi_by_speaker_conversation.csv", index=False, encoding="utf-8-sig")
    print(d.drop_duplicates("conv").groupby(["rel_group", "pair"]).size().unstack(fill_value=0))

    fukunaga = [("his_all_per_100_partner", "all 11 types (incl. 笑い)"), ("his_no_laugh_per_100_partner", "10 words (without 笑い)")]
    print("\n######## no_relationship ########")
    by_speaker(d, OUT / "no_relationship", metrics=fukunaga)
    print("\n######## by_relationship ########")
    by_speaker(d, OUT / "by_relationship", facet="rel_group", metrics=fukunaga)
    print("\n######## by_type ########")
    by_type(d, OUT / "by_type")


if __name__ == "__main__":
    main()
