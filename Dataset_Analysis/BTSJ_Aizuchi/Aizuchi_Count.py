"""Count backchannels (aizuchi) in BTSJ two-speaker native-Japanese conversations; compare by speaker gender x partner gender.

Same measures, word lists and statistics as ../CEJC_Aizuchi/Aizuchi_Count.py (imported from there).
BTSJ marks two kinds of backchannel, counted separately and together:
  standalone: a line whose whole text is backchannel words              JM002: うん。
  inline    : "(うん)" written inside the OTHER speaker's line           JM001: おれねー(うん)、ここねー  -> JM002's backchannel
Main measure = standalone + inline, per 100 partner content lines (partner lines that are not standalone backchannels).

Selection: native-speaker (母語場面), two speakers, face-to-face, genre = casual chat (雑談), natural conversation.
Relationship groups (話者同士の面識 / 関係): friend = known friends of equal status, first_meeting = strangers of equal status,
other = any status-unequal pair (senior-junior, teacher-student).

Run: python Aizuchi_Count.py
Output: output/no_relationship/        pooled, standalone + inline
        output/by_relationship/        split by friend / first_meeting / other
        output/standalone_only/        pooled, standalone lines only (directly comparable with the CEJC count)
        output/aizuchi_by_speaker_conversation.csv
"""
import importlib.util
import re
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

warnings.filterwarnings("ignore", message="Unknown extension")

HERE = Path(__file__).parent
BTSJ = HERE.parents[1] / "Corpus" / "BTSJ" / "2021年版" / "解凍版"
OUT = HERE / "output"
GENRES = {"雑談"}  # add "討論", "論文指導" to widen; phone calls and role-play are excluded on purpose

spec = importlib.util.spec_from_file_location("cejc_aizuchi", HERE.parent / "CEJC_Aizuchi" / "Aizuchi_Count.py")
cejc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cejc)
STRICT_RE, BROAD_RE = cejc.STRICT_RE, cejc.BROAD_RE


def squash(s):
    return re.sub(r"\s", "", str(s))


def sheet(wb, name):
    rows = list(wb[name].iter_rows(values_only=True))
    cols = [squash(c) if c else f"_{i}" for i, c in enumerate(rows[0])]
    return pd.DataFrame(rows[1:], columns=cols).dropna(subset=[cols[0]])


def clean(text):
    """Remove annotations; keep '(..)' groups (inline backchannels) intact."""
    t = re.sub(r"<([^<>]*)>\{[<>]\}", r"\1", str(text))  # overlapped stretch <おれ>{<}: keep the words
    t = re.sub(r"\{[<>]\}|[【】]+", "", t)
    t = re.sub(r"<[^<>]*>", "", t)  # remaining <..> are comments: <笑い>, <笑いながら>
    t = re.sub(r"\[[^\[\]]*\]", "", t)  # [↑] [小声で]
    t = re.sub(r"#+|[=＝]", "", t)
    return re.sub(r"[\s。、．，,.?？!！「」『』~〜…]+", "", t)


def analyse_line(text):
    """-> (standalone_strict, standalone_broad, inline_strict, inline_broad, content_chars)"""
    t = clean(text)
    inline = [g for g in re.findall(r"\(([^()]*)\)", t) if g]
    rest = re.sub(r"\([^()]*\)", "", t)
    rest = re.sub(r"[()]", "", rest)
    if rest:
        s, b = bool(STRICT_RE.match(rest)), bool(BROAD_RE.match(rest))
    else:  # line made only of "(..)" groups: nothing of its own was said
        s = b = False
    in_s = sum(bool(STRICT_RE.match(re.sub(r"[()]", "", g))) for g in inline)
    in_b = sum(bool(BROAD_RE.match(re.sub(r"[()]", "", g))) for g in inline)
    return s, b, in_s, in_b, (0 if b or not rest else len(rest))


def serial(x):
    return str(int(float(x))).zfill(3)  # "001", 1 and 1.0 all mean conversation 1


def to_minutes(t):
    return t.hour * 60 + t.minute + t.second / 60 if hasattr(t, "hour") else np.nan


def load():
    wb = openpyxl.load_workbook(next(BTSJ.glob("2.*.xlsx")), read_only=True)
    conv, spk = sheet(wb, "②個別会話情報"), sheet(wb, "③話者情報")
    ccol = {k: k for k in conv.columns}
    fcol = next(c for c in conv.columns if c.startswith("トランスクリプトのファイル名"))
    acol = next(c for c in conv.columns if c.startswith("真正性"))
    mcol = next(c for c in conv.columns if c.startswith("話者同士の面識"))
    conv = conv[(conv["母語場面／接触場面"].map(squash) == "母語場面") & (conv["話者の数"] == 2)
                & (conv["対面/電話"] == "対面") & conv["会話のジャンル"].isin(GENRES) & (conv[acol] == "自然会話")]
    gender = {(serial(r["会話の通し番号"]), r["話者記号"]): {"女": "F", "男": "M"}.get(r["性別"])
              for _, r in spk.iterrows()}
    files = {p.stem: p for p in BTSJ.rglob("*.xlsx") if not p.name.startswith("~$")}

    def rel_group(r):
        if "上下" in str(r["話者同士の関係"]) or "教師" in str(r["話者同士の関係"]):
            return "other"
        return "friend" if "友人" in str(r[mcol]) else "first_meeting" if "初対面" in str(r[mcol]) else "other"

    rows, skipped = [], []
    for _, c in conv.iterrows():
        name, cid = c[fcol], serial(c["会話の通し番号"])
        spks = name.split("-")[2:]
        g = {s: gender.get((cid, s)) for s in spks}
        if name not in files or len(spks) != 2 or None in g.values():
            skipped.append(name)
            continue
        ws = openpyxl.load_workbook(files[name], read_only=True).active
        lines = [(str(r[6]), r[7]) for r in ws.iter_rows(values_only=True)
                 if isinstance(r[3], (int, float)) and r[6] in spks and r[7] is not None]
        a = pd.DataFrame([(s, *analyse_line(t)) for s, t in lines],
                         columns=["spk", "sa_strict", "sa_broad", "in_strict", "in_broad", "chars"])
        for me, other in (spks, spks[::-1]):
            mine, theirs = a[a["spk"] == me], a[a["spk"] == other]
            row = dict(conv=name, pair="-".join(sorted(g.values())), gender=g[me], partner_gender=g[other], label=me,
                       relationship=f"{c[mcol]}/{c['話者同士の関係']}", rel_group=rel_group(c),
                       minutes=to_minutes(c["会話時間"]), units=len(mine),
                       partner_content_units=int((~theirs["sa_broad"].astype(bool) & (theirs["chars"] > 0)).sum()),
                       partner_chars=int(theirs["chars"].sum()))
            for k in ("strict", "broad"):
                row[f"standalone_{k}"] = int(mine[f"sa_{k}"].sum())
                row[f"inline_{k}"] = int(theirs[f"in_{k}"].sum())  # written in the partner's lines = this speaker's
                row[k] = row[f"standalone_{k}"] + row[f"inline_{k}"]
            rows.append(row)
    d = pd.DataFrame(rows)
    for k in ("strict", "broad"):
        for src, tag in ((k, ""), (f"standalone_{k}", "standalone_")):
            d[f"{tag}{k}_per_100_partner"] = 100 * d[src] / d["partner_content_units"].replace(0, np.nan)
            d[f"{tag}{k}_per_1000_partner_chars"] = 1000 * d[src] / d["partner_chars"].replace(0, np.nan)
        d[f"{k}_per_min"] = d[k] / d["minutes"]
    print(f"conversations used: {d['conv'].nunique()}  skipped (no transcript / unknown gender): {len(skipped)} {skipped}")
    return d


def main():
    d = load()
    OUT.mkdir(exist_ok=True)
    d.to_csv(OUT / "aizuchi_by_speaker_conversation.csv", index=False, encoding="utf-8-sig")
    one = d.drop_duplicates("conv")
    print("\nconversations by relationship group x gender pair\n", pd.crosstab(one["rel_group"], one["pair"]).to_string())
    print("\nstandalone vs inline (broad), mean count per speaker\n",
          d.groupby("pair")[["standalone_broad", "inline_broad"]].mean().round(1).to_string())

    for name, data, facet in (("no_relationship", d, None), ("by_relationship", d, "rel_group")):
        out = OUT / name
        out.mkdir(exist_ok=True)
        print(f"\n######## {name} ########")
        cejc.by_speaker(data, out, facet=facet)

    out = OUT / "standalone_only"
    out.mkdir(exist_ok=True)
    s = d.copy()
    for k in ("strict", "broad"):
        s[f"{k}_per_100_partner"] = s[f"standalone_{k}_per_100_partner"]
    print("\n######## standalone_only ########")
    cejc.by_speaker(s, out)


if __name__ == "__main__":
    main()
