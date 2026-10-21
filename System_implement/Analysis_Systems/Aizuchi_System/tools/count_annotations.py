"""Count the aizuchi (and other marks) in the annotation files made with annotate_aizuchi.py.

Usage:  python tools/count_annotations.py [logs/pilot_test/segawa_female ...] [--shift 0.25]     (no session = every annotated session under logs/)
        python tools/count_annotations.py --out logs/annotation_summary.csv

Per session it prints and writes: the marks by type, aizuchi per minute, how many aizuchi the system had logged as a backchannel
(and how many of the system's logged backchannels you did not confirm as an aizuchi), and how the aizuchi relate to the system's
cues: share of strong (|S) and weak (|W) slots that got an aizuchi, and where in time the aizuchi fall around a phrase end.
Everything is recomputed from the marks' times and the session log, so it also works on files saved by an older version of the tool.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

import annotate_aizuchi as A

WINDOW = (-0.3, 1.2)  # an aizuchi counts as the answer to a phrase end if it starts this many seconds before / after that clip's end


def summarize(rel: str, shift: float = 0.0) -> dict | None:
    d = A.session_dir(rel)
    f = d / "annotations.json"
    if not f.exists():
        return None
    info = A.session_info(rel)
    raw = json.loads(f.read_text(encoding="utf-8")).get("marks", [])
    marks = [dict(m, t=m["t"] + shift) for m in raw]  # shift: a correction for marks placed with the old key-press delay (read-only; the file is not changed)
    marks = [dict(m, **A.describe(m, info)) for m in marks]
    minutes = info["duration"] / 60
    n = lambda t: sum(1 for m in marks if m["type"] == t)  # noqa: E731
    aiz = [m for m in marks if m["type"] == "aizuchi"]
    caught = [m for m in aiz if m["caught_by_system"]]
    logged_bc = [e for e in info["events"] if e["role"] == "backchannel"]
    confirmed = [e for e in logged_bc if any(abs(m["t"] - e["start"]) <= 0.6 or e["start"] - 0.4 <= m["t"] <= e["end"] + 0.4 for m in aiz)]
    row = {"session": rel, "script": info.get("script"), "minutes": round(minutes, 2), "marks": len(marks),
           "aizuchi": len(aiz), "laugh": n("laugh"), "reaction": n("reaction"), "other": n("other"),
           "aizuchi_per_min": round(len(aiz) / minutes, 2) if minutes else None,
           "aizuchi_caught_by_system": len(caught), "aizuchi_missed_by_system": len(aiz) - len(caught),
           "system_backchannels_logged": len(logged_bc), "system_backchannels_confirmed": len(confirmed)}
    for cue in ("S", "W"):  # slots = clips of that cue; answered = an aizuchi within the window of the clip's end
        slots = [c for c in info["clips"] if c["cue"] == cue]
        hit = [c for c in slots if any(c["end"] + WINDOW[0] <= m["t"] <= c["end"] + WINDOW[1] for m in aiz)]
        row[f"{cue}_slots"], row[f"{cue}_slots_with_aizuchi"] = len(slots), len(hit)
    row["aizuchi_during_system_speech"] = sum(1 for m in aiz if m.get("during_clip"))
    off = [m["offset_from_nearest_clip_end"] for m in aiz if m.get("offset_from_nearest_clip_end") is not None and WINDOW[0] <= m["offset_from_nearest_clip_end"] <= WINDOW[1]]
    row["onset_after_phrase_end_median_s"] = round(st.median(off), 2) if off else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--out", default=str(A.LOGS / "annotation_summary.csv"))
    ap.add_argument("--shift", type=float, default=0.0, help="seconds added to every mark time before counting (e.g. 0.25 for marks made with the old key-press delay)")
    args = ap.parse_args()
    rels = [str(Path(s).resolve().relative_to(A.LOGS).as_posix()) for s in args.sessions] if args.sessions else \
        sorted(f.parent.relative_to(A.LOGS).as_posix() for f in A.LOGS.glob("**/annotations.json"))
    rows = [r for r in (summarize(x, args.shift) for x in rels) if r]
    if not rows:
        print("no annotated sessions found")
        return
    for r in rows:
        print(f"\n== {r['session']}  ({r['minutes']} min, {r['script']})")
        print(f"   marks {r['marks']}: aizuchi {r['aizuchi']} ({r['aizuchi_per_min']}/min), laugh {r['laugh']}, reaction {r['reaction']}, other {r['other']}")
        print(f"   aizuchi the system logged as a backchannel: {r['aizuchi_caught_by_system']} of {r['aizuchi']} (missed: {r['aizuchi_missed_by_system']})")
        print(f"   system-logged backchannels you confirmed as aizuchi: {r['system_backchannels_confirmed']} of {r['system_backchannels_logged']}")
        print(f"   strong slots with an aizuchi: {r['S_slots_with_aizuchi']} of {r['S_slots']} | weak: {r['W_slots_with_aizuchi']} of {r['W_slots']} "
              f"(window {WINDOW[0]:+}..{WINDOW[1]:+} s around the clip end)")
        print(f"   aizuchi during system speech: {r['aizuchi_during_system_speech']} | median onset after a phrase end: {r['onset_after_phrase_end_median_s']} s")
    if len(rows) > 1:
        tot = {k: sum(r[k] for r in rows) for k in ("aizuchi", "laugh", "reaction", "other", "aizuchi_caught_by_system", "aizuchi_missed_by_system")}
        print("\n== all sessions together:", tot)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
