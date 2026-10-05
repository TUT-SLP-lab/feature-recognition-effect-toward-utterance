"""
Convert a JVS corpus speaker's parallel100 subset into Style-Bert-VITS2's
Data/{model_name}/raw + esd.list format, reusing JVS's own transcripts
instead of running ASR.

Usage:
    python jvs_to_esd.py --speaker jvs093
"""

import argparse
import shutil
from pathlib import Path

SBV2_ROOT = Path("/home/ryuu/Ryu/System_implement/Style-Bert-VITS2")
JVS_ROOT = Path("/home/ryuu/Ryu/Corpus/JVS/jvs_ver1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True, help="e.g. jvs093")
    parser.add_argument(
        "--subset", default="parallel100", help="parallel100 or nonpara30"
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Defaults to the speaker id (e.g. jvs093)",
    )
    args = parser.parse_args()

    model_name = args.model_name or args.speaker
    src_dir = JVS_ROOT / args.speaker / args.subset
    wav_dir = src_dir / "wav24kHz16bit"
    transcript_file = src_dir / "transcripts_utf8.txt"

    if not wav_dir.is_dir():
        raise SystemExit(f"wav dir not found: {wav_dir}")
    if not transcript_file.is_file():
        raise SystemExit(f"transcript file not found: {transcript_file}")

    out_root = SBV2_ROOT / "Data" / model_name
    out_raw = out_root / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    # parse transcripts: "VOICEACTRESS100_001:text"
    transcripts: dict[str, str] = {}
    for line in transcript_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, text = line.split(":", 1)
        transcripts[utt_id] = text

    esd_lines = []
    missing = 0
    for utt_id, text in sorted(transcripts.items()):
        src_wav = wav_dir / f"{utt_id}.wav"
        if not src_wav.is_file():
            print(f"warning: missing wav for {utt_id}, skipping")
            missing += 1
            continue
        dst_wav = out_raw / f"{utt_id}.wav"
        shutil.copyfile(src_wav, dst_wav)
        esd_lines.append(f"{utt_id}.wav|{model_name}|JP|{text}")

    esd_path = out_root / "esd.list"
    esd_path.write_text("\n".join(esd_lines) + "\n", encoding="utf-8")

    print(f"Copied {len(esd_lines)} wavs to {out_raw}")
    if missing:
        print(f"Skipped {missing} missing wavs")
    print(f"Wrote {esd_path}")


if __name__ == "__main__":
    main()
