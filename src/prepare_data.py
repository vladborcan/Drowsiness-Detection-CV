"""Pregateste datele: video -> cadre -> splits.csv.

Exemplu:
    python -m src.prepare_data --raw-dir data/raw --out-dir data/processed
"""
import argparse
import time
from pathlib import Path

from src.config import DEFAULT_SEED
from src.data.extract_frames import extract_all
from src.data.splits import assign_splits, verify_splits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw", help="folder cu videoclipurile UTA-RLDD")
    p.add_argument("--out-dir", default="data/processed", help="unde se scriu cadrele si tabelele")
    p.add_argument("--every-sec", type=float, default=2.0, help="un cadru la cate secunde")
    p.add_argument("--size", type=int, default=160, help="latura cadrelor salvate (px)")
    p.add_argument("--workers", type=int, default=4, help="procese paralele pentru extragere")
    p.add_argument("--limit-videos", type=int, default=None, help="doar primele N video-uri (test rapid)")
    p.add_argument("--mode", choices=["temporal_blocks", "subject"], default="temporal_blocks")
    p.add_argument("--segment-sec", type=float, default=30.0)
    p.add_argument("--guard-sec", type=float, default=5.0)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TR", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    start = time.time()
    out_dir = Path(args.out_dir)
    frames = extract_all(
        args.raw_dir, out_dir, args.every_sec, args.size, args.workers, limit_videos=args.limit_videos
    )
    splits = assign_splits(
        frames, args.mode, args.seed, tuple(args.ratios), args.segment_sec, args.guard_sec
    )
    splits.to_csv(out_dir / "splits.csv", index=False)

    print(f"\nScris: {out_dir / 'splits.csv'} ({len(splits)} cadre, {len(frames) - len(splits)} eliminate din zonele-tampon)")
    for k, v in verify_splits(splits).items():
        print(f"  {k}: {v}")
    print(f"Durata totala: {(time.time() - start) / 60:.1f} minute")


if __name__ == "__main__":
    main()