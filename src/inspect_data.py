"""Inspectie rapida a datelor pregatite (pentru verificare si documentatie).

Scrie in outputs/:
  split_timeline.png - cum sunt repartizate segmentele din cateva video-uri
  samples.png        - cadre exemplu din train, pe clase
Daca PyTorch e instalat, incarca si un batch din DataLoader.

    python -m src.inspect_data --data-dir data/processed
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.config import CLASS_NAMES, LABEL_TO_INDEX, SPLITS
from src.data.splits import verify_splits

COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default="outputs")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(exist_ok=True)
    df = pd.read_csv(data_dir / "splits.csv")

    print("Cadre pe split si clasa:")
    print(pd.crosstab(df["split"], df["label"]).reindex(list(SPLITS)))
    for k, v in verify_splits(df).items():
        print(f"  {k}: {v}")

    # cronologia split-urilor pentru primele 6 video-uri
    vids = sorted(df["video"].unique())[:6]
    fig, axes = plt.subplots(len(vids), 1, figsize=(10, 1.2 * len(vids)), sharex=True)
    for ax, v in zip(axes if len(vids) > 1 else [axes], vids):
        sub = df[df["video"] == v]
        for s in SPLITS:
            m = sub["split"] == s
            ax.scatter(sub.loc[m, "time_sec"], [0] * m.sum(), s=14, c=COLORS[s], label=s, marker="|")
        ax.set_yticks([])
        ax.set_ylabel(v, rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("timp in video (s)")
    axes[0].legend(ncol=3, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "split_timeline.png", dpi=130)

    # cadre exemplu: 6 din train pentru fiecare clasa
    train = df[df["split"] == "train"]
    size = 112
    sheet = Image.new("RGB", (size * 6, size * len(LABEL_TO_INDEX)))
    for r, label in enumerate(LABEL_TO_INDEX):
        rows = train[train["label"] == label].sample(min(6, (train["label"] == label).sum()), random_state=0)
        for c, path in enumerate(rows["path"]):
            with Image.open(data_dir / "frames" / path) as im:
                sheet.paste(im.convert("RGB").resize((size, size)), (c * size, r * size))
    sheet.save(out / "samples.png")
    print(f"Salvat: {out / 'split_timeline.png'}, {out / 'samples.png'} (randuri: {CLASS_NAMES})")

    try:
        from src.data.dataset import get_dataloaders
    except ImportError:
        print("PyTorch nu este instalat - sar peste verificarea DataLoader-ului.")
        return
    loaders = get_dataloaders(data_dir / "splits.csv", data_dir / "frames", batch_size=args.batch_size, num_workers=0)
    x, y = next(iter(loaders["train"]))
    print(f"Batch train: x {tuple(x.shape)} {x.dtype}, y {tuple(y.shape)} {y.dtype}, clase in batch: {y.tolist()}")
    for s, ld in loaders.items():
        print(f"  {s}: {len(ld.dataset)} cadre, {len(ld)} batch-uri, {ld.dataset.class_counts()}")


if __name__ == "__main__":
    main()