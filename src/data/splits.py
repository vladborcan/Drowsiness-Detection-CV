"""Impartirea cadrelor in train / val / test.

Mod implicit: "temporal_blocks" (detectie dependenta de subiect).
  Fiecare video este taiat in segmente consecutive de `segment_sec` secunde.
  Segmentele (nu cadrele individuale) sunt repartizate aleatoriu (seed fix)
  in train/val/test, separat pentru fiecare video. Astfel:
    - fiecare subiect si fiecare clasa apar in toate cele 3 split-uri;
    - cadrele aproape identice dintr-o secventa scurta raman in acelasi split;
    - la granitele dintre segmente cu split-uri diferite se elimina cadrele aflate
      la mai putin de `guard_sec` de granita (separare minima intre split-uri
      >= 2 * guard_sec).

Mod alternativ: "subject" (detectie independenta de subiect) - subiecti
intregi sunt repartizati in split-uri. Util pentru comparatie in raport.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEFAULT_SEED, SPLITS


def _counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Cate elemente merg in train/val/test (val si test cel putin 1 daca n >= 3)."""
    if n < 3:
        return n, 0, 0
    n_val = max(1, round(ratios[1] * n))
    n_test = max(1, round(ratios[2] * n))
    return n - n_val - n_test, n_val, n_test


def assign_splits(
    frames: pd.DataFrame,
    mode: str = "temporal_blocks",
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    segment_sec: float = 30.0,
    guard_sec: float = 5.0,
) -> pd.DataFrame:
    """Adauga coloana `split` si elimina cadrele din zonele-tampon."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Proportiile trebuie sa insumeze 1.")
    if mode == "temporal_blocks":
        return _split_temporal_blocks(frames, seed, ratios, segment_sec, guard_sec)
    if mode == "subject":
        return _split_by_subject(frames, seed, ratios)
    raise ValueError(f"Mod necunoscut: {mode}")


def _split_temporal_blocks(frames, seed, ratios, segment_sec, guard_sec):
    out = []
    for v_idx, (_, vid) in enumerate(frames.groupby("video", sort=True)):
        vid = vid.sort_values("time_sec").copy()
        seg = np.floor(vid["time_sec"].to_numpy() / segment_sec).astype(int)
        segments = np.unique(seg)

        n_train, n_val, n_test = _counts(len(segments), ratios)
        rng = np.random.default_rng([seed, v_idx])  # deterministic, independent pe video
        order = rng.permutation(segments)
        seg_split = {}
        for s in order[:n_train]:
            seg_split[s] = "train"
        for s in order[n_train : n_train + n_val]:
            seg_split[s] = "val"
        for s in order[n_train + n_val :]:
            seg_split[s] = "test"

        t = vid["time_sec"].to_numpy()
        keep = np.ones(len(vid), dtype=bool)
        for i, (ti, si) in enumerate(zip(t, seg)):
            cur = seg_split[si]
            if ti - si * segment_sec < guard_sec and seg_split.get(si - 1, cur) != cur:
                keep[i] = False
            if (si + 1) * segment_sec - ti <= guard_sec and seg_split.get(si + 1, cur) != cur:
                keep[i] = False
        vid["split"] = [seg_split[s] for s in seg]
        out.append(vid[keep])
    return pd.concat(out, ignore_index=True)


def _split_by_subject(frames, seed, ratios):
    subjects = np.array(sorted(frames["subject"].unique()))
    n_train, n_val, n_test = _counts(len(subjects), ratios)
    order = np.random.default_rng(seed).permutation(subjects)
    mapping = {s: "train" for s in order[:n_train]}
    mapping.update({s: "val" for s in order[n_train : n_train + n_val]})
    mapping.update({s: "test" for s in order[n_train + n_val :]})
    out = frames.copy()
    out["split"] = out["subject"].map(mapping)
    return out.sort_values(["video", "time_sec"]).reset_index(drop=True)


def verify_splits(df: pd.DataFrame) -> dict:
    """Verificari de baza pentru raport: dubluri, separare temporala, subiecti comuni."""
    report = {
        "frames_per_split": df["split"].value_counts().reindex(SPLITS, fill_value=0).to_dict(),
        "duplicate_paths": int(df["path"].duplicated().sum()),
    }
    min_gap = np.inf
    for _, vid in df.groupby("video"):
        vid = vid.sort_values("time_sec")
        diff_split = vid["split"].to_numpy()[1:] != vid["split"].to_numpy()[:-1]
        gaps = np.diff(vid["time_sec"].to_numpy())[diff_split]
        if len(gaps):
            min_gap = min(min_gap, float(gaps.min()))
    report["min_gap_between_splits_sec"] = None if min_gap == np.inf else min_gap

    subj = {s: set(df.loc[df["split"] == s, "subject"]) for s in SPLITS}
    report["subjects_shared_train_test"] = len(subj["train"] & subj["test"])
    report["subjects_per_split"] = {s: len(v) for s, v in subj.items()}
    return report