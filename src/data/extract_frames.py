"""Extragerea cadrelor din videoclipurile UTA-RLDD.

Pentru fiecare video:
  * eticheta (0 / 5 / 10) se citeste din numele fisierului; daca o clasa e
    impartita in mai multe fisiere (10_1, 10_2), fiecare e tratat ca video separat;
  * subiectul se ia din numele folderului parinte;
  * se pastreaza un cadru la fiecare `every_sec` secunde;
  * cadrul se redimensioneaza (latura scurta = `size`) si se decupeaza central
    la `size x size`, apoi se salveaza ca JPEG.

Rezultatul este un tabel (frames.csv) cu cate o linie pe cadru.
Procesarea poate fi reluata: videoclipurile deja terminate au un manifest
in `<out_dir>/manifests/` si sunt sarite.
"""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
COLUMNS = ["path", "subject", "label", "video", "frame_idx", "time_sec"]

# "0", "5", "10" sau, cand clasa e impartita in mai multe fisiere, "10_1", "10_2"
NAME_PATTERN = re.compile(r"(0|5|10)(?:_(\d+))?")


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    subject: str
    label: int
    stem: str  # numele fisierului fara extensie: "0", "10", "10_1" ...

    @property
    def key(self) -> str:
        """Identificator unic al videoclipului, ex: '32/10_1'."""
        return f"{self.subject}/{self.stem}"

    @property
    def uid(self) -> str:
        """Varianta sigura pentru nume de fisier, ex: '32_10_1'."""
        return f"{self.subject}_{self.stem}"


def find_videos(raw_dir: str | Path) -> tuple[list[VideoInfo], list[Path]]:
    """Cauta recursiv videoclipuri numite 0.*, 5.*, 10.* (sau 10_1.*, 10_2.* etc.).

    Returneaza (videoclipuri recunoscute, fisiere video ignorate).
    """
    videos: list[VideoInfo] = []
    skipped: list[Path] = []
    for p in sorted(Path(raw_dir).rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        stem = p.stem.strip()
        m = NAME_PATTERN.fullmatch(stem)
        if m:
            videos.append(VideoInfo(p, p.parent.name, int(m.group(1)), stem))
        else:
            skipped.append(p)
    return videos, skipped


def _square_resize(frame, size: int):
    """Latura scurta -> size, apoi decupare centrala size x size."""
    h, w = frame.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    return frame[top : top + size, left : left + size]


def extract_video(
    video: VideoInfo,
    frames_dir: Path,
    every_sec: float,
    size: int,
    jpeg_quality: int = 90,
    max_frames: int | None = None,
) -> list[dict]:
    """Extrage cadrele unui singur video si intoarce randurile pentru tabel."""
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise OSError(f"Nu pot deschide {video.path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 240:  # 0, NaN sau valoare absurda
        fps = 30.0
    step = max(1, round(fps * every_sec))

    rel_dir = Path(video.subject) / video.stem
    (frames_dir / rel_dir).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    idx = 0
    while cap.grab():  # grab() decodeaza fara conversie; retrieve() doar cand avem nevoie
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                rel = rel_dir / f"{idx:06d}.jpg"
                dest = frames_dir / rel
                if not dest.exists():
                    cv2.imwrite(
                        str(dest),
                        _square_resize(frame, size),
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                    )
                rows.append(
                    {
                        "path": rel.as_posix(),
                        "subject": video.subject,
                        "label": video.label,
                        "video": video.key,
                        "frame_idx": idx,
                        "time_sec": round(idx / fps, 3),
                    }
                )
                if max_frames and len(rows) >= max_frames:
                    break
        idx += 1
    cap.release()
    return rows


def _process(args) -> tuple[str, int]:
    video, out_dir, every_sec, size, max_frames = args
    out_dir = Path(out_dir)
    manifest = out_dir / "manifests" / f"{video.uid}.csv"
    if manifest.exists():
        return video.key, len(pd.read_csv(manifest))
    rows = extract_video(video, out_dir / "frames", every_sec, size, max_frames=max_frames)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(manifest, index=False)
    return video.key, len(rows)


def extract_all(
    raw_dir: str | Path,
    out_dir: str | Path,
    every_sec: float = 2.0,
    size: int = 160,
    workers: int = 4,
    max_frames: int | None = None,
    limit_videos: int | None = None,
) -> pd.DataFrame:
    """Extrage cadrele din toate videoclipurile si scrie `<out_dir>/frames.csv`."""
    out_dir = Path(out_dir)
    videos, skipped = find_videos(raw_dir)
    if skipped:
        print(f"[atentie] {len(skipped)} fisiere video ignorate (nume neasteptat):")
        for p in skipped[:10]:
            print(f"   {p}")
    if not videos:
        raise FileNotFoundError(
            f"Niciun video 0/5/10 gasit in {raw_dir}. Verifica structura folderelor (vezi README)."
        )
    if limit_videos:
        videos = videos[:limit_videos]
    print(f"{len(videos)} videoclipuri, un cadru la {every_sec}s, {size}x{size}px, {workers} procese")

    jobs = [(v, str(out_dir), every_sec, size, max_frames) for v in videos]
    if workers <= 1:
        results = []
        for i, job in enumerate(jobs, 1):
            results.append(_process(job))
            print(f"[{i}/{len(jobs)}] {results[-1][0]}: {results[-1][1]} cadre")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = []
            for i, res in enumerate(ex.map(_process, jobs), 1):
                results.append(res)
                print(f"[{i}/{len(jobs)}] {res[0]}: {res[1]} cadre")

    parts = [pd.read_csv(out_dir / "manifests" / f"{v.uid}.csv") for v in videos]
    frames = pd.concat(parts, ignore_index=True)
    frames.to_csv(out_dir / "frames.csv", index=False)
    return frames