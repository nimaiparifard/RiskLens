"""Convert the recorded demo (docs/media/demo.webm) into README-friendly media.

* demo.mp4  - H.264, for uploading to GitHub (inline player) or LinkedIn
* demo.gif  - plays everywhere (README, docs), 2-pass palette for crisp UI text

    python scripts/make_demo_media.py [--fps 12] [--width 1000]
Needs ffmpeg on PATH or `pip install imageio-ffmpeg`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

MEDIA = Path(__file__).resolve().parents[1] / "docs" / "media"


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(*args: str) -> None:
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=1000)
    a = ap.parse_args()
    src = MEDIA / "demo.webm"
    if not src.exists():
        raise SystemExit(f"{src} not found - run frontend/e2e/record_demo.mjs first")

    mp4 = MEDIA / "demo.mp4"
    run("-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
        "-movflags", "+faststart", str(mp4))

    palette = MEDIA / "_palette.png"
    scale = f"fps={a.fps},scale={a.width}:-1:flags=lanczos"
    run("-i", str(src), "-vf", f"{scale},palettegen=stats_mode=diff", str(palette))
    gif = MEDIA / "demo.gif"
    run("-i", str(src), "-i", str(palette), "-lavfi",
        f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", str(gif))
    palette.unlink()

    for f in (mp4, gif):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
