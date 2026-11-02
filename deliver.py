#!/usr/bin/env python3
"""
deliver.py - end-to-end job around to_slog3.py for clips uploaded in byte-split parts.

  1. Reassemble `split` parts (sorted by name) into one file, check sha256 if given.
  2. Convert with to_slog3.py (any extra args are passed through, e.g. --exposure 2 --no-deband).
  3. Verify: same frame count and frame rate as the source, and frame alignment - output frames at
     the start, middle and end are decoded back to Rec.709 and must match the SAME source frame
     better than its neighbours (no dropped/duplicated/shifted frames -> no jump in the cut).
  4. Optionally (--split-mb N) split the result into parts for upload; print sha256.

Usage:
  python deliver.py CLIP.mp4 --name grinder -- --gamma 2.4 --exposure 2 --no-deband
  python deliver.py PARTS_DIR --name grinder --sha256 HEX --split-mb 450 -- --exposure 2   # cloud use
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import to_slog3 as T  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def reassemble(src, out):
    src = Path(src)
    if src.is_file():
        return src
    parts = sorted(p for p in src.iterdir() if p.is_file() and not p.name.startswith("."))
    if not parts:
        sys.exit(f"ERROR: no files in {src}")
    print(f"[join] {len(parts)} parts: {', '.join(p.name for p in parts)}")
    with open(out, "wb") as o:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, o, 1 << 22)
    return Path(out)


def ffprobe_bin(ff):
    return shutil.which("ffprobe", path=str(Path(ff).parent)) or shutil.which("ffprobe")


def count_frames(fp, path):
    r = subprocess.run([fp, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=nb_read_frames,r_frame_rate", "-of", "default=nw=1", str(path)],
                       capture_output=True, text=True, check=True)
    d = dict(l.split("=", 1) for l in r.stdout.split())
    return int(d["nb_read_frames"]), d["r_frame_rate"]


def grab(ff, path, idx, w, h, pix):
    """Decode frames with the given indices (in order), downscaled to w x h."""
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    r = subprocess.run([ff, "-v", "error", "-i", str(path), "-vf",
                        f"select='{sel}',scale={w}:{h}:flags=area", "-fps_mode", "passthrough",
                        "-f", "rawvideo", "-pix_fmt", pix, "-"], capture_output=True, check=True)
    ch = 3
    a = np.frombuffer(r.stdout, np.uint16)
    n = a.size // (w * h * ch)
    return a[: n * w * h * ch].reshape(n, -1)


def slog3_to_rec709(yuv_planar, w, h, exposure, gamma):
    """yuv444p10le (legal) S-Log3/S-Gamut3.Cine -> Rec.709 display RGB 0..1 (inverse of the tool)."""
    v = yuv_planar.reshape(3, h, w).astype(np.float64)
    cs = 896.0 / 876.0
    Y, Cb, Cr = v[0], (v[1] - 512) / cs, (v[2] - 512) / cs
    R = Y + 2 * (1 - T.KR) * Cr
    B = Y + 2 * (1 - T.KB) * Cb
    G = (Y - T.KR * R - T.KB * B) / T.KG
    c = np.stack([R, G, B], -1)
    lin = np.where(c >= 171.2102946929, 10 ** ((c - 420) / 261.5) * 0.19 - 0.01,
                   (c - 95) * 0.01125 / (171.2102946929 - 95)) / 2.0 ** exposure
    lin = np.clip(lin @ np.linalg.inv(T.rec709_to_sgamut3cine_matrix()).T, 0, 1)
    if gamma == "srgb":
        return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return lin ** (1 / float(gamma))


def verify_alignment(ff, src, out, n, exposure, gamma):
    w, h = 636, 270
    checks = sorted({1, n // 2, n - 2})
    want = sorted({j for i in checks for j in (i - 1, i, i + 1) if 0 <= j < n})
    s = grab(ff, src, want, w, h, "rgb48le")
    o = grab(ff, out, checks, w, h, "yuv444p10le")
    if len(s) != len(want) or len(o) != len(checks):
        print(f"[verify] FAIL could not decode check frames ({len(s)}/{len(want)}, {len(o)}/{len(checks)})")
        return False
    src_frames = {j: s[k].reshape(h, w, 3) / 65535.0 for k, j in enumerate(want)}
    ok = True
    for k, i in enumerate(checks):
        back = slog3_to_rec709(o[k], w, h, exposure, gamma)
        errs = {j: float(np.abs(back - src_frames[j]).mean() * 255) for j in (i - 1, i, i + 1) if j in src_frames}
        best = min(errs, key=errs.get)
        good = errs[i] <= errs[best] + 1e-9 and errs[i] < 2.0
        ok &= good
        print(f"[verify] out frame {i}: mean err vs source frames " +
              ", ".join(f"{j}={e:.2f}" for j, e in errs.items()) + f"  {'OK' if good else 'FAIL'}")
    return ok


def split_file(path, mb):
    size = path.stat().st_size
    limit = mb * 1024 * 1024
    if size <= limit:
        return [path]
    parts = []
    with open(path, "rb") as f:
        k = 0
        while True:
            data = f.read(limit)
            if not data:
                break
            p = path.with_name(f"{path.stem}.part{k:02d}")
            p.write_bytes(data)
            parts.append(p)
            k += 1
    return parts


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("src", help="directory of split parts, or a single video file")
    p.add_argument("--name", required=True, help="output base name")
    p.add_argument("--workdir", default=None, help="where to put the joined file and the output")
    p.add_argument("--sha256", help="expected sha256 of the joined source")
    p.add_argument("--split-mb", type=int, default=0, help="split output into parts of N MiB (0 = no)")
    a = p.parse_args(argv)

    conv = argparse.Namespace(gamma="2.4", exposure=0.0)
    for k, v in zip(passthrough, passthrough[1:] + [None]):
        if k == "--gamma":
            conv.gamma = v
        elif k == "--exposure":
            conv.exposure = float(v)

    wd = Path(a.workdir or Path(a.src).resolve().parent / f"{a.name}_job")
    wd.mkdir(parents=True, exist_ok=True)
    src = reassemble(a.src, wd / f"{a.name}_source.mp4")
    digest = sha256(src)
    print(f"[join] sha256 {digest}")
    if a.sha256 and a.sha256.lower() != digest:
        sys.exit(f"ERROR: sha256 mismatch, expected {a.sha256}")

    ff = T.find_ffmpeg()
    fp = ffprobe_bin(ff)
    n_src, rate_src = count_frames(fp, src)
    print(f"[source] {n_src} frames @ {rate_src}")

    out = wd / f"{a.name}_SLog3_SGamut3Cine.mov"
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "to_slog3.py"), str(src), "-o", str(out),
           *passthrough]
    print("[run]", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit("ERROR: conversion failed")

    n_out, rate_out = count_frames(fp, out)
    same = n_out == n_src and rate_out == rate_src
    print(f"[verify] frames {n_out} vs {n_src}, rate {rate_out} vs {rate_src}  {'OK' if same else 'FAIL'}")
    aligned = verify_alignment(ff, src, out, n_src, conv.exposure, conv.gamma)
    if not (same and aligned):
        sys.exit("ERROR: verification failed - do not deliver")

    out_sha = sha256(out)
    parts = split_file(out, a.split_mb) if a.split_mb else [out]
    print(f"[deliver] {out}  ({out.stat().st_size / 2**20:.0f} MiB)  sha256 {out_sha}")
    if len(parts) > 1:
        print(f"[deliver] {len(parts)} parts: " + ", ".join(str(x) for x in parts))
        print(f"[deliver] join with:  cat {out.stem}.part* > {out.name}")


if __name__ == "__main__":
    main()
