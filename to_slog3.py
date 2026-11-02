#!/usr/bin/env python3
"""
to_slog3.py - 8-bit Rec.709 video -> 10-bit S-Gamut3.Cine / S-Log3 ProRes 4444.

Pipeline (all heavy math in float32, nothing is quantised to 8-bit in between):
  1. Decode with high-quality 4:2:0 -> 4:4:4 chroma upsampling, 16-bit.
  2. Optional deband (ffmpeg `deband`) on the 16-bit 4:4:4 signal.
  3. 16-bit RGB -> linear light (Gamma 2.4 / 2.2 / sRGB), optional exposure.
  4. Rec.709 primaries -> S-Gamut3.Cine primaries (3x3 matrix derived from xy).
  5. Sony S-Log3 OETF.
  6. TPDF dither (+ optional grain) at 10-bit LSB amplitude, so no new banding.
  7. Encode ProRes 4444 yuv444p10le, audio copied as PCM.

What this does NOT do: it cannot create dynamic range or colour that the 8-bit
source never had. It builds a clean, correctly encoded 10-bit S-Log3 file.

Usage:
  python to_slog3.py --selftest
  python to_slog3.py input.mp4                    # -> input_SLog3_SGamut3Cine.mov
  python to_slog3.py input.mp4 --frames 30        # quick test on 30 frames
  python to_slog3.py input.mp4 --gamma srgb --grain 1.5 -o out.mov
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# --------------------------------------------------------------------------- #
# Dependency bootstrap
# --------------------------------------------------------------------------- #
def pip_install(pkg):
    for extra in ([], ["--break-system-packages"]):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg, *extra])
        if r.returncode == 0:
            return True
    return False


try:
    import numpy as np
except ImportError:
    print("[setup] installing numpy ...", file=sys.stderr)
    if not pip_install("numpy"):
        sys.exit("ERROR: could not install numpy (pip install numpy)")
    import numpy as np


def find_ffmpeg():
    p = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if p:
        return p
    print("[setup] ffmpeg not found, installing imageio-ffmpeg (bundled binary) ...", file=sys.stderr)
    if pip_install("imageio-ffmpeg"):
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    sys.exit("ERROR: ffmpeg not found. Install it (macOS: brew install ffmpeg) or set FFMPEG=/path/to/ffmpeg")


# --------------------------------------------------------------------------- #
# Colour science
# --------------------------------------------------------------------------- #
D65 = (0.3127, 0.3290)
PRIM_REC709 = [(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)]
# Sony S-Gamut3.Cine primaries (Sony technical summary for S-Log3 / S-Gamut3)
PRIM_SGAMUT3_CINE = [(0.766, 0.275), (0.225, 0.800), (0.089, -0.087)]


def rgb_to_xyz_matrix(prim, white):
    cols = np.array([[x / y, 1.0, (1.0 - x - y) / y] for x, y in prim], dtype=np.float64).T
    w = np.array([white[0] / white[1], 1.0, (1.0 - white[0] - white[1]) / white[1]])
    s = np.linalg.solve(cols, w)
    return cols * s


def rec709_to_sgamut3cine_matrix():
    return np.linalg.inv(rgb_to_xyz_matrix(PRIM_SGAMUT3_CINE, D65)) @ rgb_to_xyz_matrix(PRIM_REC709, D65)


def build_eotf_lut(gamma):
    """65536-entry LUT: 16-bit full-range display code -> linear light (float32)."""
    v = np.arange(65536, dtype=np.float64) / 65535.0
    if gamma == "srgb":
        lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    else:
        lin = v ** float(gamma)
    return lin.astype(np.float32)


def slog3_oetf(x):
    """Sony S-Log3: scene-linear (0.18 = 18% grey) -> normalised code value (code / 1023)."""
    x = x.astype(np.float32, copy=False)
    lo = (x * ((171.2102946929 - 95.0) / 0.01125) + 95.0) / 1023.0
    hi = (420.0 + np.log10((np.maximum(x, 0.01125) + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0
    return np.where(x >= 0.01125, hi, lo)


def to_slog3_frame(rgb16, lut, matrix, gain, range_mode, rng, dither, grain):
    """uint16 HxWx3 full-range display RGB -> uint16 HxWx3 full-range RGB ready for ffmpeg."""
    lin = lut[rgb16]                                   # float32, linear Rec.709
    if gain != 1.0:
        lin *= np.float32(gain)
    lin = lin @ matrix.T.astype(np.float32)            # -> S-Gamut3.Cine primaries
    code = slog3_oetf(lin) * np.float32(1023.0)        # 10-bit code values (float)
    h, w = code.shape[:2]
    if dither:
        code += (rng.random((h, w, 1), dtype=np.float32) - rng.random((h, w, 1), dtype=np.float32))
    if grain > 0:
        code += rng.standard_normal((h, w, 1), dtype=np.float32) * np.float32(grain)
    if range_mode == "legal":
        # Stored so that luma code == S-Log3 code (95 / 420 / 598), legal-range stream.
        # ffmpeg will apply 64 + 876*Y, so pre-warp: Y_full = (code - 64) / 876.
        full = (code - 64.0) / 876.0
    else:
        full = code / 1023.0
    return np.rint(np.clip(full, 0.0, 1.0) * 65535.0).astype(np.uint16)


# --------------------------------------------------------------------------- #
# ffmpeg plumbing
# --------------------------------------------------------------------------- #
UP = "spline+accurate_rnd+full_chroma_int"


def probe(ff, path):
    r = subprocess.run([ff, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    txt = r.stderr
    vline = next((l for l in txt.splitlines() if "Video:" in l), None)
    if not vline:
        sys.exit(f"ERROR: no video stream found in {path}\n{txt[-500:]}")
    m = re.search(r"[\s,](\d{2,5})x(\d{2,5})[\s,\[]", vline)
    f = re.search(r"(\d+(?:\.\d+)?)\s*fps", vline)
    if not m or not f:
        sys.exit(f"ERROR: could not parse size/fps from: {vline}")
    matrix = "bt709"
    if re.search(r"smpte170m|bt470bg", vline):
        matrix = "bt601"
    return dict(w=int(m.group(1)), h=int(m.group(2)), fps=f.group(1),
                matrix=matrix, has_audio="Audio:" in txt, vline=vline.strip())


def reader_cmd(ff, src, info, deband, thr, matrix):
    m = "bt709" if matrix == "bt709" else "smpte170m"
    vf = []
    if deband:
        vf += [f"scale=flags={UP}", "format=yuv444p16le",
               f"deband=1thr={thr}:2thr={thr}:3thr={thr}:range=20:blur=1"]
    vf += [f"scale=in_range=tv:out_range=pc:in_color_matrix={m}:flags={UP}", "format=rgb48le"]
    return [ff, "-v", "error", "-i", str(src), "-an", "-vf", ",".join(vf),
            "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]


def writer_cmd(ff, out, info, src_for_audio, range_mode, w, h):
    if range_mode == "legal":
        vf = f"scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags={UP},format=yuv444p10le"
    else:
        vf = f"scale=in_range=pc:out_range=pc:out_color_matrix=bt709:flags={UP},format=yuv444p10le"
    cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le",
           "-s", f"{w}x{h}", "-r", str(info["fps"]), "-i", "-"]
    if src_for_audio and info["has_audio"]:
        cmd += ["-i", str(src_for_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "pcm_s16le", "-shortest"]
    cmd += ["-vf", vf, "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le",
            "-vendor", "apl0", "-colorspace", "bt709", str(out)]
    return cmd


# --------------------------------------------------------------------------- #
# Main conversion
# --------------------------------------------------------------------------- #
def convert(args):
    ff = find_ffmpeg()
    src = Path(args.input)
    if not src.exists():
        sys.exit(f"ERROR: input not found: {src}")
    out = Path(args.output) if args.output else src.with_name(f"{src.stem}_SLog3_SGamut3Cine.mov")
    info = probe(ff, src)
    matrix_name = info["matrix"] if args.matrix == "auto" else args.matrix
    w, h = info["w"], info["h"]
    print(f"[info] {src.name}: {w}x{h} @ {info['fps']} fps, matrix={matrix_name}, audio={info['has_audio']}")
    print(f"[info] gamma={args.gamma} exposure={args.exposure:+.2f} stops, deband={not args.no_deband}, "
          f"range={args.range}, grain={args.grain}")

    lut = build_eotf_lut(args.gamma)
    mat = rec709_to_sgamut3cine_matrix().astype(np.float32)
    gain = 2.0 ** args.exposure
    rng = np.random.default_rng(12345)

    rerr = tempfile.TemporaryFile()
    werr = tempfile.TemporaryFile()
    reader = subprocess.Popen(reader_cmd(ff, src, info, not args.no_deband, args.deband_thr, matrix_name),
                              stdout=subprocess.PIPE, stderr=rerr, bufsize=w * h * 6)
    writer = subprocess.Popen(writer_cmd(ff, out, info, src, args.range, w, h),
                              stdin=subprocess.PIPE, stderr=werr)
    nbytes = w * h * 6
    n, t0 = 0, time.time()
    try:
        while args.frames == 0 or n < args.frames:
            buf = reader.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            rgb16 = np.frombuffer(buf, dtype=np.uint16).reshape(h, w, 3)
            frame = to_slog3_frame(rgb16, lut, mat, gain, args.range, rng, not args.no_dither, args.grain)
            writer.stdin.write(frame.tobytes())
            n += 1
            if n % 10 == 0:
                el = time.time() - t0
                print(f"[run] {n} frames, {n / el:.2f} fps", file=sys.stderr, flush=True)
    finally:
        try:
            writer.stdin.close()
        except Exception:
            pass
        reader.stdout.close()
        reader.terminate()
        writer.wait()
        reader.wait()
    for name, f in (("reader", rerr), ("writer", werr)):
        f.seek(0)
        msg = f.read().decode(errors="replace").strip()
        if msg and writer.returncode != 0:
            print(f"[{name} stderr] {msg}", file=sys.stderr)
    if writer.returncode != 0 or n == 0:
        sys.exit("ERROR: conversion failed")
    print(f"[done] {n} frames -> {out}  ({time.time() - t0:.1f}s)")
    print("[note] Untagged S-Gamut3.Cine / S-Log3. In Resolve: Data levels = Video, "
          "then verify on the 10-bit scope: black ~95, 18% grey ~420, 90% white ~598. "
          "If levels look off by 64/876, re-run with --range full.")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest():
    ok = True
    m = rec709_to_sgamut3cine_matrix()
    print("Rec.709 -> S-Gamut3.Cine matrix:\n", np.round(m, 6))
    print("row sums (must be 1.0):", np.round(m.sum(axis=1), 6))
    ok &= bool(np.allclose(m.sum(axis=1), 1.0, atol=1e-6))

    # Reference points from Sony's S-Log3 definition (10-bit code values)
    ref = {0.0: 95.0, 0.18: 420.0, 0.90: 598.0}
    for lin, code in ref.items():
        got = float(slog3_oetf(np.array([lin], dtype=np.float32))[0] * 1023.0)
        flag = abs(got - code) < 1.0
        ok &= flag
        print(f"S-Log3({lin:>5}) = {got:7.2f}  expected {code:6.1f}  {'OK' if flag else 'FAIL'}")

    # Display grey patches whose linear value is 0, 0.18, 0.9 under gamma 2.4
    lut = build_eotf_lut("2.4")
    mat = m.astype(np.float32)
    rng = np.random.default_rng(1)
    disp = [0.0, 0.18 ** (1 / 2.4), 0.90 ** (1 / 2.4)]
    h, w = 64, 192
    img = np.zeros((h, w, 3), dtype=np.uint16)
    for i, d in enumerate(disp):
        img[:, i * 64:(i + 1) * 64, :] = int(round(d * 65535))
    for mode in ("legal", "full"):
        fr = to_slog3_frame(img, lut, mat, 1.0, mode, rng, False, 0.0)
        # codes expected in the *file*: legal -> Y code == S-Log3 code
        ff = find_ffmpeg()
        info = dict(fps="1", has_audio=False)
        with tempfile.TemporaryDirectory() as td:
            outp = Path(td) / "t.mov"
            wp = subprocess.run(writer_cmd(ff, outp, info, None, mode, w, h), input=fr.tobytes(),
                                capture_output=True)
            if wp.returncode != 0:
                print("writer failed:", wp.stderr.decode()[-400:])
                ok = False
                continue
            rp = subprocess.run([ff, "-v", "error", "-i", str(outp), "-f", "rawvideo",
                                 "-pix_fmt", "yuv444p10le", "-"], capture_output=True)
            raw = np.frombuffer(rp.stdout, dtype=np.uint16)[: h * w].reshape(h, w)  # Y plane
            codes = [float(raw[h // 2, i * 64 + 32]) for i in range(3)]
        exp = [95.0, 420.0, 598.0] if mode == "legal" else [95.0, 420.0, 598.0]
        # in 'full' mode the file is full-range: Y code == code too
        good = all(abs(c - e) <= 2.0 for c, e in zip(codes, exp))
        ok &= good
        print(f"file readback [{mode}] Y codes = {codes}  expected ~{exp}  {'OK' if good else 'FAIL'}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description="8-bit Rec.709 -> 10-bit S-Gamut3.Cine/S-Log3 ProRes 4444")
    p.add_argument("input", nargs="?")
    p.add_argument("-o", "--output")
    p.add_argument("--gamma", default="2.4", choices=["2.4", "2.2", "srgb"],
                   help="display gamma of the source (default 2.4)")
    p.add_argument("--exposure", type=float, default=0.0, help="stops added before the log curve")
    p.add_argument("--no-deband", action="store_true")
    p.add_argument("--deband-thr", type=float, default=0.02)
    p.add_argument("--no-dither", action="store_true")
    p.add_argument("--grain", type=float, default=0.0,
                   help="gaussian luma grain, sigma in 10-bit code values (try 1.0-2.5)")
    p.add_argument("--range", default="legal", choices=["legal", "full"])
    p.add_argument("--matrix", default="auto", choices=["auto", "bt709", "bt601"])
    p.add_argument("--frames", type=int, default=0, help="only process the first N frames (0 = all)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.input:
        p.error("input is required (or use --selftest)")
    convert(a)


if __name__ == "__main__":
    main()
