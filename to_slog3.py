#!/usr/bin/env python3
"""
to_slog3.py - 8-bit Rec.709 video -> 10-bit S-Gamut3.Cine / S-Log3 ProRes 4444.

Pipeline (all heavy math in float32, nothing is quantised to 8-bit in between):
  1. Decode with high-quality 4:2:0 -> 4:4:4 chroma upsampling, 16-bit.
  2. Optional deband (ffmpeg `deband`) on the 16-bit 4:4:4 signal.
  3. 16-bit RGB -> linear light (Gamma 2.4 / 2.2 / sRGB), optional exposure.
  4. Rec.709 primaries -> S-Gamut3.Cine primaries (3x3 matrix derived from xy).
  5. Sony S-Log3 OETF.
  6. RGB -> BT.709 YCbCr + range scaling in float32 (not swscale, which has a gain error).
  7. TPDF dither (+ optional grain) at 10-bit LSB, then the one and only quantisation to 10-bit.
  8. Encode ProRes 4444 yuv444p10le, audio copied as PCM. Frames run in parallel (--workers).

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


# BT.709 luma coefficients (the file's YCbCr matrix, not the colour primaries)
KR, KB = 0.2126, 0.0722
KG = 1.0 - KR - KB
# 10-bit code limits per range: (Y lo, Y hi, C lo, C hi, chroma scale relative to luma scale)
RANGES = {
    "legal": (64.0, 940.0, 64.0, 960.0, 896.0 / 876.0),
    "full": (0.0, 1023.0, 0.0, 1023.0, 1.0),
}


def tpdf(rng, shape):
    """Triangular-PDF noise, +-1 LSB peak, zero mean."""
    return rng.random(shape, dtype=np.float32) - rng.random(shape, dtype=np.float32)


def to_slog3_frame(rgb16, lut, matrix, gain, range_mode, rng, dither, grain):
    """uint16 HxWx3 full-range display RGB -> uint16 3xHxW planar yuv444p10le (Y, Cb, Cr).

    The RGB -> YCbCr matrix and range scaling are done here in float32, so the only
    quantisation in the whole pipeline is the final rint() to 10-bit, with the dither
    applied exactly at that step. (Letting swscale do it from rgb48le costs a 65535/65280
    gain error in legal range: +2 codes at 598, +3 near the top.)

    Both ranges store luma code == S-Log3 code (95 / 420 / 598):
      legal: Y = 64 + 876*Y', E' = (code - 64)/876  ->  Y = Kr*R + Kg*G + Kb*B  (in codes)
      full:  Y = 1023*Y',     E' = code/1023        ->  same
    Chroma: C = 512 + s*(B-Y)/1.8556 (Cb), 512 + s*(R-Y)/1.5748 (Cr), s = 896/876 or 1.
    """
    lin = lut[rgb16]                                   # float32, linear Rec.709
    if gain != 1.0:
        lin *= np.float32(gain)
    lin = lin @ matrix.T.astype(np.float32)            # -> S-Gamut3.Cine primaries
    code = slog3_oetf(lin) * np.float32(1023.0)        # 10-bit S-Log3 code values (float)
    ylo, yhi, clo, chi, cs = RANGES[range_mode]
    np.clip(code, ylo, yhi, out=code)
    r, g, b = code[..., 0], code[..., 1], code[..., 2]
    y = np.float32(KR) * r + np.float32(KG) * g + np.float32(KB) * b
    cb = (b - y) * np.float32(cs / (2.0 * (1.0 - KB))) + np.float32(512.0)
    cr = (r - y) * np.float32(cs / (2.0 * (1.0 - KR))) + np.float32(512.0)
    h, w = y.shape
    if dither:
        y += tpdf(rng, (h, w))
        cb += tpdf(rng, (h, w))
        cr += tpdf(rng, (h, w))
    if grain > 0:
        y += rng.standard_normal((h, w), dtype=np.float32) * np.float32(grain)
    out = np.empty((3, h, w), dtype=np.uint16)
    out[0] = np.rint(np.clip(y, ylo, yhi))
    out[1] = np.rint(np.clip(cb, clo, chi))
    out[2] = np.rint(np.clip(cr, clo, chi))
    return out


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
    f = parse_rate(vline)
    if not m or not f:
        sys.exit(f"ERROR: could not parse size/fps from: {vline}")
    matrix = "bt709"
    if re.search(r"smpte170m|bt470bg", vline):
        matrix = "bt601"
    return dict(w=int(m.group(1)), h=int(m.group(2)), fps=f,
                matrix=matrix, has_audio="Audio:" in txt, vline=vline.strip())


def parse_rate(vline):
    """Frame rate from ffmpeg's stream line. Prefer 'tbr' (the stream's real frame rate) over
    'fps' (the *average*, which a trimmed/stream-copied clip skews, e.g. 29.56 for a 30 fps file)."""
    fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", vline)
    tbr = re.search(r"(\d+(?:\.\d+)?)(k?)\s*tbr", vline)
    if tbr and not tbr.group(2):
        rate = tbr.group(1)
        if fps and abs(float(fps.group(1)) - float(rate)) > 0.01:
            print(f"[warn] average fps {fps.group(1)} != stream rate {rate} tbr; using {rate}. "
                  f"Override with --fps if needed.", file=sys.stderr)
    elif fps:
        rate = fps.group(1)
    else:
        return None
    return exact_fps(rate)


def exact_fps(s):
    """ffmpeg prints NTSC rates rounded ('23.98'); return the exact rational ('24000/1001')."""
    v = float(s)
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    n = round(v * 1.001)
    if n > 0 and abs(v - n / 1.001) < 0.006:
        return f"{n * 1000}/1001"
    return s


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
    """Input is already yuv444p10le from numpy: no scaling/conversion here, only encode + tag."""
    rng = "tv" if range_mode == "legal" else "pc"
    cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv444p10le",
           "-color_range", rng, "-colorspace", "bt709",
           "-s", f"{w}x{h}", "-r", str(info["fps"]), "-i", "-"]
    if src_for_audio and info["has_audio"]:
        cmd += ["-i", str(src_for_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "pcm_s16le", "-shortest"]
    cmd += ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le",
            "-vendor", "apl0", "-colorspace", "bt709", "-color_range", rng, str(out)]
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
    if args.fps:
        info["fps"] = args.fps
    matrix_name = info["matrix"] if args.matrix == "auto" else args.matrix
    w, h = info["w"], info["h"]
    print(f"[info] {src.name}: {w}x{h} @ {info['fps']} fps, matrix={matrix_name}, audio={info['has_audio']}")
    print(f"[info] gamma={args.gamma} exposure={args.exposure:+.2f} stops, deband={not args.no_deband}, "
          f"range={args.range}, grain={args.grain}")

    workers = args.workers if args.workers > 0 else default_workers()
    params = dict(gamma=args.gamma, gain=2.0 ** args.exposure, range_mode=args.range,
                  dither=not args.no_dither, grain=args.grain, seed=args.seed, shape=(h, w))
    print(f"[info] workers={workers}")

    rerr = tempfile.TemporaryFile()
    werr = tempfile.TemporaryFile()
    reader = subprocess.Popen(reader_cmd(ff, src, info, not args.no_deband, args.deband_thr, matrix_name),
                              stdout=subprocess.PIPE, stderr=rerr, bufsize=w * h * 6)
    writer = subprocess.Popen(writer_cmd(ff, out, info, src, args.range, w, h),
                              stdin=subprocess.PIPE, stderr=werr)
    nbytes = w * h * 6
    n, t0 = 0, time.time()

    def frames():
        i = 0
        while args.frames == 0 or i < args.frames:
            buf = reader.stdout.read(nbytes)
            if len(buf) < nbytes:
                return
            yield i, buf
            i += 1

    def emit(data):
        nonlocal n
        writer.stdin.write(data)
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"[run] {n} frames, {n / el:.2f} fps", file=sys.stderr, flush=True)

    pool = None
    try:
        if workers == 1:
            _worker_init(params)
            for item in frames():
                emit(_worker_frame(item))
        else:
            import multiprocessing as mp
            from collections import deque
            pool = mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(params,))
            pending = deque()                     # bounded: never read far ahead of the writer
            for item in frames():
                pending.append(pool.apply_async(_worker_frame, (item,)))
                while len(pending) >= 2 * workers:
                    emit(pending.popleft().get())
            while pending:
                emit(pending.popleft().get())
    finally:
        if pool is not None:
            pool.terminate()
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
    el = time.time() - t0
    print(f"[done] {n} frames -> {out}  ({el:.1f}s, {el / n:.2f} s/frame)")
    print("[note] Untagged S-Gamut3.Cine / S-Log3. The file stores luma codes 95 / 420 / 598 for "
          "black / 18% grey / 90% white, like a Sony camera. In Resolve (Data Levels = Video) that should "
          "read ~3.5% / 41% / 61% on the waveform. --range full only matters with Data Levels = Full.")


# --------------------------------------------------------------------------- #
# Worker (pure per-frame function; runs in-process or in a multiprocessing Pool)
# --------------------------------------------------------------------------- #
_W = {}


def default_workers():
    """Leave one core for ffmpeg; cap at 4 (measured ~4.4 GB total with 3 workers at 5084x2160)."""
    return max(1, min((os.cpu_count() or 2) - 1, 4))


def _worker_init(params):
    _W.clear()
    _W.update(params)
    _W["lut"] = build_eotf_lut(params["gamma"])
    _W["mat"] = rec709_to_sgamut3cine_matrix().astype(np.float32)


def _worker_frame(item):
    """(index, rgb48le bytes) -> yuv444p10le bytes. RNG seeded by (seed, index), so the
    output is identical for any number of workers."""
    i, buf = item
    h, w = _W["shape"]
    rgb16 = np.frombuffer(buf, dtype=np.uint16).reshape(h, w, 3)
    rng = np.random.default_rng([_W["seed"], i])
    return to_slog3_frame(rgb16, _W["lut"], _W["mat"], _W["gain"], _W["range_mode"], rng,
                          _W["dither"], _W["grain"]).tobytes()


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

    # exact NTSC rates
    for s_in, exp_fps in (("23.98", "24000/1001"), ("29.97", "30000/1001"), ("59.94", "60000/1001"),
                          ("24", "24"), ("25", "25"), ("12.5", "12.5")):
        got = exact_fps(s_in)
        flag = got == exp_fps
        ok &= flag
        print(f"exact_fps({s_in!r}) = {got!r}  expected {exp_fps!r}  {'OK' if flag else 'FAIL'}")

    # frame rate from ffmpeg stream lines (tbr wins over the average 'fps')
    lines = {
        "Video: hevc (Main), yuv420p(tv, progressive), 5084x2160, 51703 kb/s, 29.56 fps, 30 tbr, 15360 tbn": "30",
        "Video: h264 (High), yuv420p(tv, bt709), 1920x1080, 23.98 fps, 23.98 tbr, 24k tbn": "24000/1001",
        "Video: h264 (High), yuv420p, 1280x720, 25 fps, 25 tbr, 12800 tbn": "25",
        "Video: prores, yuv444p12le, 1920x1080, 30 fps": "30",
    }
    for line, exp_fps in lines.items():
        got = parse_rate(line)
        flag = got == exp_fps
        ok &= flag
        print(f"parse_rate(...{line[-34:]!s}) = {got!r}  expected {exp_fps!r}  {'OK' if flag else 'FAIL'}")

    # Patches: display grey whose linear value is 0 / 0.18 / 0.90 under gamma 2.4, then
    # saturated red, green, blue and a skin-ish tone, to test chroma scaling end to end.
    lut = build_eotf_lut("2.4")
    mat = m.astype(np.float32)
    rng = np.random.default_rng(1)
    g = lambda v: v ** (1 / 2.4)
    patches = [(0.0,) * 3, (g(0.18),) * 3, (g(0.90),) * 3,
               (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.85, 0.62, 0.50)]
    pw, h = 32, 32
    w = pw * len(patches)
    img = np.zeros((h, w, 3), dtype=np.uint16)
    for i, rgb in enumerate(patches):
        img[:, i * pw:(i + 1) * pw, :] = np.rint(np.array(rgb) * 65535).astype(np.uint16)
    # reference S-Log3 RGB codes, straight from the colour math (no YCbCr, no file)
    ref_rgb = slog3_oetf(lut[img] @ mat.T) * 1023.0
    ff = find_ffmpeg()
    ffprobe = shutil.which("ffprobe", path=os.path.dirname(ff)) or shutil.which("ffprobe")
    for mode in ("legal", "full"):
        fr = to_slog3_frame(img, lut, mat, 1.0, mode, rng, False, 0.0)
        info = dict(fps="24000/1001", has_audio=False)
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
            yuv = np.frombuffer(rp.stdout, dtype=np.uint16)[: 3 * h * w].reshape(3, h, w).astype(np.float64)
            tags = None
            if ffprobe:
                pr = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
                                     "-read_intervals", "%+#1", "-show_entries",
                                     "frame=color_space,color_range:stream=r_frame_rate",
                                     "-of", "default=nw=1", str(outp)], capture_output=True, text=True)
                tags = dict(l.split("=", 1) for l in pr.stdout.split() if "=" in l)
        # 1) grey: Y code == S-Log3 code exactly, chroma neutral
        Y = [int(yuv[0, h // 2, i * pw + pw // 2]) for i in range(3)]
        C = [int(yuv[c, h // 2, i * pw + pw // 2]) for c in (1, 2) for i in range(3)]
        good = Y == [95, 420, 598] and all(c == 512 for c in C)
        ok &= good
        print(f"file readback [{mode}] grey Y = {Y} (expect [95, 420, 598]), "
              f"Cb/Cr = {sorted(set(C))} (expect [512])  {'OK' if good else 'FAIL'}")
        # 2) colour: decode the file's YCbCr back to S-Log3 RGB codes, compare to reference
        _, _, _, _, cs = RANGES[mode]
        Yf, Cb, Cr = yuv[0], (yuv[1] - 512.0) / cs, (yuv[2] - 512.0) / cs
        R = Yf + 2 * (1 - KR) * Cr
        B = Yf + 2 * (1 - KB) * Cb
        G = (Yf - KR * R - KB * B) / KG
        err = np.abs(np.stack([R, G, B], -1) - ref_rgb).max()
        good = err <= 1.5
        ok &= good
        print(f"file readback [{mode}] colour patches, max RGB code error = {err:.2f} (<= 1.5)  "
              f"{'OK' if good else 'FAIL'}")
        # 3) tags: matrix bt709 in the bitstream, exact NTSC rate
        if tags is None:
            print(f"file tags [{mode}] skipped (no ffprobe next to {ff})")
        else:
            good = tags.get("color_space") == "bt709" and tags.get("r_frame_rate") == "24000/1001"
            ok &= good
            print(f"file tags [{mode}] {tags}  {'OK' if good else 'FAIL'}")
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
    p.add_argument("--workers", type=int, default=0,
                   help="parallel processes (0 = auto: cores-1, max 4; 1 = single process)")
    p.add_argument("--seed", type=int, default=12345, help="dither/grain RNG seed")
    p.add_argument("--fps", help="force the output frame-rate tag (e.g. 30 or 30000/1001); "
                                 "never drops or duplicates frames")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.input:
        p.error("input is required (or use --selftest)")
    convert(a)


if __name__ == "__main__":
    main()
