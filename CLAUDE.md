# slog3-pipeline

Single-file tool: `to_slog3.py` converts an 8-bit Rec.709 clip (typically AI-generated, H.264 4:2:0)
into a clean 10-bit ProRes 4444 file encoded as S-Gamut3.Cine / S-Log3.

## Commands
- Self-test (run this first, after any change): `python to_slog3.py --selftest`
- Quick test on a few frames: `python to_slog3.py clip.mp4 --frames 20 -o /tmp/test.mov`
- Full run: `python to_slog3.py clip.mp4`
- Dependencies install themselves (numpy via pip; ffmpeg via system or `imageio-ffmpeg` fallback).
  Override ffmpeg with `FFMPEG=/path/to/ffmpeg`.

## Invariants - do not change without a reason and without re-running --selftest
- S-Log3 constants: 0 -> code 95, 0.18 -> 420, 0.90 -> 598 (10-bit). Selftest asserts this.
- S-Gamut3.Cine primaries: R(0.766,0.275) G(0.225,0.800) B(0.089,-0.087), white D65 (0.3127,0.3290).
- All math is float32 between decode (rgb48le) and encode. Never insert an 8-bit step.
- Dither is TPDF at 1 LSB of 10-bit; keep it on by default (it prevents banding in dark shots).
- Output is untagged except `-colorspace bt709` (matrix). Do not tag it as Rec.709 transfer.

## Known assumptions (verify, don't silently "fix")
- Source gamma defaults to 2.4. AI video is often closer to sRGB/2.2 - use `--gamma`.
- `--range legal` stores luma code == S-Log3 code in a legal-range stream. If Resolve's 10-bit scope
  shows black/grey/white away from 95/420/598, try `--range full`.
- This tool cannot add dynamic range or colour that the source never had.

## Performance
Measured ~2.7 s/frame at 5084x2160 (numpy, single process): a 10 s / 300-frame clip takes ~13 min.
Use `--frames` for tests. If speed matters, the obvious next step is splitting frames across a
`multiprocessing.Pool` (the per-frame function `to_slog3_frame` is pure and thread/process safe
apart from the RNG, which should get a per-worker seed).

## Verified on the real clip (10 frames)
Output: ProRes 4444, 5084x2160, PCM audio. ffprobe may report the pixel format as yuv444p12le;
that is normal for ProRes 4444 (10-bit data in a 12-bit-capable container).
Luma codes on the test clip: 0.1% = 97, median = 133, 99.9% = 559, max = 614 (a dark shot; whites
sit near 610 because display white 1.0 maps to S-Log3 ~0.596).
