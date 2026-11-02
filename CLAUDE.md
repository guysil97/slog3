# slog3-pipeline

Single-file tool: `to_slog3.py` converts an 8-bit Rec.709 clip (typically AI-generated, H.264 4:2:0)
into a clean 10-bit ProRes 4444 file encoded as S-Gamut3.Cine / S-Log3.

## Commands
- Self-test (run this first, after any change): `python to_slog3.py --selftest`
- Quick test on a few frames: `python to_slog3.py clip.mp4 --frames 20 -o /tmp/test.mov`
- Full run: `python to_slog3.py clip.mp4` (parallel by default; `--workers 1` for single process)
- Dependencies install themselves (numpy via pip; ffmpeg via system or `imageio-ffmpeg` fallback).
  Override ffmpeg with `FFMPEG=/path/to/ffmpeg`. The selftest's tag check needs `ffprobe` and is
  skipped (with a message) when only the imageio-ffmpeg binary is available.

## Invariants - do not change without a reason and without re-running --selftest
- S-Log3 constants: 0 -> code 95, 0.18 -> 420, 0.90 -> 598 (10-bit). Selftest asserts this.
- S-Gamut3.Cine primaries: R(0.766,0.275) G(0.225,0.800) B(0.089,-0.087), white D65 (0.3127,0.3290).
- All math is float32 between decode (rgb48le) and encode. Never insert an 8-bit step.
- RGB -> YCbCr (BT.709 matrix) and range scaling happen in numpy; ffmpeg receives yuv444p10le and only
  encodes. Do not move this back into swscale: from rgb48le it scales 16-bit as 255*256 instead of
  65535, a 1.0039 gain error in legal range (+2 codes at 598, +3 near the top). The selftest now
  requires exact codes (95/420/598, chroma 512 on grey) and a colour round-trip within 1.5 codes.
- The only quantisation is the final rint() to 10-bit. Dither is TPDF at 1 LSB of 10-bit on Y, Cb and
  Cr; keep it on by default (it prevents banding in dark shots). Grain is luma only.
- Dither RNG is seeded per frame with (seed, frame index): output is bit-identical for any `--workers`.
- Output is untagged except the matrix: bt709 in the ProRes frame header (`-colorspace bt709` on both
  the rawvideo input and the output). Do not tag it as Rec.709 transfer. Note: the mov `colr` atom is
  not written because primaries/transfer are unspecified, so ffprobe's *stream* line says
  `color_space=unknown`; the *frame* (`-show_frames`) says bt709. That is expected.
- NTSC rates are written exactly (24000/1001 etc.). ffmpeg's banner prints them rounded ("23.98");
  `exact_fps()` restores the rational. Without it the file was 1199/50 fps.

## Known assumptions (verify, don't silently "fix")
- Source gamma defaults to 2.4. AI video is often closer to sRGB/2.2 - use `--gamma`.
- `--range legal` (default) and `--range full` store the *same luma codes* (Y = 95/420/598). They differ
  only in chroma scale (896 vs 1023 per unit) and clip limits. ProRes has no range flag in its
  bitstream, so every decoder reads the file as video range; `--range full` is only coherent if the
  clip's Data Levels are forced to Full in Resolve.
- Expected reading in Resolve (unverified hypothesis, reasoned from Sony's spec, not tested in Resolve):
  with Data Levels = Video the waveform should show black / grey / 90% white at ~3.5% / 41% / 61%
  (10-bit scale ~36 / 416 / 624), the same as genuine Sony S-Log3 footage, because the *file* codes are
  95/420/598. If someone "corrects" this so the scope shows 95/420/598, the file no longer matches real
  camera S-Log3. Check against a real Sony clip before changing anything.
- This tool cannot add dynamic range or colour that the source never had.

## Target project (confirmed by the user)
- The scene the AI shot cuts into: Sony FX6, `CaptureGammaEquation=s-log3-cine`,
  `CaptureColorPrimaries=s-gamut3-cine` -> exactly what this tool writes. Timeline 25 fps.
- The AI clip is 30 fps. Do NOT change the frame rate; the tool keeps the source rate (conform is
  done in Resolve).
- Reference export `log_ex.mov` (Resolve H.264 8-bit, 1080p, 2.84:1 letterbox burned in at Y=16).
  Picture area only (rows 202-877), x4 to 10-bit: 0.1% ~148, median ~296, 99.9% ~500, max ~560.
  That scene was UNDER-exposed on purpose/by accident: it is not an exposure target. Place the AI
  shot by Sony's standard (18% grey -> 420), not by matching this clip's median.
- Noise cannot be measured from that export (H.264 removed it; temporal std = 0).

## Performance
- numpy per 5084x2160 frame: ~2.0 s. ffmpeg reader (deband + spline 4:2:0->4:4:4): ~0.6 s/frame.
  ProRes 4444 encode: ~0.4 s/frame. The reader is single-threaded and sets a ceiling of ~1.6 fps.
- Measured on a 4-core cloud container: `--workers 1` 2.93 s/frame, `--workers 3` 1.95 s/frame
  (ffmpeg competes for the same cores). Expect better scaling on machines with more cores.
- Memory: ~4.4 GB total with 3 workers at 5084x2160. Auto = cores-1, capped at 4.
- Next step if still too slow: ffmpeg `-filter_threads` for the reader, or a lighter upsampler.

## Verified on the real clip (10 frames, BEFORE the swscale gain fix)
Output: ProRes 4444, 5084x2160, PCM audio. ffprobe may report the pixel format as yuv444p12le;
that is normal for ProRes 4444 (10-bit data in a 12-bit-capable container).
Luma codes on the test clip: 0.1% = 97, median = 133, 99.9% = 559, max = 614 (a dark shot; whites
sit near 610 because display white 1.0 maps to S-Log3 ~0.596). With the fix, expect the upper values
~2 codes lower (e.g. max ~611). Re-measure when the clip is available again.
