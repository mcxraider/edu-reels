# Cost — BERT Embeddings Reel

| Item | Value |
|---|---|
| Date | 2026-11-11 |
| Balance before run | $454 |
| Balance after run | $432 |
| **Cost of this run** | **$22** |

The figures were reported by the user after the run. Earlier messages said "$330 – $432", then "$440 – $432", then "$448 – $432". The user corrected them to "$454 – $432", and this file uses that final figure.

## What the run did

- **ElevenLabs calls:** 2 calls to `/v1/text-to-speech/{Adam}/with-timestamps`, model `eleven_multilingual_v2`.
  1. Speed 1.04, 1,780 characters. The result was 124.5 s, over the 120 s limit, so it was discarded. A copy is kept in `output/narration_v1_104.mp3`.
  2. Speed 1.12, 1,760 characters (paragraph breaks joined by single newlines). This take became `narration.mp3`.
  - Total sent: about 3,540 characters.
- **Local work:** the Puppeteer, Chrome and FFmpeg rendering costs nothing beyond compute.
- **No Higgsfield or image-generation credits** were used.

## Re-running

- `python3 generate_audio.py --retime` and `node render.mjs` make no API calls.
- Only `python3 generate_audio.py` without `--retime` makes a paid ElevenLabs call (about 1,760 characters).
