# How DeepSeek Sparse Attention works — animated reel

A ~142 s, 1080×1920, 30 fps Instagram Reel explaining DeepSeek Sparse Attention (DSA) from DeepSeek-V3.2.
Every frame is drawn with plain Canvas 2D. The voiceover is OpenAI `gpt-4o-mini-tts`. Puppeteer captures the
frames and FFmpeg encodes them.

**Output:** `output/deepseek_sparse_attention.mp4` (H.264 high / yuv420p + AAC 48 kHz, −14 LUFS).

## Files

| File | Role |
|---|---|
| `script.txt` | The exact narration, one sentence or paragraph per block. It is the single source of truth for TTS, alignment and subtitles. |
| `generate_audio.py` | OpenAI TTS (`gpt-4o-mini-tts`, voice `cedar`, alternative `onyx`) → `narration.wav` |
| `align_audio.py` | Forced alignment of `narration.wav` against `script.txt` → `timings.json` |
| `timings.json` | Word and sentence timings, detected pauses, the video timeline (lead-in, tail hold, total) and QA stats |
| `index.html` | The deterministic `draw(t)` animation plus a live preview (play/pause, seek, scene jump) |
| `render.mjs` | Builds the audio, renders every frame with Puppeteer, streams the frames into FFmpeg and muxes the MP4 |
| `package.json`, `requirements.txt` | Dependencies |

## Setup

```bash
# Python (3.10+)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Node (18+), plus a local Google Chrome or Chromium
npm install
# FFmpeg must be on PATH
```

Environment variables (these can also go in `.env` in this folder or its parent):

| Variable | Needed for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `generate_audio.py`, `align_audio.py` | Required |
| `CHROME_PATH` | `render.mjs` | Defaults to `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` |
| `MUSIC` | `render.mjs` | Optional path to a music bed. It is mixed at −28 dB and side-chain-ducked under the voice. Off by default. |

## Commands

```bash
.venv/bin/python generate_audio.py              # cedar, speed 1.0 (try --voice onyx, --speed 1.05)
.venv/bin/python align_audio.py                 # -> timings.json, prints the word-match rate per run
node render.mjs --inject                        # embed timings.json into index.html (works offline via file://)
node render.mjs --audio                         # -> output/audio_final.m4a (also used by the preview)
node render.mjs --stills 12 45 90               # PNG stills -> output/stills/
node render.mjs                                 # full render -> output/deepseek_sparse_attention.mp4
```

**Preview:** open `index.html` directly in Chrome. The timings are embedded, so the animation runs without a server.
The preview plays audio only when `output/audio_final.m4a` exists and the browser allows the file to load. If audio
is not available, playback falls back to a silent wall-clock timer. Alternatively, run `npm run preview` and open
`http://localhost:8000`.

Controls: Play/Pause (Space), ←/→ to step 2 s, a seek bar and a scene selector. Add `?t=63` to the URL to open at a
given second. The preview calls the same `draw(t)` function as the renderer, so a preview frame matches the rendered
frame at the same timestamp.

## Architecture

**Audio → timings.** OpenAI TTS returns no timestamps, so `align_audio.py` derives them from the audio itself:

1. `whisper-1` produces approximate word times. These serve only as timing anchors; the transcript text is never used.
2. The script's own words are aligned to the recognised words with Needleman–Wunsch. Script words that do not match,
   such as numerals or "V3.2", are interpolated between their matched neighbours.
3. Sentence edges are snapped to pauses measured in a 10 ms RMS envelope of the waveform.

The current take matches 98.3% of script words. The six unmatched words are spelling or number variants, and the
recognised text is otherwise verbatim. Subtitles show the exact script text, not the transcript.

**Timeline / time warp.** The video clock is defined as lead-in (1.0 s) + narration time + a 3.2 s hold on the final
frame. Each of the nine scenes covers a contiguous range of sentences. A scene boundary falls in the middle of the
pause between its last sentence and the next scene's first sentence. The crossfade (±0.36 s) therefore happens while
nobody is speaking.

Inside a scene, each animation beat is keyed to a specific spoken word through `cue(sentence, word)`. For example, the
ReLU bars animate at the word "ReLU", and the softmax column appears at the word "weights". If the voice or speed
changes, rerunning the alignment re-times the whole video.

**Determinism.** `draw(t)` is a pure function of `t`. It keeps no state between frames and never reads the wall clock.
All illustrative data comes from a seeded PRNG evaluated once at load, and all motion uses eased windows (`ss`, `eio`,
`eo`).

**Rendering.** `render.mjs` opens 6–7 headless Chrome pages and has each page render every n-th frame. A reorder
buffer streams the PNG frames to FFmpeg strictly in frame order (`image2pipe` → libx264 CRF 18, yuv420p, CFR 30), so
no temporary image files are written. The audio is built by delaying the narration 1 s, running two-pass `loudnorm`
to −14 LUFS / −1.5 dBTP, applying fades and padding to the total length, then encoding AAC. Finally the audio and
video are muxed into the MP4.

## Scenes

| # | Narration | Visual |
|---|---|---|
| 1 | 128K context, dense query | 320-cell context grid with a 128,000 counter, and a query fanning out to all earlier tokens |
| 2 | "But here's the problem…" | A causal attention triangle grows, O(N²) appears, and the few high-relevance cells sharpen |
| 3 | DSA intro and the indexer | A dense triangle morphs to sparse per-query selections; the pipeline Query → Lightning Indexer → Top-K → Sparse Attention → Output appears stage by stage |
| 4 | How the indexer works | k/v tokens, query scan, 4 indexer heads showing dot product → ReLU → × w, Σ over heads, the formula, and a score for every token |
| 5 | Top-K selection | 48 scored sample tokens, of which 10 are kept; a strip shows their scattered positions, plus k = 2,048 and an "illustrative" note |
| 6 | Main attention | Selected MLA latent KV entries are retrieved, then q·k → softmax → weighted v → output, with the o_t formula and the indexer-vs-attention role split |
| 7 | Savings | Dense card: O(N²), 128,000 keys/query. DSA card: indexer (all pairs, sage) vs. main attention (k per row), O(N·k), 2,048 keys/query |
| 8 | The catch | The indexer scans all preceding tokens, then sparse attention runs on the selection; real-speed factors and a conceptual (not measured) cost bar |
| 9 | Takeaway | The pipeline lights up once more, then "Spend expensive attention compute where it matters." held for about 3 s |

## Technical notes and sources

- The indexer score is I(t,s) = Σⱼ w(t,j) · ReLU(q(t,j) · k(s)). It is learned token-level scoring: DeepSeek-V3.2
  Technical Report, §2.1 (DeepSeek-AI, 2025).
- DeepSeek-V3.2 config: `index_topk = 2048`, `index_n_heads = 64`, `index_head_dim = 128`, built on MLA
  (`kv_lora_rank = 512`). Source: [`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json).
  The video shows 4 heads for legibility.
- DSA is implemented on MLA in MQA mode, so each selected latent KV entry is shared across query heads.
- O(N²) → O(N·k) refers to the main attention only. The indexer is still O(N²) over all pairs, but it is much cheaper
  per pair (few small heads, FP8). Real speedups depend on kernels, memory traffic and hardware. See, for example,
  [NVIDIA's TensorRT-LLM notes](https://nvidia.github.io/TensorRT-LLM/1.4.0rc3/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.html).
- All scores, weights and probabilities on screen are illustrative, not measurements.
- The narration voice is AI-generated, and this is disclosed on screen throughout.
