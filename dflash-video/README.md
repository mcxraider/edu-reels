# DFlash explained — educational Instagram reel

A ~177 s, 1080 × 1920, 30 fps explainer of **DFlash: Block Diffusion for Flash Speculative Decoding**
(Chen, Liang & Liu, 2026 — [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)), built entirely from
code: Canvas 2D animation → Puppeteer frame capture → ffmpeg (H.264 + AAC), narrated with OpenAI
`gpt-4o-mini-tts` (voice `cedar`). The narration is AI-generated; this is disclosed in the header
badge for the entire video and on the end card.

Output: `output/dflash_explained.mp4`

## Files

| File | Role |
|---|---|
| `narration.txt` | The verbatim script, one sentence per paragraph (41 sentences). |
| `generate_audio.py` | One coherent TTS take → pause tightening → lead-in, scene pauses, loudnorm, outro → `output/narration.wav`. |
| `align_audio.py` | Forced alignment → `timings.json` (sentence + word timestamps, scene bounds, QA). |
| `timings.json` | Measured timings on the **final** `output/narration.wav`. |
| `index.html` | Canvas renderer: `draw(t)`, eight scenes, captions, live preview with play/scrub. |
| `render.mjs` | Puppeteer → `draw(i/30)` → PNG piped into ffmpeg → mux narration. |

## Install

```bash
# Node 18+ and ffmpeg/ffprobe on PATH, Google Chrome installed
npm install                         # puppeteer-core (uses the system Chrome)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # or put it in .env here or in the parent folder (never commit it)
```

`CHROME_PATH` overrides the Chrome binary (default: `/Applications/Google Chrome.app/...`).

## Pipeline

```bash
npm run audio     # TTS (1 request, ~2.9k chars) + assembly          -> output/narration.wav
npm run align     # whisper-1 anchors + alignment, fails if < 95 % verbatim -> timings.json
npm run render    # inject timings, render 5,300 frames, mux          -> output/dflash_explained.mp4
```

Re-assemble without a new TTS call: `.venv/bin/python generate_audio.py --skip-tts`.

### Audio

1. **TTS** — the whole script is sent in one request (scene paragraphs joined) for a single
   consistent take, `speed=1.0`, WAV, with style instructions (calm, technical educator; "D Flash",
   "G P U", "Eagle three").
2. **Pause tightening** — the raw take was 183.2 s. Rather than speeding up speech, silences longer
   than 0.5 s are shortened to 0.5 s by cutting the *middle* of the silence (speech is never cut or
   time-stretched). 183.2 → 170.1 s.
3. **Assembly** — 1.0 s lead-in, +0.35 s at each of the 7 scene boundaries (cut in the middle of the
   natural silence), two-pass `loudnorm` to −16 LUFS / −1.5 dBTP, ~3.2 s ending-card hold → 176.6 s.

### Alignment (`timings.json`)

The Speech API returns no timestamps, so they are measured from the audio:

1. `whisper-1` (verbose_json, word granularity) on ~20 s windows cut inside pauses — **only its
   timestamps are used**, never its text.
2. The script's own words are aligned to the recognised words with Needleman–Wunsch. Numerals
   ("6", "2.5") are spelled out so they compare against "six", "two point five".
3. Unmatched runs ≥ 2 words are re-transcribed in isolation; leftovers are interpolated.
4. Sentence edges are snapped to measured RMS pause edges.
5. QA: `qa.match_rate` = fraction of script words matched in the audio. Current: **100 %** (485/485).

Scenes start at the midpoint of the silence before their first sentence.

### Animation (`index.html`)

- `draw(t)` is pure: every element is a function of `t` and `window.TIMING`; no frame-to-frame state
  (verified by `npm run qa`, which renders 40 timestamps out of order and compares hashes).
- **Time-warp:** keyframes are written as narration cues, e.g. `cue(18, 'single')` = start of the word
  "single" in sentence 18, resolved to measured seconds. Unresolvable cues abort the render.
- Captions are generated from the aligned words (≤ 2 lines, karaoke-style), key terms highlighted.
- Colour semantics: terracotta = target model, muted orange = draft model/tokens, sage = accepted,
  soft grey dashed = mask, muted red = rejected.

## Preview

```bash
npm run preview          # injects timings and serves http://localhost:8080
```

Open `http://localhost:8080/index.html` in Chrome: ▶/❚❚ (space) plays in sync with the narration,
the slider scrubs, ←/→ jump 1 s, `?t=95` opens at 95 s. Opening the file directly also works
(timings are embedded), but some browsers block audio from `file://`.

Stills for review: `node render.mjs --stills 12 64 100` → `output/stills/`.

## Editing

- **Change narration:** edit `narration.txt` (keep one sentence per paragraph) and update
  `SCENE_STARTS` in `generate_audio.py` if sentence indices move, then `audio → align → render`.
  Cues in `index.html` reference sentence indices and words; a missing cue fails loudly.
- **Change visuals:** edit the `sceneN(t)` functions; scene list/titles are in `SCENES`.

## Technical notes (checked against the paper and vLLM Speculators docs)

- Drafter = lightweight block-diffusion model; one forward pass predicts the whole masked block,
  anchored on a clean token from the target (the previous verification's bonus token). Paper: block
  size 16 (10 for LLaMA 3.1), 5 draft layers; the video shows 5 masked positions for clarity.
- Conditioning: hidden states from 5 target layers (shallow → deep) are concatenated, projected and
  RMS-normalised into a context feature that is **injected into the K/V of every draft layer**;
  within a block, attention is bidirectional. Embeddings and LM head are shared with the target.
- Verification is standard speculative decoding by the target: longest valid prefix; under sampling,
  accept with probability min(1, p/q) and resample on rejection — lossless. The video avoids
  presenting verification as a simple string comparison.
- Reported results: "over 6× lossless acceleration" (up to 6.1× on Qwen3-8B) and "up to 2.5× higher
  speedup than EAGLE-3". Shown as reported research results; timeline proportions are illustrative.

## Troubleshooting

- `OPENAI_API_KEY is not set` — export it or add it to `.env`.
- `assembled duration > 180 s` — lower `PAUSE_CAP` in `generate_audio.py` first; only then raise
  `--speed` slightly (e.g. 1.04).
- `match rate below 95 %` — inspect `qa.unmatched_script_words` in `timings.json`; regenerate the take.
  Delete `output/whisper_*.json` to force re-transcription.
- `unresolved narration cues` — a `cue(i, 'word')` in `index.html` no longer matches the script.
- Chrome not found — set `CHROME_PATH`.
- Blank or shifted text in renders — fonts load before the first frame (`document.fonts.ready`); the
  renderer uses the system font stack (SF Pro / Helvetica on macOS).
