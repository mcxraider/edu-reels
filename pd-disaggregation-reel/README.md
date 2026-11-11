# Why LLM Servers Separate Prefill and Decode

A ~176 s vertical (1080 × 1920, 30 fps) Instagram Reel explaining **prefill–decode disaggregation** in LLM inference serving.
Every visual is drawn with native Canvas 2D in `index.html`. Narration is OpenAI `gpt-4o-mini-tts` (voice `onyx`).
Rendering uses Puppeteer → FFmpeg (H.264 + AAC).

| File | Purpose |
|---|---|
| `script.txt` | Exact narration (verbatim from the brief, one paragraph per line block) |
| `generate_audio.py` | TTS + forced alignment → `audio/narration.wav`, `timings.json` |
| `timings.json` | Word- and sentence-level narration timestamps plus QA report |
| `index.html` | Deterministic `draw(t)` animation + live Chrome preview |
| `render.mjs` | Injects timings, builds the final audio, renders frames with Puppeteer, pipes PNGs into FFmpeg |
| `output.mp4` | Final video |

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
npm install                      # puppeteer-core; uses the installed Google Chrome
export OPENAI_API_KEY=sk-...     # or put it in .env (git-ignored); never hard-coded
```

Requires FFmpeg on `PATH`. Set `CHROME_PATH` if Chrome is not at the default macOS location.

## Generate narration

```bash
.venv/bin/python generate_audio.py --speed 1.30 --max-pause 0.35
```

1. **TTS.** The script (~3.6k characters) is synthesised one scene per request (8 scenes, plus one extra split before
   "The result?"; see below). All splits are at paragraph boundaries. Each take is transcribed with `gpt-4o-transcribe`
   and diffed against `script.txt`. A take that drops or changes words is regenerated, up to `--tries` times.
   Takes are joined with a 0.42 s paragraph pause, so there are no audible seams.
   `--scenes 3,4` regenerates only those segments and reuses the cached others in `audio/scenes/`.
2. **Pause cap.** `--max-pause 0.35` shortens silences longer than 0.35 s. Only dead air is removed, never speech.
3. **Alignment.** `whisper-1` word timestamps are aligned to the script with Needleman–Wunsch. Unmatched words are interpolated.
   Sentence edges snap to pauses in the RMS envelope. Results go to `timings.json`. A second, independent `gpt-4o-transcribe`
   pass diffs the whole take against the script and stores the result under `qa.verbatim_check`.

`--align-only` re-runs step 3 on the existing audio.

**Why not one request:** a single 3.6k-character request repeatedly omitted "But disaggregation isn't free." (6 of 6 takes)
and read "who were" as "who are". Splitting at paragraph boundaries, which the brief allows, fixed both.

## Preview

```bash
node render.mjs --inject && node render.mjs --audio && python3 -m http.server 8000
# open http://localhost:8000/index.html   (Space = play/pause, ←/→ = ±2 s, scene jump menu, ?t=95 to seek)
```

`index.html` also opens directly from `file://` after `--inject`, because the timings are embedded. Audio playback needs `output/audio_final.m4a`.

## Render

```bash
node render.mjs                  # → output.mp4
node render.mjs --qa             # 2 stills per scene + output/qa_contact_sheet.png
node render.mjs --stills 12 96   # stills at given timestamps → output/stills/
MUSIC=bed.mp3 node render.mjs    # optional music bed at −28 dB, sidechain-ducked under the voice
```

For frame `i`, t = i / 30. `draw(t)` is evaluated in 6 parallel headless-Chrome pages. Frames are reordered and streamed as
PNG into FFmpeg (`libx264 -crf 18 -preset medium -pix_fmt yuv420p -profile high`, CFR 30, `+faststart`), so no frame files are stored.

The audio pipeline does the following: 1 s lead-in → two-pass `loudnorm` to −14 LUFS / −1.5 dBTP → short fades → pad through a 3 s final hold → AAC 192 kbps, 48 kHz.
Total duration = lead-in + narration + hold. It is derived from the narration, not a fixed length.

## Timing / sync

The narration is the source of truth. `S(i)`/`E(i)` give sentence start and end times, and `cue(i, word)` gives the instant a word is spoken. Examples:

* Scene boundaries fall in the pause between the last sentence of one scene and the first sentence of the next.
* The KV blocks start moving on the word "**transfers**" ("…after prefill finishes, the system *transfers* the KV cache…").
* The RDMA label appears on "RDMA", output tokens start on "continue", and the moves/stays legend appears on "transferring".
* The scheduling-timeline playheads (scenes 3 and 6) use `warp(t, keys)`: piecewise-linear time-warping between narration anchors.
  For example, the long prefill lands on "arrives", and decode resumes as "successive tokens" is spoken.

## Technical notes (what the diagrams claim)

* Prefill processes all prompt tokens in parallel (matrix × matrix) and is shown as compute-bound. It writes K and V for every token in every layer.
* Decode adds **one token per sequence per step**, and sequences are batched, as the diagram states. At small batch sizes, every step
  re-reads the weights and the KV cache, so decode is shown as memory-bandwidth-bound.
* In disaggregation, the **KV cache** moves from prefill to decode over the network (e.g. RDMA). The weights are resident on both pools and are never transferred.
* The colocated vs disaggregated timelines use the same simulated workload, labelled *hypothetical schedule, not a benchmark*.
  The first token is shown coming from prefill, so TTFT spans the prefill in both layouts. In the disaggregated layout, R's second token
  waits for the KV transfer, which shows the overhead honestly.
* KV size uses the formula 2 × layers × KV heads × head dim × bytes × tokens. The example config is 32 layers, 8 KV heads (GQA), dim 128, FP16,
  giving 128 KiB per token. That is arithmetic, explicitly labelled illustrative, not a measurement.
