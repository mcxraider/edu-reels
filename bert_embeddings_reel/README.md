# How BERT Embeddings Actually Work — Instagram Reel

A 1080 × 1920, 30 fps, ~114 s educational Reel. The narration is ElevenLabs **Adam**, in one take, over a
deterministic HTML Canvas 2D animation. The renderer uses Puppeteer and FFmpeg. No AI-generated imagery or stock footage.

| File | Role |
|---|---|
| `narration.txt` | Exact narration script. It is the source of truth for TTS and the alignment check |
| `generate_audio.py` | ElevenLabs `/v1/text-to-speech/{voice}/with-timestamps` (stdlib only). Writes `narration.mp3` and `alignment.json` |
| `alignment.json` | Raw character alignment (`raw_response_alignment`) plus the derived timeline: words, sentences (audio + video time, scene), scene windows, subtitle phrases, cues and time-warp |
| `index.html` | Canvas 2D animation built around a pure `draw(t)`, plus a live preview with play/pause, seek and sound |
| `render.mjs` | Puppeteer calls `draw(t)` for each frame and pipes the PNGs to FFmpeg (H.264). It also builds and normalises the audio and muxes `final_reel.mp4` |
| `narration.mp3` | Raw Adam voiceover |
| `final_reel.mp4` | Finished H.264 / AAC MP4, yuv420p, CFR 30 fps, `+faststart` |

## Requirements

- Python 3.9+ (standard library only)
- Node.js 18+. Run `npm install` to get `puppeteer-core`.
- Google Chrome at `/Applications/Google Chrome.app/...`, or set `CHROME_PATH`
- `ffmpeg` / `ffprobe` on `PATH`

## Environment

```bash
export ELEVENLABS_API_KEY=sk_...   # required for generate_audio.py; never commit it
```

`generate_audio.py` also reads `ELEVENLABS_API_KEY` from `./.env` or `../.env` when it is not already set.
The key never appears in code.

## Run

```bash
python3 generate_audio.py          # 1 API call → narration.mp3 + alignment.json
npm install
node render.mjs                    # → final_reel.mp4 (+ output/audio_final.m4a, output/silent_video.mp4)
```

Other commands:

- `python3 generate_audio.py --retime` rebuilds the timeline from the saved alignment, with no API call.
- `node render.mjs --stills 20 50 90` writes preview PNGs to `output/stills/`.
- `node render.mjs --audio` builds only the processed audio track, which the preview uses.
- `node render.mjs --inject` embeds the timeline into `index.html` so the `file://` preview works.

## Preview

Run `node render.mjs --audio` once, then open `index.html` in Chrome. It plays in real time with the
processed narration. Controls:

- Play/Pause button, also the Space key
- Seek bar
- ←/→ jumps 5 s
- Sound toggle

If Chrome blocks file audio, serve the folder (`python3 -m http.server`) and open
`http://localhost:8000/index.html`. The preview and the MP4 call the same `draw(t)` with the same timeline.

## Timing and sync

- **Alignment.** The ElevenLabs character timestamps are the only timing source. `generate_audio.py` asserts
  that the joined alignment characters equal the narration text exactly. Words come from character spans,
  sentences from `. ? !` boundaries, and cues from exact phrase lookups. Nothing is estimated from word counts.
- **Paragraph breaks.** Paragraphs are sent to TTS joined by single newlines. This shortens the pauses and
  keeps the reel under 120 s, but the words are identical.
- **Speed.** `speed = 1.12` is a small speed-up. At 1.04 the take ran 124.5 s, which is too long.
- **Time warp.** `video = 1 s lead-in + narration time + Σ inserted pauses before it`. The extra silences
  (0.3–0.5 s) sit at the midpoints of natural pauses at scene transitions. The same warp drives:
  - `index.html`, through `warp()`
  - the FFmpeg audio graph in `render.mjs`, which cuts the take and inserts silence
  - `video_start` / `video_end` in `alignment.json`
- **Scene windows.** Each window runs from the midpoint before its first sentence to the next window,
  using the sentence→scene map. Every scene fades in and out at its window edges.
- **Audio.**
  - 1 s lead-in, inserted pauses, then a ~3 s silent tail that holds the final visual.
  - Two-pass `loudnorm` to −16 LUFS integrated / −1.5 dBTP.
  - 48 kHz stereo AAC.
- **Duration.** Read from `alignment.json` (`warp.total`) and from `window.TOTAL` in the page. It is never
  hardcoded, and `render.mjs` refuses to render anything over 120 s.
- **Subtitles.**
  - Drawn on the canvas from the alignment-based phrase list: one phrase at a time, at most 2 lines.
  - Placed at y ≈ 1430–1560, above Instagram's bottom UI and below every diagram card, which end at y ≤ 1380.
  - The box is shifted left of the right-hand action buttons.
  - Technical keywords are terracotta, and words not yet spoken are dimmed.

## Technical accuracy notes

- **Tokens and IDs.** These are real `bert-base-uncased` WordPiece outputs: `bank` = 2924, `deposited` = 14140,
  and `embeddings` → `em ##bed ##ding ##s`.
- **Input embeddings.** Input = token + position + token-type, summed element-wise. The on-screen numbers
  really add up, and a self-check asserts it. Token type is shown as segment A/B (which sentence), and the
  frame says explicitly that it is not part of speech. LayerNorm follows the sum.
- **Encoder.** BERT-base has 12 layers with 12 heads each. Every layer is multi-head self-attention
  followed by a position-wise feed-forward network, each with a residual connection and Add & Norm.
- **Attention.**
  - Weights are labelled *illustrative, not measured from a trained model*.
  - Each row is a softmax distribution that sums to 1 (asserted).
  - Q/K/V are learned projections, and the formula `softmax(q·kᵀ/√d)` weights the values.
- **Contextual vectors.** The two `bank` input embeddings are identical because the token and position are
  the same. The outputs diverge layer by layer. The 2-D projection is labelled *illustrative*.
- **Sentence embeddings.**
  - Mean pooling (`(1/n) Σ hᵢ`) and the final `[CLS]` hidden state are shown as two separate strategies,
    distinct from token-level vectors.
  - Vanilla BERT's pre-training objectives are shown as MLM and NSP, which are not a similarity objective.
  - Sentence-BERT is a Siamese setup with shared weights and pooling, fine-tuned on sentence pairs and
    compared with cosine similarity.
