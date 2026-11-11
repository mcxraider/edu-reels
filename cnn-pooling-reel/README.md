# CNN Max Pooling — Educational Instagram Reel

A 1080 × 1920, 30 fps, ~85 s quiz Reel. It's an ElevenLabs **Adam** narration over deterministic
HTML Canvas 2D animation. The video never reveals the answer.

| File | Role |
|---|---|
| `script.txt` | Exact narration script (source of truth for TTS and the alignment check) |
| `generate_audio.py` | ElevenLabs `/with-timestamps` call. Writes `narration.mp3`, `alignment.json`, `timings.json` |
| `timings.json` | Words, sentences (audio + video start/end, scene), option blocks, named cues, time-warp, total/frames |
| `index.html` | Canvas 2D animation: pure `draw(t)`, plus a live preview with play/pause, seek and sound |
| `render.mjs` | Puppeteer → `draw(t)` per frame → PNG piped to ffmpeg. Builds the audio and muxes `final_reel.mp4` |
| `narration.mp3` | Raw single-take Adam voiceover |
| `final_reel.mp4` | Finished H.264/AAC MP4 with `+faststart` |

## Requirements

- Python 3.9+ (standard library only)
- Node.js 18+ with `npm install` (installs `puppeteer-core`)
- Google Chrome. Default path: `/Applications/Google Chrome.app/...`. Override it with `CHROME_PATH`.
- `ffmpeg` / `ffprobe` on `PATH`

## Reproduce

```bash
export ELEVENLABS_API_KEY=sk_...        # never commit this
python3 generate_audio.py              # 1 API call → narration.mp3, alignment.json, timings.json
npm install
node render.mjs                        # → final_reel.mp4 (+ output/audio_final.m4a, output/silent_video.mp4)
```

Other commands:

- `python3 generate_audio.py --retime` recomputes `timings.json` from the saved alignment, with no API call.
- `node render.mjs --stills 15 30 80` writes PNG previews to `output/stills/`.
- `node render.mjs --audio` builds only the final audio track. The live preview plays this track.

## Live preview

Run `node render.mjs --audio`, then serve the folder, for example with `python3 -m http.server 8000`.
Open `http://localhost:8000/index.html` in Chrome. The controls are Play/Pause (also the Space key),
a seek bar and a sound toggle. Opening `index.html` directly from disk also works, because
`render.mjs --inject` embeds the timings into the page.

## Timing and sync

- ElevenLabs character alignment is the only timing source. Words come from character spans and
  sentences from punctuation. Nothing is estimated from character counts. The script checks that
  the joined alignment characters equal `script.txt` exactly.
- **Time warp.** The video adds 1 s of lead-in. It also inserts short silences at the midpoints of
  the narrator's natural pauses (`PAUSES_AFTER`), which gives each animation time to land. The same
  warp is applied in three places:
  - `index.html`, through `warp()`
  - `render.mjs`, which cuts the audio, inserts the silences, delays the start, pads the end and
    normalises loudness to −14 LUFS / −1.5 dBTP
  - `timings.json`, which stores `video_start` / `video_end` for each sentence
- **Cues.** Named cues in `timings.json` drive the key animation beats:
  - "two-by-two max pooling": the pooling window appears.
  - "stride of two": the window jumps two cells.
  - "half its original height and width": the H/W dimension brackets appear.
  - "a feature shifts slightly": the 9 slides one cell.
  - "remain unchanged": the 9 = 9 comparison appears.
  - Each answer option: its card appears with a neutral highlight while it is being read.
- The total length is `max(80, ceil(narration_end + 9 s))`, capped at 90 s. The frame count is
  `total × 30`. The narration speed is 1.1× because Adam at 1.0× runs about 81 s, which leaves no
  room for the thinking period.

## Maths shown (asserted in `index.html`)

- A 4×4 map `[[1,3,2,4],[5,6,1,2],[2,1,8,3],[4,2,5,7]]` with a 2×2 window, stride 2 and no padding
  gives `[[6,4],[4,8]]`. That is 16 → 4 positions.
- The shift demo moves a 9 from (0,0) to (0,1). It stays inside the same top-left region, so both
  pooled maps are `[[9,2],[1,3]]`. The video presents this as limited local tolerance, not full
  translation invariance.
- The Scene 1 feature map comes from a real valid 3×3 edge filter plus ReLU applied to the 10×10 digit.
