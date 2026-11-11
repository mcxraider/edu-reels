# MTP heads & speculative decoding: educational Reel

A 1080 × 1920, 30 FPS Instagram Reel that explains how Multi-Token Prediction (MTP) heads draft future tokens and how speculative decoding verifies them. The animation is plain Canvas 2D. The narration comes from OpenAI `gpt-4o-mini-tts` (voice `cedar`). Puppeteer and FFmpeg render the video.

```
index.html         Canvas animation: deterministic draw(t), one function per scene, plus a live preview
render.mjs         Puppeteer → draw(frame/30) → PNG piped into FFmpeg (libx264, yuv420p) → mux → output/final.mp4
generate_audio.py  OpenAI TTS (verbatim script.txt) + FFmpeg assembly (lead-in, section pauses, loudnorm, outro hold)
align_audio.py     forced alignment of the final audio against script.txt → timings.json
script.txt         exact narration, one paragraph = one narration segment
timings.json       measured segment and word times plus the scene map (generated)
assets/narration.wav  final narration track; assets/narration_raw.wav is the untouched TTS take
output/final.mp4   finished Reel
```

## 1. Configure `OPENAI_API_KEY`

```bash
export OPENAI_API_KEY=sk-...
```
Instead of exporting it, you can put `OPENAI_API_KEY=sk-...` in a `.env` file in this folder or its parent. Both Python scripts read it.

## 2. Install dependencies

You need Python 3.10+, Node 18+, FFmpeg/ffprobe on `PATH`, and Google Chrome.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # openai, numpy
npm install                                                          # puppeteer-core
```
The renderer uses the installed Chrome. If Chrome is not at the default macOS location, set `CHROME_PATH=/path/to/chrome`.

## 3. Generate the narration

```bash
.venv/bin/python generate_audio.py --speed 1.1      # = npm run audio
```
1. **TTS.** The full script (about 3,000 characters) fits in one request, so one continuous take is generated. A longer script would be split at paragraph boundaries and joined sample-exactly. Output: `assets/narration_raw.wav`.
2. **Section boundaries.** The raw take is aligned (same method as step 4) to find the silence between scenes. A 0.45 s pause is inserted there, never inside speech.
3. **Assembly.** Adds a 1.0 s lead-in, applies two-pass `loudnorm` to −16 LUFS / −1.5 dBTP, and adds a 4 s outro hold for the final diagram. Output: `assets/narration.wav`.

The audio is never stretched or padded to a fixed length. The script exits with an error if the assembled audio would exceed 180 s. At speed 1.0 the take came out at 172.6 s, which would have made the reel 181 s, so this build uses `--speed 1.1` and runs **166.3 s**. `output/tts_meta.json` records the settings.

## 4. Obtain and validate alignment timestamps

```bash
.venv/bin/python align_audio.py --check             # = npm run align
```
The TTS endpoint returns no timestamps, so they are measured from the audio:

* **Recognition anchors.** `whisper-1` produces word timestamps on roughly 20 s chunks cut at measured pauses. Short chunks matter because Whisper sometimes skips a phrase in long files. Only the timing is used; the captions come from `script.txt`.
* **Matching.** Needleman–Wunsch alignment matches script words to recognised words. Hyphenated words are matched part by part. Any run of two or more unmatched words is transcribed again as an isolated window.
* **Refinement.** A 10 ms RMS energy envelope locates the real pauses, and segment edges snap to them.
* **Validation.** The word match rate checks that the narration is verbatim. `--check` fails if it is below 95 %. This build: **99.8 %**; the one unmatched word, "pass.", was interpolated.

`timings.json` stores, for every segment: `id`, `scene`, `text`, `start`, `end`, and word times. It also stores `scenes[]` (scene boundaries at the middle of each inter-section silence), `pauses[]` and `qa`.

## 5. Preview locally

```bash
node render.mjs --inject          # embeds timings.json into index.html
python3 -m http.server 8080       # then open http://localhost:8080/index.html
```
The preview plays the animation in sync with `assets/narration.wav`; the animation follows the audio clock.
* Space plays or pauses.
* ← / → seek by 1 s.
* The slider scrubs.
* `?t=72.5` opens at a given timestamp.

Opening the file directly via `file://` also works, apart from audio playback restrictions in some browsers.

## 6. Render the final MP4

```bash
node render.mjs                    # = npm run render → output/final.mp4
node render.mjs --stills 10 75.5   # PNG stills → output/stills/
node render.mjs --qa               # renders 40 timestamps out of order twice; frames must be byte-identical
```
The frame count is `round(timings.duration × 30)`, taken from the measured final audio rather than a fixed 180 s. Each frame calls `draw(i/30)`, and the PNG is piped straight to FFmpeg, so no intermediate files are written. The video is encoded with `libx264 -crf 18 -preset slow -pix_fmt yuv420p`, then muxed with AAC 192 kb/s audio, and the result is ffprobe-checked for both streams.

## 7. Change timings or narration and re-render

* **Narration text.** Edit `script.txt`, keeping one paragraph per narration segment. If paragraphs were added or removed, update `SCENE_STARTS` in `generate_audio.py`. Then run `generate_audio.py`, `align_audio.py --check` and `render.mjs`. Captions, the scene map and every keyframe follow automatically.
* **Animation timing.** Keyframes are not hard-coded seconds. They are narration cues such as `cue(8, 'paris')`, meaning "the moment the word *Paris* starts in segment 8". Change the cue word or add an offset (`cue(8,'paris') + 0.3`). The renderer aborts if a cue word can't be found.
* **Pacing.** Use `--speed`, or change `LEAD_IN`, `SECTION_PAUSE` and `OUTRO_HOLD` in `generate_audio.py`. `generate_audio.py --skip-tts` reassembles from the existing raw take without calling the API.

## Technical notes on what the animation shows

* **Running example.** Prompt *"The capital of France is"*, MTP draft **Paris · . · It**. The target model's greedy predictions are **Paris · . · The**. The first two draft tokens are accepted; *It* is rejected and replaced by the target's *The*. The prediction made after *It* is discarded because it was conditioned on the rejected token.
* **MTP designs differ.** Scene 2 says so on screen: some use independent parallel heads (Medusa-style), others chain sequential modules (DeepSeek-V3-style). The drafter is drawn separately from the target model that verifies.
* **Verification.** Shown as one forward pass over the prompt plus the draft. The prompt is usually already in the KV cache. One pass yields four next-token distributions, and the causal mask keeps each position from seeing later ones. A lower-triangular 8 × 8 mask highlights `v3`: it sees the prompt and *Paris*, but not *It*.
* **Sampling mode.** Uses exact speculative-sampling acceptance: accept with probability `min(1, p/q)` and, on rejection, resample from `norm(max(0, p−q))`. This preserves the target distribution.
* **No invented performance figures.** Speed-up visuals are labelled "illustrative". Total computation is shown increasing; only the number of serial passes falls.
* **Disclosure.** The reel is labelled "AI-generated narration" and "Independent educational explainer". The asterisk is an original geometric motif, not an Anthropic logo.
