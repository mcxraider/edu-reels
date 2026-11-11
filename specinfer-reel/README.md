# How SpecInfer works — Instagram Reel (1080×1920, 30 fps)

An explainer for **SpecInfer: tree-based speculative decoding** ([Miao et al., arXiv:2305.09781](https://arxiv.org/abs/2305.09781)).
All visuals are drawn with the Canvas 2D API. The narration is OpenAI TTS (`gpt-4o-mini-tts`, voice `cedar`), and the on-screen disclosure reads "AI-generated narration".

```
specinfer-reel/
  index.html          deterministic draw(t) animation + live preview (play / scrub)
  render.mjs          Puppeteer -> draw(i/30) -> PNG pipe -> FFmpeg H.264 yuv420p -> mux audio
  generate_audio.py   OpenAI TTS -> FFmpeg assembly -> forced alignment -> timings.json
  narration.txt       the exact script (one sentence/paragraph per blank-line block)
  timings.json        measured sentence + word timestamps, scene assignment, animation events, QA
  assets/voiceover.wav           final narration (1 s lead-in, scene pauses, -16 LUFS, 5 s end hold)
  output/specinfer_explained.mp4 final reel
```

## Commands

```bash
# install
npm install                                   # puppeteer-core (uses the installed Google Chrome; set CHROME_PATH otherwise)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export OPENAI_API_KEY=...                     # or put it in .env here / in the parent folder

# audio + timings (TTS, assembly, forced alignment)
.venv/bin/python generate_audio.py --speed 0.93          # npm run audio
.venv/bin/python generate_audio.py --align-only          # re-measure timings.json only

# preview
node render.mjs --inject && open index.html              # Space = play/pause, ←/→ = ±1 s, ?t=60 to start at 60 s
node render.mjs --stills 30 75 100                       # PNG stills -> output/stills/

# render
node render.mjs --qa                                     # determinism check
node render.mjs                                          # -> output/specinfer_explained.mp4
EXTRA_HOLD=1 node render.mjs                             # optional: lengthen the final title card further
```

## How sync works

1. **TTS**: `narration.txt` goes out verbatim in one Speech API request, giving one continuous take. A take with a dropped or garbled sentence (each sentence needs at least 60% of its words recognised) is regenerated automatically. This happened during development: the model sometimes drops the last sentence.
2. **Assembly** (FFmpeg): a 1.0 s lead-in, then 0.5 s pauses inserted inside existing silence at scene boundaries. Two-pass linear `loudnorm` brings it to -16 LUFS / -1.5 dBTP, with a limiter as a no-clip guard. A 5 s end hold is added. The speech itself is never time-stretched.
3. **Forced alignment** (the Speech API returns no timestamps, so none are invented):
   * whisper-1 word timestamps are taken on about 20 s windows cut inside measured pauses;
   * the script words are aligned to them with Needleman–Wunsch;
   * edges are snapped to RMS-measured pauses;
   * the word match rate is reported as a verbatim QA check (99.4% for the shipped take).
4. **Time-warp**: `index.html` keys every animation to narration cues, e.g. `cue(18, 'disagrees')` is the moment that word is spoken. Regenerating the audio at a different pace keeps everything in sync.
5. **Frame count**: `ceil((ffprobe duration of voiceover.wav + EXTRA_HOLD) × 30)`. Nothing is hardcoded.

## The worked example (identical in tree, mask, verification and output)

Prompt `The animal`; draft tree (packed in DFS order):

* `ran → through → the`
* `sat → on → the → floor`
* `jumped → over → a`

One target pass with the tree attention mask produces a next-token prediction at every node. Greedy verification walks from the root:

1. At the root the target predicts `sat` = draft `sat` ✓, so the `ran` and `jumped` branches are rejected.
2. At `sat` it predicts `on` ✓.
3. At `on` it predicts `the` ✓.
4. At `the` it predicts `mat`, but the draft says `floor` ✗.

So the output gains 3 accepted draft tokens plus 1 correction token from the target: **The animal sat on the mat**. The rejected branches and the rest of the draft are discarded.

The mask row for `on` makes the core idea visible: it attends to the prompt, `sat` and itself, never to `ran / through / the` or `jumped / over / a`. An ordinary causal mask over the packed sequence would leak those earlier-packed sibling tokens.

## Accuracy notes

* The timeline in scene 7 is explicitly marked illustrative; no benchmark numbers are shown or implied.
* SpecInfer's tree comes from small speculative models (SSMs), not Medusa-style extra heads on the target.
* Greedy verification guarantees the target model's own output. The paper extends this to stochastic decoding with multi-step speculative sampling.
