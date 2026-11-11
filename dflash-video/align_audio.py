#!/usr/bin/env python3
"""Forced alignment of the narration against the exact script -> timings.json.

The TTS endpoint returns no timestamps, so timing is measured from the audio itself:

1. Recognition anchors: OpenAI whisper-1 (verbose_json, word granularity) gives word times.
   The recognised TEXT is never used for captions - only its timestamps.
2. Script alignment: the script's own words are aligned to the recognised words with
   Needleman-Wunsch dynamic programming over normalised tokens. Matched words take the anchor
   time; unmatched runs are spread between neighbouring matches.
3. Acoustic refinement: a 10 ms RMS envelope finds real pauses; segment starts/ends are snapped to
   the nearest pause edge (within 0.6 s) because recogniser boundaries drift at phrase edges.
4. QA: the word match rate (script vs recognised audio) is the verbatim-narration check.

    python align_audio.py             # aligns assets/narration.wav -> timings.json
    python align_audio.py --check     # also fail (exit 1) if match rate < 95 %

timings.json: segments[] {id, scene, text, start, end, words[] {w, start, end, matched}},
scenes[] {id, start, end}, pauses[], duration, qa.  Times are absolute in the final audio.
"""
import json, re, sys, wave, pathlib
import numpy as np
from generate_audio import load_env, read_script, scene_of

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "output"


def norm(w):
    w = w.lower().replace("’", "'")
    w = {"dflash": "dflash", "d": "d", "eagle3": "eagle3"}.get(w, w)
    return re.sub(r"[^a-z0-9]", "", w)


def transcribe(path, cache_name, chunk_s=20.0):
    """whisper-1 word timestamps, transcribed in ~20 s chunks cut in the middle of measured pauses.
    Whisper occasionally skips a phrase in long files; short clips are far more reliable."""
    cache = OUT / cache_name
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        return json.loads(cache.read_text())
    ps, dur = pauses(path, min_len=0.3)
    cuts, last = [0.0], 0.0
    for a, b in ps:
        mid = (a + b) / 2
        if mid - last >= chunk_s and dur - mid > 5:
            cuts.append(round(mid, 3)); last = mid
    cuts.append(dur)
    words, texts = [], []
    for t0, t1 in zip(cuts, cuts[1:]):
        w = transcribe_window(path, t0, t1, cache_name)
        words += w["words"]; texts.append(w["text"])
    res = {"text": " ".join(t.strip() for t in texts), "words": words, "chunks": cuts}
    OUT.mkdir(exist_ok=True)
    cache.write_text(json.dumps(res, indent=1))
    return res


def transcribe_window(wav, t0, t1, cache_name):
    """Re-transcribe one window of the audio in isolation (whisper sometimes skips a phrase in long files)."""
    import subprocess, tempfile
    cache = OUT / f"{cache_name}.gap_{t0:.2f}_{t1:.2f}.json"
    if cache.exists() and cache.stat().st_mtime > wav.stat().st_mtime:
        return json.loads(cache.read_text())
    load_env()
    from openai import OpenAI
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(wav),
                        tmp.name], check=True)
        with open(tmp.name, "rb") as f:
            r = OpenAI().audio.transcriptions.create(model="whisper-1", file=f, response_format="verbose_json",
                                                     timestamp_granularities=["word"], language="en",
                                                     prompt="DFlash, speculative decoding, block-diffusion, EAGLE-3, GPU, autoregressively.")
    res = {"text": r.text, "words": [{"w": w.word, "start": w.start + t0, "end": w.end + t0} for w in r.words]}
    cache.write_text(json.dumps(res, indent=1))
    return res


NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
       "8": "eight", "9": "nine", ".": "point"}


def expand(rec):
    """Whisper writes numerals ("6", "2.5") and may merge "speed up"; spell them out so the
    verbatim check compares spoken words. Sub-word times are spread across the original word."""
    out = []
    for k, w in enumerate(rec):
        # word granularity drops the decimal point: "2.5" arrives as "2" "5" -> two point five
        if (re.fullmatch(r"\d", w["w"].strip()) and k + 1 < len(rec) and re.fullmatch(r"\d", rec[k + 1]["w"].strip())
                and abs(rec[k + 1]["start"] - w["end"]) < 0.3):
            out.append({"w": NUM[w["w"].strip()], "start": w["start"], "end": w["end"]})
            out.append({"w": "point", "start": w["end"], "end": rec[k + 1]["start"]})
            continue
        t = w["w"].strip().strip(",.?!")
        parts = None
        if re.fullmatch(r"\d(\.\d)?x?", t) and not (k > 0 and norm(rec[k - 1]["w"]) == "eagle"):
            parts = [NUM[c] for c in t if c in NUM]
        elif t.lower() == "speed-up":
            parts = ["speedup"]
        if not parts:
            out.append(w); continue
        d = (w["end"] - w["start"]) / len(parts)
        out += [{"w": p, "start": w["start"] + i * d, "end": w["start"] + (i + 1) * d} for i, p in enumerate(parts)]
    return out


def match_words(sw, rec):
    """Align at sub-token level: whisper splits "Multi-Token" into "Multi" "Token"."""
    sub, parent = [], []
    for k, (_, w) in enumerate(sw):
        for part in [p for p in re.split(r"[-‐-—]", w) if norm(p)]:
            sub.append(norm(part)); parent.append(k)
    rec = expand(rec)
    pairs = nw(sub, [norm(w["w"]) for w in rec])
    N = len(sw)
    st = np.full(N, np.nan); en = np.full(N, np.nan)
    need = np.bincount(parent, minlength=N); got = np.zeros(N, int)
    for i, j in pairs.items():
        k = parent[i]; got[k] += 1
        st[k] = rec[j]["start"] if np.isnan(st[k]) else min(st[k], rec[j]["start"])
        en[k] = rec[j]["end"] if np.isnan(en[k]) else max(en[k], rec[j]["end"])
    partial = got < need                     # only fully matched words count as anchors
    st[partial] = np.nan; en[partial] = np.nan
    return st, en


def nw(a, b, gap=-1, mis=-1, match=2):
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1)); P = np.zeros((n + 1, m + 1), dtype=np.int8)
    S[:, 0] = np.arange(n + 1) * gap; S[0, :] = np.arange(m + 1) * gap
    P[1:, 0] = 1; P[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mis)
            u = S[i - 1, j] + gap; l = S[i, j - 1] + gap
            k = int(np.argmax([d, u, l])); S[i, j] = (d, u, l)[k]; P[i, j] = k
    i, j, pairs = n, m, {}
    while i > 0 or j > 0:
        k = P[i, j]
        if i > 0 and j > 0 and k == 0:
            if a[i - 1] == b[j - 1]: pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and (k == 1 or j == 0): i -= 1
        else: j -= 1
    return pairs


def pauses(path, thr_db=-40, min_len=0.16):
    with wave.open(str(path)) as w:
        sr = w.getframerate(); ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if ch > 1: x = x.reshape(-1, ch).mean(1)
    hop = int(sr * 0.01); n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    quiet = 20 * np.log10(rms) < thr_db
    out, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]: j += 1
            if (j - i) * 0.01 >= min_len: out.append([round(i * 0.01, 2), round(j * 0.01, 2)])
            i = j
        else: i += 1
    return out, len(x) / sr


def align(wav, cache_name="whisper_final.json"):
    paras = read_script()
    tr = transcribe(wav, cache_name)
    rec = [w for w in tr["words"] if norm(w["w"])]
    sw = [(si, w) for si, s in enumerate(paras) for w in s.split()]
    N = len(sw)
    st, en = match_words(sw, rec)
    # gap repair: any run of >= 2 unmatched script words is re-transcribed as an isolated window
    ok = ~np.isnan(st); repaired = []
    i = 0
    while i < N:
        if not ok[i]:
            j = i
            while j < N and not ok[j]: j += 1
            if j - i >= 2:
                # window spans the matched neighbours themselves: their anchors are often the
                # stretched, unreliable words (e.g. one "word" lasting 3 s) around the skipped phrase
                t0 = max(0.0, (st[i - 1] if i > 0 else 0.0) - 0.4)
                t1 = (en[j] if j < N else rec[-1]["end"] + 1.0) + 0.4
                win = transcribe_window(wav, t0, t1, cache_name)
                rec = [w for w in rec if not (t0 < w["start"] < t1)] + [w for w in win["words"] if norm(w["w"])]
                rec.sort(key=lambda w: w["start"]); repaired.append([round(t0, 2), round(t1, 2)])
            i = j
        else: i += 1
    if repaired:
        st, en = match_words(sw, rec)
    ok = ~np.isnan(st)
    i = 0
    while i < N:                                   # spread unmatched runs between matched neighbours
        if not ok[i]:
            j = i
            while j < N and not ok[j]: j += 1
            a = en[i - 1] if i > 0 else 0.0
            b = st[j] if j < N else (en[i - 1] + 0.4 * (j - i))
            for k in range(i, j):
                st[k] = a + (b - a) * (k - i) / (j - i); en[k] = a + (b - a) * (k - i + 1) / (j - i)
            i = j
        else: i += 1
    ps, dur = pauses(wav)
    segs = []
    for si, s in enumerate(paras):
        ks = [k for k, (q, _) in enumerate(sw) if q == si]
        s0, s1 = float(st[ks[0]]), float(en[ks[-1]])
        for a, b in ps:
            if a <= s0 < b: s0 = b
            if a < s1 <= b: s1 = a
        c = [p[1] for p in ps if abs(p[1] - s0) < 0.6 and p[1] < s1]
        if c: s0 = min(c, key=lambda v: abs(v - s0))
        c = [p[0] for p in ps if abs(p[0] - s1) < 0.6 and p[0] > s0]
        if c: s1 = min(c, key=lambda v: abs(v - s1))
        words = [{"w": sw[k][1], "start": round(float(st[k]), 3), "end": round(float(en[k]), 3), "matched": bool(ok[k])}
                 for k in ks]
        words[0]["start"] = max(words[0]["start"], round(s0, 3))
        segs.append({"id": f"seg{si:02d}", "scene": scene_of(si), "text": s,
                     "start": round(s0, 3), "end": round(s1, 3), "words": words})
    for a, b in zip(segs, segs[1:]):
        if b["start"] < a["end"]: a["end"] = b["start"] = round((a["end"] + b["start"]) / 2, 3)
    unmatched = [sw[k][1] for k in range(N) if not ok[k]]
    return {"duration": round(dur, 3), "pauses": ps, "sentences": segs,
            "qa": {"script_words": N, "matched_words": int(ok.sum()), "match_rate": round(float(ok.mean()), 4),
                   "unmatched_script_words": unmatched, "repaired_windows": repaired,
                   "recognised_text": tr["text"]}}


def main():
    wav = ROOT / "output" / "narration.wav"
    al = align(wav)
    segs = al["sentences"]
    scenes = []
    for s in segs:
        if not scenes or scenes[-1]["id"] != s["scene"]:
            scenes.append({"id": s["scene"], "first_segment": s["id"], "speech_start": s["start"]})
        scenes[-1]["speech_end"] = s["end"]
    # scene boundaries: midpoint of the silence between scenes; first starts at 0, last ends at duration
    for i, sc in enumerate(scenes):
        sc["start"] = 0.0 if i == 0 else round((scenes[i - 1]["speech_end"] + sc["speech_start"]) / 2, 3)
    for i, sc in enumerate(scenes):
        sc["end"] = al["duration"] if i == len(scenes) - 1 else scenes[i + 1]["start"]
    meta = json.loads((OUT / "tts_meta.json").read_text()) if (OUT / "tts_meta.json").exists() else {}
    res = {"source": "output/narration.wav", "duration": al["duration"], "fps": 30,
           "method": "whisper-1 word anchors + Needleman-Wunsch alignment to narration.txt + RMS pause snapping",
           "tts": meta, "qa": al["qa"], "scenes": scenes, "pauses": al["pauses"], "segments": segs}
    (ROOT / "timings.json").write_text(json.dumps(res, indent=1))
    q = al["qa"]
    print(f"duration {al['duration']:.2f}s, {len(segs)} segments, word match {q['match_rate']:.1%}")
    print("unmatched script words:", q["unmatched_script_words"])
    for sc in scenes: print(f"  {sc['id']:18s} {sc['start']:7.2f} -> {sc['end']:7.2f}")
    for s in segs: print(f"  {s['id']} {s['start']:7.2f} {s['end']:7.2f}  {s['text'][:64]}")
    if "--check" in sys.argv and q["match_rate"] < 0.95:
        sys.exit("match rate below 95 %: narration may not be verbatim")


if __name__ == "__main__":
    main()
