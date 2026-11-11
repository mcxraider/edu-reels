#!/usr/bin/env python3
"""Forced alignment of narration.wav against the known exact script -> timings.json.

OpenAI TTS returns no timestamps, so this stage derives them from the audio itself:

1. Recognition anchors: whisper-1 (verbose_json, word granularity) gives approximate word times.
   The transcript is used ONLY as timing anchors - its text is never trusted.
2. Script alignment: the script's own words are aligned to the recognised words with a
   Needleman-Wunsch DP (normalised tokens; numbers like "2,048" vs "two thousand and forty-eight"
   become gaps). Every script word gets a time: matched words take the anchor time, gaps are
   interpolated between neighbouring matches.
3. Acoustic refinement: a 10 ms RMS energy envelope of the waveform locates real pauses. Each
   sentence start/end is snapped to the edge of the nearest detected pause (within 0.6 s), because
   recogniser word boundaries drift at phrase edges.
4. Report: word-level match rate (script vs. recognised audio) as a verbatim-narration check.

Output timings.json: sentences[] {i, text, start, end, words[] {w, start, end, matched}}, plus
pauses[], duration, and qa.  Times are relative to narration.wav (lead-in is added in the page).
"""
import json, os, re, sys, wave, pathlib
import numpy as np
from generate_audio import load_env, read_script

ROOT = pathlib.Path(__file__).resolve().parent
LEAD_IN, TAIL = 1.0, 3.2


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower().replace("’", "'"))


def transcribe(path):
    cache = ROOT / "output" / "whisper_words.json"
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        return json.loads(cache.read_text())
    from openai import OpenAI
    with open(path, "rb") as f:
        r = OpenAI().audio.transcriptions.create(model="whisper-1", file=f, response_format="verbose_json",
                                                 timestamp_granularities=["word"], language="en",
                                                 prompt=" ".join(read_script())[:800])
    words = [{"w": w.word, "start": w.start, "end": w.end} for w in r.words]
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"text": r.text, "words": words}, indent=1))
    return {"text": r.text, "words": words}


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


def pauses(path, thr_db=-38, min_len=0.18):
    with wave.open(str(path)) as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    hop = int(sr * 0.01); n = len(x) // hop
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms); quiet = db < thr_db
    out, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]: j += 1
            if (j - i) * 0.01 >= min_len: out.append([round(i * 0.01, 2), round(j * 0.01, 2)])
            i = j
        else: i += 1
    return out, len(x) / sr


def main():
    load_env()
    wav = ROOT / "narration.wav"
    sents = read_script()
    tr = transcribe(wav)
    rec = [w for w in tr["words"] if norm(w["w"])]
    script_words = [(si, w) for si, s in enumerate(sents) for w in s.split()]
    pairs = nw([norm(w) for _, w in script_words], [norm(w["w"]) for w in rec])
    N = len(script_words)
    st = np.full(N, np.nan); en = np.full(N, np.nan)
    for i, j in pairs.items(): st[i], en[i] = rec[j]["start"], rec[j]["end"]
    idx = np.arange(N); ok = ~np.isnan(st)
    st = np.interp(idx, idx[ok], st[ok]); en = np.interp(idx, idx[ok], en[ok])
    # unmatched runs: spread evenly between surrounding matched words
    i = 0
    while i < N:
        if not ok[i]:
            j = i
            while j < N and not ok[j]: j += 1
            a = en[i - 1] if i > 0 else 0.0; b = st[j] if j < N else en[-1]
            for k in range(i, j):
                f0, f1 = (k - i) / (j - i), (k - i + 1) / (j - i)
                st[k], en[k] = a + (b - a) * f0, a + (b - a) * f1
            i = j
        else: i += 1
    ps, dur = pauses(wav)
    out = []
    for si, s in enumerate(sents):
        ks = [k for k, (q, _) in enumerate(script_words) if q == si]
        s0, s1 = float(st[ks[0]]), float(en[ks[-1]])
        for a, b in ps:                      # boundary inside a pause -> pause edge
            if a <= s0 < b: s0 = b
            if a < s1 <= b: s1 = a
        # snap start to end of nearest preceding pause, end to start of nearest following pause
        c = [p[1] for p in ps if abs(p[1] - s0) < 0.6]
        if c: s0 = min(c, key=lambda v: abs(v - s0))
        c = [p[0] for p in ps if abs(p[0] - s1) < 0.6 and p[0] > s0]
        if c: s1 = min(c, key=lambda v: abs(v - s1))
        out.append({"i": si, "text": s, "start": round(s0, 3), "end": round(s1, 3),
                    "words": [{"w": script_words[k][1], "start": round(float(st[k]), 3), "end": round(float(en[k]), 3),
                               "matched": bool(ok[k])} for k in ks]})
    for a, b in zip(out, out[1:]):          # keep monotonic, non-overlapping
        if b["start"] < a["end"]: a["end"] = b["start"] = round((a["end"] + b["start"]) / 2, 3)
    rate = float(ok.mean())
    unmatched = [script_words[k][1] for k in range(N) if not ok[k]]
    res = {"source": "narration.wav", "duration": round(dur, 3), "method":
           "whisper-1 word anchors + Needleman-Wunsch alignment to script.txt + RMS pause snapping",
           "qa": {"script_words": N, "matched_words": int(ok.sum()), "match_rate": round(rate, 4),
                  "unmatched_script_words": unmatched, "recognised_text": tr["text"]},
           "pauses": ps,
           # page timeline: 1 s lead-in silence, narration, then a ~3 s hold of the final frame
           "timeline": {"lead_in": LEAD_IN, "tail_hold": TAIL, "fps": 30,
                        "total": round(LEAD_IN + dur + TAIL, 3)},
           "sentences": out}
    (ROOT / "timings.json").write_text(json.dumps(res, indent=1))
    print(f"duration {dur:.2f}s, {len(out)} sentences, word match {rate:.1%}")
    print("unmatched:", unmatched)
    for s in out: print(f"{s['i']:2d} {s['start']:7.2f} {s['end']:7.2f}  {s['text'][:70]}")


if __name__ == "__main__":
    main()
