#!/usr/bin/env python3
"""Narration for "Why LLM Servers Separate Prefill and Decode".

    python generate_audio.py                 # TTS (gpt-4o-mini-tts) -> audio/narration.wav, then align -> timings.json
    python generate_audio.py --speed 1.15    # moderate pace adjustment if the reel would exceed 180 s
    python generate_audio.py --align-only    # re-run the alignment on the existing audio/narration.wav

Stage 1 - TTS.  The script is read VERBATIM from script.txt (paragraphs separated by blank lines).
The whole script (~3.7k chars) is sent as ONE request when it fits MAX_CHARS, so sentence transitions
are natural. Otherwise it is split only at paragraph boundaries and the PCM is concatenated sample-exactly
(same voice / instructions / format), so there is no audible gap.

Optional pause tightening (--max-pause): silences longer than the cap are shortened to the cap. This only
trims dead air between sentences, never speech. Used instead of pushing the TTS speed too far.

Stage 2 - alignment. OpenAI TTS returns no timestamps, so they are derived from the audio:
  1. whisper-1 (verbose_json, word granularity) gives approximate word times (text used only as anchors).
  2. The script's own words are aligned to the recognised words with Needleman-Wunsch; unmatched script
     words are interpolated between matched neighbours.
  3. A 10 ms RMS envelope finds real pauses; sentence boundaries snap to pause edges.
  4. gpt-4o-transcribe independently re-transcribes the take and is diffed against script.txt
     (verbatim check, reported in timings.json -> qa).

Reads OPENAI_API_KEY from the environment (or .env in this folder or its parent). Never hardcoded.
"""
import argparse, difflib, io, json, os, re, sys, wave, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
MODEL = "gpt-4o-mini-tts"
MAX_CHARS = 3900
LEAD_IN, TAIL = 1.0, 3.0

INSTRUCTIONS = (
    "Speak like an experienced machine learning engineer explaining an interesting systems concept. "
    "Maintain a steady, conversational pace. Emphasize technical distinctions naturally, and pause briefly "
    "after important concepts. "
    "Pronounce GPU as 'G P U', LLM as 'L L M', KV as 'K V', TTFT as 'T T F T', RDMA as 'R D M A'. "
    "Say 'prefill' as 'pre-fill' and 'decode' clearly. No exaggerated presentation style."
)


def load_env():
    for p in (ROOT / ".env", ROOT.parent / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$", line)
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def read_script():
    text = (ROOT / "script.txt").read_text(encoding="utf-8").strip()
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk(paras):
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > MAX_CHARS:
            chunks.append(cur); cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def read_wav(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return x, sr


def write_wav(path, x, sr):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.astype(np.int16).tobytes())


def tighten_pauses(x, sr, cap, thr_db=-42):
    """Shorten silent runs longer than `cap` seconds to exactly `cap` (crossfade-free: cut in silence)."""
    hop = int(sr * 0.01); n = len(x) // hop
    f = x[: n * hop].astype(np.float32) / 32768
    db = 20 * np.log10(np.sqrt((f.reshape(n, hop) ** 2).mean(1) + 1e-12))
    keep = np.ones(len(x), bool); i = 0; removed = 0.0
    while i < n:
        if db[i] < thr_db:
            j = i
            while j < n and db[j] < thr_db: j += 1
            L = (j - i) * 0.01
            if L > cap and i > 0 and j < n:          # keep leading/trailing silence as is
                cut = int((L - cap) * sr); mid = (i * hop + j * hop) // 2
                keep[mid - cut // 2: mid - cut // 2 + cut] = False; removed += cut / sr
            i = j
        else:
            i += 1
    return x[keep], removed


# Scene boundaries (paragraph index where each of the 8 scenes starts). Splitting ONLY here.
SCENE_STARTS = [0, 6, 14, 20, 27, 34, 41, 48]
# Request segments = scenes, plus one extra paragraph-boundary split inside scene 3: the TTS model tends to
# read "Users who were already ..." as "who are" in context, so "The result? Users who were ..." is synthesised (and verified)
# on its own. All splits are at paragraph boundaries and joined with the normal paragraph pause.
SEGMENT_STARTS = sorted(SCENE_STARTS + [17])
JOIN_GAP = 0.42   # natural paragraph pause inserted between scene takes (same as the in-take pauses)


def trim_silence(x, sr, thr_db=-45, keep=0.04):
    f = np.abs(x.astype(np.float32) / 32768); thr = 10 ** (thr_db / 20)
    idx = np.where(f > thr)[0]
    if not len(idx): return x
    k = int(keep * sr)
    return x[max(0, idx[0] - k): idx[-1] + k]


def toks(s):
    s = s.lower().replace("10,000", "ten thousand").replace("10000", "ten thousand")
    s = s.replace("pre-fill", "prefill").replace("pre fill", "prefill")
    return [norm(w) for w in re.split(r"[\s\-]+", s) if norm(w)]


def deviations(ref, heard):
    a, b = toks(ref), toks(heard)
    return [{"op": o, "script": " ".join(a[i1:i2]), "heard": " ".join(b[j1:j2])}
            for o, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if o != "equal" and "".join(a[i1:i2]) != "".join(b[j1:j2])]


def tts(a):
    """One request per scene (paragraph-boundary split), each verified by gpt-4o-transcribe and
    regenerated (up to --tries) if any word is dropped or changed; the best take per scene is kept."""
    from openai import OpenAI
    client = OpenAI()
    paras = read_script()
    bounds = SEGMENT_STARTS + [len(paras)]
    scenes = ["\n\n".join(paras[bounds[k]:bounds[k + 1]]) for k in range(len(SEGMENT_STARTS))]
    only = {int(v) for v in a.scenes.split(",")} if a.scenes else None
    cache = ROOT / "audio" / "scenes"; cache.mkdir(parents=True, exist_ok=True)
    pcm, sr, report = [], None, []
    for k, text in enumerate(scenes):
        f = cache / f"seg{k + 1}.wav"
        if f.exists() and only is not None and (k + 1) not in only:
            x, sr = read_wav(f); pcm.append(trim_silence(x, sr)); report.append({"segment": k + 1, "reused": True}); continue
        best = None
        for t in range(1, a.tries + 1):
            r = client.audio.speech.create(model=MODEL, voice=a.voice, input=text, instructions=INSTRUCTIONS,
                                           response_format="wav", speed=a.speed)
            buf = r.read()
            with wave.open(io.BytesIO(buf)) as w:
                sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            write_wav("/tmp/_scene.wav", x, sr)
            heard = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=open("/tmp/_scene.wav", "rb")).text
            dev = deviations(text, heard)
            print(f"  segment {k + 1} try {t}: {len(x) / sr:5.1f}s  deviations={dev or 'none'}")
            if best is None or len(dev) < len(best[1]):
                best = (x, dev); write_wav(f, x, sr)
            if not dev: break
        pcm.append(trim_silence(best[0], sr)); report.append({"segment": k + 1, "paragraphs": [bounds[k], bounds[k + 1] - 1], "deviations": best[1]})
    gap = np.zeros(int(JOIN_GAP * sr), np.int16)
    x = np.concatenate([q for seg in pcm for q in (seg, gap)][:-1])
    write_wav(ROOT / "audio" / "narration_raw.wav", x, sr)
    raw = len(x) / sr; removed = 0.0
    if a.max_pause > 0:
        x, removed = tighten_pauses(x, sr, a.max_pause)
    write_wav(ROOT / "audio" / "narration.wav", x, sr)
    meta = {"model": MODEL, "voice": a.voice, "speed": a.speed, "instructions": INSTRUCTIONS,
            "requests": f"{len(scenes)} (one per scene + 1 extra paragraph split)", "join_gap_s": JOIN_GAP, "scenes": report,
            "sample_rate": sr, "raw_duration_s": round(raw, 3), "max_pause_s": a.max_pause,
            "silence_removed_s": round(removed, 3), "duration_s": round(len(x) / sr, 3)}
    (ROOT / "audio" / "tts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"raw {raw:.2f}s, trimmed {removed:.2f}s of silence -> {len(x) / sr:.2f}s")


# ------------------------------------------------------------------ alignment
def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower().replace("’", "'"))


def transcribe(path):
    from openai import OpenAI
    with open(path, "rb") as f:
        r = OpenAI().audio.transcriptions.create(model="whisper-1", file=f, response_format="verbose_json",
                                                 timestamp_granularities=["word"], language="en")
    return {"text": r.text, "words": [{"w": w.word, "start": w.start, "end": w.end} for w in r.words]}


def verbatim_check(path):
    def toks(s):
        s = s.lower().replace("10,000", "ten thousand").replace("10000", "ten thousand")
        s = s.replace("pre-fill", "prefill").replace("pre fill", "prefill")
        return [norm(w) for w in re.split(r"[\s\-]+", s) if norm(w)]
    from openai import OpenAI
    with open(path, "rb") as f:
        txt = OpenAI().audio.transcriptions.create(model="gpt-4o-transcribe", file=f).text
    a, b = toks(" ".join(read_script())), toks(txt)
    diffs = [{"op": o, "script": " ".join(a[i1:i2]), "heard": " ".join(b[j1:j2])}
             for o, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o != "equal"]
    return txt, diffs


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


def pauses(path, thr_db=-38, min_len=0.16):
    x, sr = read_wav(path)
    f = x.astype(np.float32) / 32768; hop = int(sr * 0.01); n = len(f) // hop
    db = 20 * np.log10(np.sqrt((f[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)); quiet = db < thr_db
    out, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]: j += 1
            if (j - i) * 0.01 >= min_len: out.append([round(i * 0.01, 2), round(j * 0.01, 2)])
            i = j
        else: i += 1
    return out, len(f) / sr


def align():
    wav = ROOT / "audio" / "narration.wav"
    sents = read_script()
    tr = transcribe(wav)
    rec = []
    for w in tr["words"]:                          # spell recogniser digits the way the script does
        if norm(w["w"]) in ("10000", "10"):
            m = (w["start"] + w["end"]) / 2
            rec += [{"w": "ten", "start": w["start"], "end": m}, {"w": "thousand", "start": m, "end": w["end"]}]
        elif norm(w["w"]):
            rec.append(w)
    sw = [(si, w) for si, s in enumerate(sents) for w in s.split()]
    pairs = nw([norm(w) for _, w in sw], [norm(w["w"]) for w in rec])
    N = len(sw); st = np.full(N, np.nan); en = np.full(N, np.nan)
    for i, j in pairs.items(): st[i], en[i] = rec[j]["start"], rec[j]["end"]
    ok = ~np.isnan(st); i = 0
    while i < N:                                   # spread unmatched runs between matched neighbours
        if not ok[i]:
            j = i
            while j < N and not ok[j]: j += 1
            a = en[i - 1] if i > 0 else 0.0; b = st[j] if j < N else (en[i - 1] + 0.4 * (j - i))
            for k in range(i, j):
                st[k], en[k] = a + (b - a) * (k - i) / (j - i), a + (b - a) * (k - i + 1) / (j - i)
            i = j
        else: i += 1
    ps, dur = pauses(wav)
    out = []
    for si, s in enumerate(sents):
        ks = [k for k, (q, _) in enumerate(sw) if q == si]
        s0, s1 = float(st[ks[0]]), float(en[ks[-1]])
        for a, b in ps:
            if a <= s0 < b: s0 = b
            if a < s1 <= b: s1 = a
        c = [p[1] for p in ps if abs(p[1] - s0) < 0.6]
        if c: s0 = min(c, key=lambda v: abs(v - s0))
        c = [p[0] for p in ps if abs(p[0] - s1) < 0.6 and p[0] > s0]
        if c: s1 = min(c, key=lambda v: abs(v - s1))
        out.append({"i": si, "text": s, "start": round(s0, 3), "end": round(s1, 3),
                    "words": [{"w": sw[k][1], "start": round(float(st[k]), 3), "end": round(float(en[k]), 3),
                               "matched": bool(ok[k])} for k in ks]})
    for a, b in zip(out, out[1:]):
        if b["start"] < a["end"]: a["end"] = b["start"] = round((a["end"] + b["start"]) / 2, 3)
    heard, vdiff = verbatim_check(wav)
    res = {"source": "audio/narration.wav", "duration": round(dur, 3),
           "method": "whisper-1 word anchors + Needleman-Wunsch alignment to script.txt + RMS pause snapping",
           "qa": {"script_words": N, "matched_words": int(ok.sum()), "match_rate": round(float(ok.mean()), 4),
                  "unmatched_script_words": [sw[k][1] for k in range(N) if not ok[k]],
                  "verbatim_check": {"recogniser": "gpt-4o-transcribe", "differences": vdiff, "heard": heard}},
           "pauses": ps,
           "timeline": {"lead_in": LEAD_IN, "tail_hold": TAIL, "fps": 30, "total": round(LEAD_IN + dur + TAIL, 3)},
           "sentences": out}
    (ROOT / "timings.json").write_text(json.dumps(res, indent=1))
    print(f"narration {dur:.2f}s -> video {res['timeline']['total']:.2f}s; word match {ok.mean():.1%}")
    print("unmatched:", res["qa"]["unmatched_script_words"])
    print("verbatim differences:", vdiff or "none")
    for s in out: print(f"{s['i']:2d} {s['start']:7.2f} {s['end']:7.2f}  {s['text'][:70]}")
    return vdiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="onyx")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-pause", type=float, default=0.0, help="cap inter-sentence silences (s); 0 = off")
    ap.add_argument("--align-only", action="store_true")
    ap.add_argument("--tries", type=int, default=4, help="per-scene regenerations if the take deviates from the script")
    ap.add_argument("--scenes", default="", help="comma list: regenerate only these segments, reuse cached others")
    a = ap.parse_args()
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (environment or .env)")
    if not a.align_only:
        tts(a)
    align()


if __name__ == "__main__":
    main()
