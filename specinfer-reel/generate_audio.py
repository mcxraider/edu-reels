#!/usr/bin/env python3
"""Narration + measured sentence timings for the SpecInfer reel.

    python generate_audio.py                 # TTS -> assemble assets/voiceover.wav -> forced alignment -> timings.json
    python generate_audio.py --voice marin   # alternative voice (default: cedar)
    python generate_audio.py --speed 1.05    # tighten delivery if the reel would exceed 180 s
    python generate_audio.py --skip-tts      # re-assemble + re-align from the existing raw take (no TTS call)
    python generate_audio.py --align-only    # only re-measure timings.json from assets/voiceover.wav

Pipeline
  1. TTS (OpenAI Speech API, gpt-4o-mini-tts). narration.txt is read verbatim and sent in ONE request so the
     narration is a single continuous take.                                   -> assets/voiceover_raw.wav
  2. Raw alignment: measure sentence boundaries in the raw take so we know where each scene ends.
  3. FFmpeg assembly: 1.0 s lead-in, a short extra pause inserted in the natural silence between scenes
     (never inside speech), two-pass loudnorm to -16 LUFS / -1.5 dBTP (linear, so nothing clips), and an
     ending hold for the final title card.                                     -> assets/voiceover.wav
  4. Forced alignment on the FINAL file. The Speech API returns no timestamps, so timing is measured from the
     audio itself and never invented:
       a. recognition anchors: whisper-1 word timestamps on ~20 s windows cut inside measured pauses
          (only the timestamps are used - the recognised text is never shown);
       b. the exact script words are aligned to the recognised words (Needleman-Wunsch over normalised tokens);
          matched words take the measured time, unmatched runs are spread between their matched neighbours;
       c. sentence edges are snapped to RMS-measured pause edges;
       d. QA: the word match rate (script vs audio) verifies the narration is verbatim.   -> timings.json

Reads OPENAI_API_KEY from the environment or a .env file in this folder or its parent.
"""
import argparse, io, json, os, re, subprocess, sys, tempfile, wave, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
ASSETS, OUT = ROOT / "assets", ROOT / "output"
RAW, FINAL = ASSETS / "voiceover_raw.wav", ASSETS / "voiceover.wav"
MODEL = "gpt-4o-mini-tts"
LEAD_IN = 1.0          # s of silence before the first word
SECTION_PAUSE = 0.5    # s of extra silence inserted between scenes (inside existing silence)
END_HOLD = 5.0         # s after the last word: final title card hold
MAX_TOTAL = 180.0

INSTRUCTIONS = (
    "Voice: an experienced machine learning engineer explaining a challenging idea to fellow engineers. "
    "Tone: calm, engaging, technically confident and conversational - never theatrical or salesy. "
    "Pacing: about 145 to 150 words per minute, with brief natural pauses after key concepts and between paragraphs. "
    "Pronounce 'SpecInfer' as 'Spec-Infer' (spek IN-fur). Read the text exactly as written."
)

# sentence (paragraph) index in narration.txt at which each scene begins
SCENES = [("s1_bottleneck", 0), ("s2_speculative", 3), ("s3_tree", 6), ("s4_mask", 10),
          ("s5_verify", 16), ("s6_accept", 18), ("s7_speed", 20), ("s8_takeaway", 25)]

# animation events attached to each sentence (the cue words index.html keys its motion to)
EVENTS = {
    0: ["title card fades out", "large target model block appears", "prompt 'The cat' appears"],
    1: ["pass 1..3 light the target block; 'sat' 'on' 'the' appear one per pass on cue words 'one'/'another'",
        "timeline of sequential passes grows; 'forward pass' highlights pass counter"],
    2: ["weights stream from memory into the block each pass ('loading')", "'expensive' tags the per-pass cost"],
    3: ["screen splits into Traditional vs Speculative lanes", "draft tokens appear rapidly on 'several'",
        "target verifies them together on 'verify'"],
    4: ["'speculative decoding' label; draft tokens turn green after verification"],
    5: ["a linear draft chain morphs into a branching tree ('a step further')"],
    6: ["small draft model block appears beside the large target model ('draft model')"],
    7: ["draft model proposes several continuations; target model stays idle"],
    8: ["prompt 'The animal' at root; three branches grow on 'branching token tree'", "label: Token Tree"],
    9: ["each root-to-leaf path is highlighted in turn ('each path')"],
    10: ["tree slides to the top; question framing"],
    11: ["three paths flash together; warning: must not mix branch information"],
    12: ["label 'tree-based attention'"],
    13: ["nodes fly from the tree into one packed sequence ('pack')", "attention mask grid builds ('attention mask')"],
    14: ["query 'on' (sat branch) row: prompt, 'sat' and itself light up ('ancestors')"],
    15: ["'ran' and 'jumped' columns masked with crosses ('unrelated branches')", "full block-diagonal tree mask revealed"],
    16: ["packed sequence + mask enter the target model; all positions light together ('single forward pass')",
        "one next-token prediction emerges above every node"],
    17: ["greedy check: each drafted child is compared with its parent's target prediction ('agree')"],
    18: ["'sat' 'on' 'the' accepted one by one (green)", "'ran'/'jumped' branches rejected (red)",
         "'mat' supplied by the target model as the correction token", "output becomes 'The animal sat on the mat'"],
    19: ["loop arrow: draft -> tree -> verify repeats from the new prefix"],
    20: ["comparison title"],
    21: ["timeline: 4 sequential target passes vs draft + 1 tree-verification pass (illustrative)"],
    22: ["both outputs shown identical: same target-model output"],
    23: ["bigger tree lengthens verify bar; poor draft accepts fewer tokens (trade-off)"],
    24: ["latency comparison settles: benefit only when savings outweigh overhead"],
    25: ["Draft -> Tree -> Verify blocks"],
    26: ["sequential chain crossed out"],
    27: ["tagline: 'More verified tokens per expensive model pass.'"],
    28: ["final title card SpecInfer - held to the end"],
}


def scene_of(i):
    return [name for name, start in SCENES if i >= start][-1]


def load_env():
    for p in (ROOT / ".env", ROOT.parent / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$", line)
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (environment or .env)")


def read_script():
    text = (ROOT / "narration.txt").read_text(encoding="utf-8").strip()
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# ------------------------------------------------------------------ 1. TTS
def tts(voice, speed):
    load_env()
    from openai import OpenAI
    paras = read_script()
    text = "\n\n".join(paras)
    if len(text) > 4000:
        sys.exit("narration exceeds the 4096-character single-request limit")
    print(f"TTS: {len(paras)} sentences, {len(text.split())} words, voice={voice}, speed={speed}")
    resp = OpenAI().audio.speech.create(model=MODEL, voice=voice, input=text, instructions=INSTRUCTIONS,
                                        response_format="wav", speed=speed)
    data = resp.read()
    with wave.open(io.BytesIO(data)) as w:
        p, frames = w.getparams(), w.readframes(w.getnframes())
    ASSETS.mkdir(exist_ok=True)
    with wave.open(str(RAW), "wb") as w:
        w.setnchannels(p.nchannels); w.setsampwidth(p.sampwidth); w.setframerate(p.framerate); w.writeframes(frames)
    dur = len(frames) / (p.sampwidth * p.nchannels * p.framerate)
    OUT.mkdir(exist_ok=True)
    (OUT / "tts_meta.json").write_text(json.dumps({"model": MODEL, "voice": voice, "speed": speed,
        "instructions": INSTRUCTIONS, "requests": 1, "sample_rate": p.framerate, "raw_duration_s": round(dur, 3),
        "words": len(text.split()), "raw_wpm": round(len(text.split()) / dur * 60, 1)}, indent=2))
    print(f"wrote {RAW.relative_to(ROOT)}: {dur:.2f}s ({len(text.split()) / dur * 60:.0f} wpm)")


# ------------------------------------------------------------------ helpers: audio measurement
def read_mono(path):
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return (x.reshape(-1, ch).mean(1) if ch > 1 else x), sr


def pauses(path, thr_db=-40, min_len=0.16):
    x, sr = read_mono(path)
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


# ------------------------------------------------------------------ forced alignment
def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower().replace("’", "'"))


def transcribe_window(wav, t0, t1, tag):
    cache = OUT / f"whisper_{tag}_{t0:.2f}_{t1:.2f}.json"
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
                                                     prompt="SpecInfer, speculative decoding, draft model, token tree.")
    res = {"text": r.text, "words": [{"w": w.word, "start": w.start + t0, "end": w.end + t0} for w in r.words]}
    OUT.mkdir(exist_ok=True)
    cache.write_text(json.dumps(res, indent=1))
    return res


def transcribe(wav, tag, chunk_s=20.0):
    ps, dur = pauses(wav, min_len=0.45)   # cut only in clear sentence-level pauses
    cuts, last = [0.0], 0.0
    for a, b in ps:
        mid = (a + b) / 2
        if mid - last >= chunk_s and dur - mid > 5:
            cuts.append(round(mid, 3)); last = mid
    cuts.append(round(dur, 3))
    words, texts = [], []
    for t0, t1 in zip(cuts, cuts[1:]):
        r = transcribe_window(wav, t0, t1, tag)
        words += r["words"]; texts.append(r["text"].strip())
    return {"text": " ".join(texts), "words": words}


def split_rec(rec):
    """normalise recogniser output: 'Spec' 'Infer' -> one token; hyphenated words -> parts."""
    out = []
    for w in rec:
        for part in [p for p in re.split(r"[-‐–—]", w["w"]) if norm(p)]:
            out.append({"w": part, "start": w["start"], "end": w["end"]})
    merged, i = [], 0
    while i < len(out):
        if i + 1 < len(out) and norm(out[i]["w"]) + norm(out[i + 1]["w"]) == "specinfer":
            merged.append({"w": "SpecInfer", "start": out[i]["start"], "end": out[i + 1]["end"]}); i += 2
        else:
            merged.append(out[i]); i += 1
    return merged


def nw(a, b, gap=-1, mis=-1, match=2):
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1)); P = np.zeros((n + 1, m + 1), dtype=np.int8)
    S[:, 0] = np.arange(n + 1) * gap; S[0, :] = np.arange(m + 1) * gap
    P[1:, 0] = 1; P[0, 1:] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + (match if ai == b[j - 1] else mis)
            u = S[i - 1, j] + gap; l = S[i, j - 1] + gap
            if d >= u and d >= l: S[i, j] = d; P[i, j] = 0
            elif u >= l: S[i, j] = u; P[i, j] = 1
            else: S[i, j] = l; P[i, j] = 2
    i, j, pairs = n, m, {}
    while i > 0 or j > 0:
        k = P[i, j]
        if i > 0 and j > 0 and k == 0:
            if a[i - 1] == b[j - 1]: pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and (k == 1 or j == 0): i -= 1
        else: j -= 1
    return pairs


def match_words(sw, rec):
    sub, parent = [], []
    for k, (_, w) in enumerate(sw):
        for part in [p for p in re.split(r"[-‐–—]", w) if norm(p)]:
            sub.append(norm(part)); parent.append(k)
    pairs = nw(sub, [norm(w["w"]) for w in rec])
    N = len(sw)
    st = np.full(N, np.nan); en = np.full(N, np.nan)
    need = np.bincount(parent, minlength=N); got = np.zeros(N, int)
    for i, j in pairs.items():
        k = parent[i]; got[k] += 1
        st[k] = rec[j]["start"] if np.isnan(st[k]) else min(st[k], rec[j]["start"])
        en[k] = rec[j]["end"] if np.isnan(en[k]) else max(en[k], rec[j]["end"])
    partial = got < need
    st[partial] = np.nan; en[partial] = np.nan
    return st, en


def align(wav, tag):
    paras = read_script()
    tr = transcribe(wav, tag)
    rec = split_rec([w for w in tr["words"] if norm(w["w"])])
    sw = [(si, w) for si, s in enumerate(paras) for w in s.split()]
    N = len(sw)
    st, en = match_words(sw, rec)
    # gap repair: a run of >= 2 unmatched script words is re-transcribed as an isolated window
    ok, repaired, i = ~np.isnan(st), [], 0
    while i < N:
        if not ok[i]:
            j = i
            while j < N and not ok[j]: j += 1
            if j - i >= 2:
                t0 = max(0.0, (st[i - 1] if i > 0 else 0.0) - 0.4)
                t1 = (en[j] if j < N else rec[-1]["end"] + 1.0) + 0.4
                win = split_rec([w for w in transcribe_window(wav, t0, t1, tag + "_gap")["words"] if norm(w["w"])])
                cand = sorted([w for w in rec if not (t0 < w["start"] < t1)] + win, key=lambda w: w["start"])
                if (~np.isnan(match_words(sw, cand)[0])).sum() > (~np.isnan(match_words(sw, rec)[0])).sum():
                    rec = cand; repaired.append([round(t0, 2), round(t1, 2)])
            i = j
        else: i += 1
    if repaired:
        st, en = match_words(sw, rec)
    ok, i = ~np.isnan(st), 0
    while i < N:   # spread unmatched runs between matched neighbours
        if not ok[i]:
            j = i
            while j < N and not ok[j]: j += 1
            a = en[i - 1] if i > 0 else 0.0
            b = st[j] if j < N else en[i - 1] + 0.4 * (j - i)
            for k in range(i, j):
                st[k] = a + (b - a) * (k - i) / (j - i); en[k] = a + (b - a) * (k - i + 1) / (j - i)
            i = j
        else: i += 1
    ps, dur = pauses(wav)
    segs = []
    for si, s in enumerate(paras):
        ks = [k for k, (q, _) in enumerate(sw) if q == si]
        s0, s1 = float(st[ks[0]]), float(en[ks[-1]])
        for a, b in ps:                    # never start/end inside a measured silence
            if a <= s0 < b: s0 = b
            if a < s1 <= b: s1 = a
        c = [p[1] for p in ps if abs(p[1] - s0) < 0.6 and p[1] < s1]
        if c: s0 = min(c, key=lambda v: abs(v - s0))
        c = [p[0] for p in ps if abs(p[0] - s1) < 0.6 and p[0] > s0]
        if c: s1 = min(c, key=lambda v: abs(v - s1))
        words = [{"w": sw[k][1], "start": round(float(st[k]), 3), "end": round(float(en[k]), 3), "matched": bool(ok[k])}
                 for k in ks]
        words[0]["start"] = max(words[0]["start"], round(s0, 3))
        segs.append({"id": f"seg{si:02d}", "index": si, "scene": scene_of(si), "text": s,
                     "start": round(s0, 3), "end": round(s1, 3), "events": EVENTS.get(si, []), "words": words})
    for a, b in zip(segs, segs[1:]):
        if b["start"] < a["end"]: a["end"] = b["start"] = round((a["end"] + b["start"]) / 2, 3)
    return {"duration": round(dur, 3), "pauses": ps, "segments": segs,
            "qa": {"script_words": N, "matched_words": int(ok.sum()), "match_rate": round(float(ok.mean()), 4),
                   "unmatched_script_words": [sw[k][1] for k in range(N) if not ok[k]],
                   "repaired_windows": repaired, "recognised_text": tr["text"]}}


# ------------------------------------------------------------------ 3. assembly
def ffmpeg_stderr(args):
    return subprocess.run(["ffmpeg", "-hide_banner", *args], capture_output=True, text=True).stderr


def raw_complete(al):
    """every sentence must be audibly present in the take (>= 60 % of its words recognised)."""
    bad = [s["index"] for s in al["segments"] if np.mean([w["matched"] for w in s["words"]]) < 0.6]
    return bad


def assemble(al=None):
    al = al or align(RAW, "raw")
    (OUT / "raw_alignment.json").write_text(json.dumps(al, indent=1))
    segs, (_, raw_dur) = al["segments"], pauses(RAW)
    cuts = [round((segs[i - 1]["end"] + segs[i]["start"]) / 2, 3)
            for i in range(1, len(segs)) if segs[i]["scene"] != segs[i - 1]["scene"]]
    last_word = segs[-1]["end"]
    speech = min(raw_dur, last_word + 0.25)          # trim TTS tail silence; END_HOLD is added explicitly
    total = LEAD_IN + speech + SECTION_PAUSE * len(cuts) + END_HOLD
    if total > MAX_TOTAL:
        sys.exit(f"assembled duration {total:.1f}s > {MAX_TOTAL}s: regenerate with a higher --speed")
    with wave.open(str(RAW)) as w: sr = w.getframerate()
    parts, labels, prev = [], [], 0.0
    for i, c in enumerate(cuts):
        parts.append(f"[0:a]atrim={prev}:{c},asetpts=PTS-STARTPTS[s{i}]")
        parts.append(f"aevalsrc=0:d={SECTION_PAUSE}:s={sr}:c=mono[z{i}]")
        labels += [f"[s{i}]", f"[z{i}]"]; prev = c
    parts.append(f"[0:a]atrim={prev}:{speech:.3f},asetpts=PTS-STARTPTS[sl]"); labels.append("[sl]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[cat]")
    pre = (f"[cat]aformat=sample_fmts=fltp:channel_layouts=mono,afade=t=out:st={speech + SECTION_PAUSE * len(cuts) - 0.2:.3f}:d=0.2,"
           f"adelay={int(LEAD_IN * 1000)}:all=1,apad,atrim=0:{total:.3f}")
    graph = lambda tail: ";".join(parts + [f"{pre},{tail}"])
    err = ffmpeg_stderr(["-i", str(RAW), "-filter_complex", graph("loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json[out]"),
                         "-map", "[out]", "-f", "null", "-"])
    m = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(RAW), "-filter_complex",
                    graph(f"{ln},aresample=48000,alimiter=limit=0.84:level=false,apad,atrim=0:{total:.3f}[out]"),
                    "-map", "[out]", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1", str(FINAL)], check=True)
    x, _ = read_mono(FINAL)
    peak_db = 20 * np.log10(np.abs(x).max() + 1e-9)
    meta = json.loads((OUT / "tts_meta.json").read_text()) if (OUT / "tts_meta.json").exists() else {}
    meta.update({"lead_in_s": LEAD_IN, "section_pause_s": SECTION_PAUSE, "section_cuts_raw_s": cuts,
                 "end_hold_s": END_HOLD, "measured_input_lufs": m["input_i"], "target_lufs": -16,
                 "peak_dbfs": round(float(peak_db), 2), "final_duration_s": round(total, 3)})
    (OUT / "tts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {FINAL.relative_to(ROOT)}: {total:.2f}s, {len(cuts)} scene pauses, "
          f"{m['input_i']} LUFS -> -16, peak {peak_db:.2f} dBFS")


# ------------------------------------------------------------------ 4. timings.json
def write_timings(check=True):
    al = align(FINAL, "final")
    segs = al["segments"]
    scenes = []
    for s in segs:
        if not scenes or scenes[-1]["id"] != s["scene"]:
            scenes.append({"id": s["scene"], "first_segment": s["id"], "speech_start": s["start"]})
        scenes[-1]["speech_end"] = s["end"]
    for i, sc in enumerate(scenes):   # boundaries: middle of the silence between scenes
        sc["start"] = 0.0 if i == 0 else round((scenes[i - 1]["speech_end"] + sc["speech_start"]) / 2, 3)
    for i, sc in enumerate(scenes):
        sc["end"] = al["duration"] if i == len(scenes) - 1 else scenes[i + 1]["start"]
    meta = json.loads((OUT / "tts_meta.json").read_text()) if (OUT / "tts_meta.json").exists() else {}
    speech = segs[-1]["end"] - segs[0]["start"]
    res = {"source": "assets/voiceover.wav", "duration": al["duration"], "fps": 30,
           "end_hold_s": round(al["duration"] - segs[-1]["end"], 3),
           "method": "whisper-1 word anchors + Needleman-Wunsch alignment to narration.txt + RMS pause snapping",
           "tts": meta, "qa": {**al["qa"], "speech_wpm": round(al["qa"]["script_words"] / speech * 60, 1)},
           "scenes": scenes, "pauses": al["pauses"], "segments": segs}
    (ROOT / "timings.json").write_text(json.dumps(res, indent=1))
    q = res["qa"]
    print(f"timings.json: {al['duration']:.2f}s, {len(segs)} sentences, match {q['match_rate']:.1%}, "
          f"{q['speech_wpm']} wpm; unmatched: {q['unmatched_script_words']}")
    for sc in scenes: print(f"  {sc['id']:16s} {sc['start']:7.2f} -> {sc['end']:7.2f}")
    if check and q["match_rate"] < 0.95:
        sys.exit("word match rate < 95 %: narration may not be verbatim - regenerate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="cedar", choices=["cedar", "marin"])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--align-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if not a.align_only:
        al = None
        for attempt in range(1 if a.skip_tts else 4):
            if not a.skip_tts:
                tts(a.voice, a.speed)
            al = align(RAW, "raw")
            bad = raw_complete(al)
            if not bad:
                break
            print(f"take {attempt + 1} is missing/garbled sentence(s) {bad}; regenerating")
            for f in OUT.glob("whisper_raw*.json"): f.unlink()
        else:
            sys.exit("TTS kept dropping sentences")
        assemble(al)
    write_timings()


if __name__ == "__main__":
    main()
