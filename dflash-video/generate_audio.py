#!/usr/bin/env python3
"""Narration for the DFlash speculative-decoding reel (OpenAI gpt-4o-mini-tts + FFmpeg).

    python generate_audio.py                  # TTS -> raw alignment -> assemble output/narration.wav
    python generate_audio.py --voice onyx     # alternative male voice
    python generate_audio.py --speed 1.06     # tighten delivery if the reel would exceed 180 s
    python generate_audio.py --skip-tts       # re-assemble from the existing raw take (no API call)

Pipeline
  1. TTS: narration.txt is read verbatim and sent in ONE request (about 3k characters, inside the
     model's input limit). If a future script is longer it is split at paragraph boundaries and
     the PCM chunks are concatenated sample-exactly.            -> output/narration_raw.wav
  2. Raw alignment (align_audio.align): measured sentence times inside the raw take, so we know
     where each logical section (scene) ends.                   -> output/raw_alignment.json
  2b. Pause tightening: silences longer than PAUSE_CAP are shortened (the middle of the silence is cut,
     speech is never touched or time-stretched).                -> output/narration_tight.wav
  3. Assembly with FFmpeg: 1.0 s lead-in, a short extra pause inserted inside the natural silence
     between sections (never inside speech), two-pass loudnorm to -16 LUFS / -1.5 dBTP, and a
     natural outro hold.                                         -> output/narration.wav
  The narration is never time-stretched or padded to a fixed length.

Afterwards run align_audio.py to measure timings.json on the FINAL assembled file.
Reads OPENAI_API_KEY from the environment or from a .env file here or in the parent folder.
"""
import argparse, io, json, os, re, subprocess, sys, wave, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
ASSETS = ROOT / "output"
OUT = ROOT / "output"
MODEL = "gpt-4o-mini-tts"
MAX_CHARS = 3800
LEAD_IN = 1.0          # s of silence before the first word
SECTION_PAUSE = 0.35   # s of extra silence added between scenes
OUTRO_HOLD = 3.2       # s after the last word (ending card)
MAX_TOTAL = 180.0
PAUSE_CAP = 0.5        # s: silences longer than this inside the raw take are shortened to this (middle removed)

INSTRUCTIONS = (
    "Speak like an engaging technical educator explaining an advanced machine learning concept. "
    "Use a natural conversational pace, clear pronunciation, and short pauses after important concepts. "
    "Avoid sounding robotic, theatrical, or overly excited. Emphasize technical terms naturally. "
    "Pronounce 'DFlash' as 'D Flash', 'GPU' as the letters G P U, and 'EAGLE-3' as 'Eagle three'."
)

# paragraph index (0-based, in narration.txt) at which each scene starts
SCENE_STARTS = {"s1_bottleneck": 0, "s2_specdec": 5, "s3_problem": 11, "s4_dflash": 15, "s5_conditioning": 20,
                "s6_verify": 26, "s7_speed": 31, "s8_takeaway": 36}


def scene_of(i):
    best = None
    for k, v in SCENE_STARTS.items():
        if i >= v:
            best = k
    return best


def load_env():
    for p in (ROOT / ".env", ROOT.parent / ".env"):  # never committed; see .gitignore
        if p.exists():
            for line in p.read_text().splitlines():
                m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$", line)
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def read_script():
    text = (ROOT / "narration.txt").read_text(encoding="utf-8").strip()
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


def tts(voice, speed):
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (env or .env)")
    from openai import OpenAI
    client = OpenAI()
    paras = read_script()
    groups = {}
    for i, p in enumerate(paras):
        groups.setdefault(scene_of(i), []).append(p)
    chunks = chunk([" ".join(g) for g in groups.values()])
    print(f"{len(paras)} paragraphs, {sum(map(len, paras))} chars -> {len(chunks)} request(s), voice={voice}, speed={speed}")
    frames, params = [], None
    for i, c in enumerate(chunks):
        resp = client.audio.speech.create(model=MODEL, voice=voice, input=c, instructions=INSTRUCTIONS,
                                          response_format="wav", speed=speed)
        data = resp.read()
        with wave.open(io.BytesIO(data)) as w:
            p = w.getparams()
            if params and (p.nchannels, p.sampwidth, p.framerate) != (params.nchannels, params.sampwidth, params.framerate):
                sys.exit("chunk audio formats differ")
            params = params or p
            frames.append(w.readframes(w.getnframes()))
    ASSETS.mkdir(exist_ok=True)
    raw = ASSETS / "narration_raw.wav"
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(params.nchannels); w.setsampwidth(params.sampwidth); w.setframerate(params.framerate)
        for f in frames:
            w.writeframes(f)
    dur = sum(len(f) for f in frames) / (params.sampwidth * params.nchannels * params.framerate)
    OUT.mkdir(exist_ok=True)
    (OUT / "tts_meta.json").write_text(json.dumps({"model": MODEL, "voice": voice, "speed": speed,
        "instructions": INSTRUCTIONS, "requests": len(chunks), "sample_rate": params.framerate,
        "raw_duration_s": round(dur, 3)}, indent=2))
    print(f"wrote output/narration_raw.wav: {dur:.2f}s")


def ffmpeg_stderr(args):
    return subprocess.run(["ffmpeg", "-hide_banner", *args], capture_output=True, text=True).stderr


def tighten():
    """Shorten over-long silences (never speech): each pause > PAUSE_CAP keeps PAUSE_CAP/2 on both sides."""
    import align_audio
    src = ASSETS / "narration_raw.wav"
    ps, dur = align_audio.pauses(src, min_len=PAUSE_CAP + 0.05)
    keep, prev = [], 0.0
    for a, b in ps:
        if a < 0.05 or b > dur - 0.05:
            continue
        mid_a, mid_b = a + PAUSE_CAP / 2, b - PAUSE_CAP / 2
        keep.append((prev, mid_a)); prev = mid_b
    keep.append((prev, dur))
    parts = [f"[0:a]atrim={x:.3f}:{y:.3f},asetpts=PTS-STARTPTS[k{i}]" for i, (x, y) in enumerate(keep)]
    graph = ";".join(parts) + ";" + "".join(f"[k{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=0:a=1[out]"
    out = ASSETS / "narration_tight.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-filter_complex", graph, "-map", "[out]",
                    "-c:a", "pcm_s16le", str(out)], check=True)
    new = sum(y - x for x, y in keep)
    print(f"tightened pauses > {PAUSE_CAP}s: {dur:.2f}s -> {new:.2f}s ({len(keep) - 1} pauses shortened)")
    return round(dur - new, 3)


def assemble():
    import align_audio
    removed = tighten()
    raw = ASSETS / "narration_tight.wav"
    al = align_audio.align(raw, cache_name="whisper_raw.json")
    (OUT / "raw_alignment.json").write_text(json.dumps(al, indent=1))
    sents = al["sentences"]
    # cut points: middle of the silence between the last sentence of a scene and the next scene
    cuts = []
    for i in range(1, len(sents)):
        if scene_of(i) != scene_of(i - 1):
            a, b = sents[i - 1]["end"], sents[i]["start"]
            cuts.append(round((a + b) / 2, 3))
    total_speech = al["duration"]
    total = LEAD_IN + total_speech + SECTION_PAUSE * len(cuts) + OUTRO_HOLD
    tail_hold = OUTRO_HOLD - max(0.0, total_speech - sents[-1]["end"])
    if total > MAX_TOTAL:
        sys.exit(f"assembled duration {total:.1f}s > {MAX_TOTAL}s: regenerate with a higher --speed")
    parts, segs, prev = [], [], 0.0
    for i, c in enumerate(cuts):
        parts.append(f"[0:a]atrim={prev}:{c},asetpts=PTS-STARTPTS[s{i}]")
        parts.append(f"aevalsrc=0:d={SECTION_PAUSE}:s=24000:c=mono[z{i}]")
        segs += [f"[s{i}]", f"[z{i}]"]
        prev = c
    parts.append(f"[0:a]atrim={prev},asetpts=PTS-STARTPTS[sl]")
    segs.append("[sl]")
    parts.append(f"{''.join(segs)}concat=n={len(segs)}:v=0:a=1[cat]")
    pre = f"[cat]aresample=24000,aformat=sample_fmts=fltp:channel_layouts=mono,adelay={int(LEAD_IN * 1000)}:all=1,apad,atrim=0:{total:.3f}"
    graph = lambda tail: ";".join(parts + [f"{pre},{tail}"])
    err = ffmpeg_stderr(["-i", str(raw), "-filter_complex", graph("loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json[out]"),
                         "-map", "[out]", "-f", "null", "-"])
    m = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    out = ASSETS / "narration.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-filter_complex",
                    graph(f"{ln},aresample=48000,apad,atrim=0:{total:.3f}[out]"), "-map", "[out]",
                    "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1", str(out)], check=True)
    meta = json.loads((OUT / "tts_meta.json").read_text()) if (OUT / "tts_meta.json").exists() else {}
    meta.update({"pause_cap_s": PAUSE_CAP, "silence_removed_s": removed, "lead_in_s": LEAD_IN, "section_pause_s": SECTION_PAUSE, "section_cuts_raw_s": cuts,
                 "outro_hold_s": OUTRO_HOLD, "measured_input_lufs": m["input_i"], "target_lufs": -16,
                 "final_duration_s": round(total, 3)})
    (OUT / "tts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote output/narration.wav: {total:.2f}s ({len(cuts)} section pauses, measured {m['input_i']} LUFS -> -16)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="cedar")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--skip-tts", action="store_true")
    a = ap.parse_args()
    if not a.skip_tts:
        tts(a.voice, a.speed)
    assemble()
    print("next: python align_audio.py   (measures timings.json on output/narration.wav)")


if __name__ == "__main__":
    main()
