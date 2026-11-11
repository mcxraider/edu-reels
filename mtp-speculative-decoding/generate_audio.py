#!/usr/bin/env python3
"""Narration for the MTP / speculative-decoding reel (OpenAI gpt-4o-mini-tts + FFmpeg).

    python generate_audio.py                  # TTS -> raw alignment -> assemble assets/narration.wav
    python generate_audio.py --voice onyx     # alternative male voice
    python generate_audio.py --speed 1.06     # tighten delivery if the reel would exceed 180 s
    python generate_audio.py --skip-tts       # re-assemble from the existing raw take (no API call)

Pipeline
  1. TTS: script.txt is read verbatim and sent in ONE request (about 3k characters, inside the
     model's input limit). If a future script is longer it is split at paragraph boundaries and
     the PCM chunks are concatenated sample-exactly.            -> assets/narration_raw.wav
  2. Raw alignment (align_audio.align): measured sentence times inside the raw take, so we know
     where each logical section (scene) ends.                   -> output/raw_alignment.json
  3. Assembly with FFmpeg: 1.0 s lead-in, a short extra pause inserted inside the natural silence
     between sections (never inside speech), two-pass loudnorm to -16 LUFS / -1.5 dBTP, and a
     natural outro hold.                                         -> assets/narration.wav
  The narration is never time-stretched or padded to a fixed length.

Afterwards run align_audio.py to measure timings.json on the FINAL assembled file.
Reads OPENAI_API_KEY from the environment or from a .env file here or in the parent folder.
"""
import argparse, io, json, os, re, subprocess, sys, wave, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUT = ROOT / "output"
MODEL = "gpt-4o-mini-tts"
MAX_CHARS = 3800
LEAD_IN = 1.0          # s of silence before the first word
SECTION_PAUSE = 0.45   # s of extra silence added between scenes
OUTRO_HOLD = 4.0       # s after the last word (final diagram hold)
MAX_TOTAL = 180.0

INSTRUCTIONS = (
    "Speak like an engaging technical educator explaining an advanced machine learning concept. "
    "Use a natural conversational pace, clear pronunciation, and short pauses after important concepts. "
    "Avoid sounding robotic, theatrical, or overly excited. Emphasize technical terms naturally. "
    "Pronounce 'MTP' as the letters M T P, and 'GPUs' as G P U's."
)

# paragraph index (0-based, in script.txt) at which each scene starts
SCENE_STARTS = {"s1_autoregressive": 0, "s2_mtp": 5, "s3_draft": 8, "s4_verify": 13, "s5_mask": 16,
                "s6_accept": 19, "s7_speedup": 23, "s8_failure": 29, "s9_takeaway": 31}


def scene_of(i):
    best = None
    for k, v in SCENE_STARTS.items():
        if i >= v:
            best = k
    return best


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


def tts(voice, speed):
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (env or .env)")
    from openai import OpenAI
    client = OpenAI()
    paras = read_script()
    chunks = chunk(paras)
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
    print(f"wrote assets/narration_raw.wav: {dur:.2f}s")


def ffmpeg_stderr(args):
    return subprocess.run(["ffmpeg", "-hide_banner", *args], capture_output=True, text=True).stderr


def assemble():
    import align_audio
    raw = ASSETS / "narration_raw.wav"
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
    meta.update({"lead_in_s": LEAD_IN, "section_pause_s": SECTION_PAUSE, "section_cuts_raw_s": cuts,
                 "outro_hold_s": OUTRO_HOLD, "measured_input_lufs": m["input_i"], "target_lufs": -16,
                 "final_duration_s": round(total, 3)})
    (OUT / "tts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote assets/narration.wav: {total:.2f}s ({len(cuts)} section pauses, measured {m['input_i']} LUFS -> -16)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="cedar")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--skip-tts", action="store_true")
    a = ap.parse_args()
    if not a.skip_tts:
        tts(a.voice, a.speed)
    assemble()
    print("next: python align_audio.py   (measures timings.json on assets/narration.wav)")


if __name__ == "__main__":
    main()
