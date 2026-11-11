#!/usr/bin/env python3
"""Generate the voiceover with OpenAI gpt-4o-mini-tts.

    python generate_audio.py                 # voice=cedar, speed=1.0  -> narration.wav
    python generate_audio.py --voice onyx    # alternative male voice
    python generate_audio.py --speed 1.05    # nudge pacing if the video would exceed 180 s

The narration is read verbatim from script.txt. One request is used when the script fits the
model's input limit (it does: ~2.5k characters); otherwise the script is split at sentence
boundaries, each chunk is synthesised with the same voice + instructions, and the PCM is
concatenated sample-exactly (no re-encoding seams).

Reads OPENAI_API_KEY from the environment or from a .env file in this folder or its parent.
"""
import argparse, io, json, os, re, sys, wave, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
MODEL = "gpt-4o-mini-tts"
MAX_CHARS = 3800   # conservative: model input limit is 2000 tokens

INSTRUCTIONS = (
    "Voice: a calm, confident, articulate male machine-learning educator. "
    "Tone: warm and technically knowledgeable, never salesy or over-excited. "
    "Pacing: natural educational narration at a measured pace, with short pauses between "
    "conceptual sections and a brief beat after rhetorical questions. "
    "Pronunciation: say 'DeepSeek' as one word; 'V3.2' as 'vee three point two'; 'ReLU' as 'ray-loo'; "
    "'top-k' as 'top kay'."
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
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="cedar", help="cedar (default) or onyx")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="narration.wav")
    a = ap.parse_args()
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (env or .env)")
    from openai import OpenAI
    client = OpenAI()

    paras = read_script()
    chunks = chunk(paras)
    print(f"{len(paras)} sentences/paragraphs, {sum(map(len, paras))} chars -> {len(chunks)} request(s)")
    frames, params = [], None
    for i, c in enumerate(chunks):
        resp = client.audio.speech.create(model=MODEL, voice=a.voice, input=c,
                                          instructions=INSTRUCTIONS, response_format="wav", speed=a.speed)
        data = resp.read()
        with wave.open(io.BytesIO(data)) as w:
            p = w.getparams()
            if params and (p.nchannels, p.sampwidth, p.framerate) != (params.nchannels, params.sampwidth, params.framerate):
                sys.exit("chunk audio formats differ")
            params = params or p
            frames.append(w.readframes(w.getnframes()))
        print(f"  chunk {i + 1}/{len(chunks)}: {len(c)} chars ok")
    out = ROOT / a.out
    with wave.open(str(out), "wb") as w:
        w.setnchannels(params.nchannels); w.setsampwidth(params.sampwidth); w.setframerate(params.framerate)
        for f in frames:
            w.writeframes(f)
    dur = sum(len(f) for f in frames) / (params.sampwidth * params.nchannels * params.framerate)
    meta = {"model": MODEL, "voice": a.voice, "speed": a.speed, "instructions": INSTRUCTIONS,
            "requests": len(chunks), "sample_rate": params.framerate, "duration_s": round(dur, 3)}
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "tts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out.name}: {dur:.2f}s @ {params.framerate} Hz")


if __name__ == "__main__":
    main()
