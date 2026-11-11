#!/usr/bin/env python3
"""Generate the EAGLE-3 narration with OpenAI gpt-4o-mini-tts  ->  audio/narration.wav

    python generate_audio.py                  # voice=onyx, speed=1.0
    python generate_audio.py --voice cedar    # alternative male voice
    python generate_audio.py --speed 1.06     # moderate pacing adjustment if the reel would exceed 180 s

The narration is read VERBATIM from narration.txt (one sentence/paragraph per block).
The whole script (~3.2k characters, well under the model's 2000-token input limit) is
synthesised in ONE request, so there are no splice seams. If the script ever grows past
MAX_CHARS it is split only at paragraph (sentence) boundaries and the PCM is concatenated
sample-exactly with the same voice and instructions.

Reads OPENAI_API_KEY from the environment or from .env in this folder or its parent.
"""
import argparse, io, json, os, re, sys, wave, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
MODEL = "gpt-4o-mini-tts"
MAX_CHARS = 3800

INSTRUCTIONS = (
    "Speak like an excellent computer science educator explaining a fascinating engineering idea. "
    "Use a natural, confident, conversational tone. Maintain clear pronunciation of technical terms, "
    "including EAGLE-3, Transformer, speculative decoding, and hidden states. Introduce brief natural "
    "pauses around complex ideas. Avoid sounding robotic, overly dramatic, or like an advertisement. "
    "Pronounce 'EAGLE-3' as 'Eagle three', 'LLM' as 'L L M', and 'GPUs' as 'G P Us'."
)


def load_env():
    for p in (ROOT / ".env", ROOT.parent / ".env"):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="onyx", help="onyx (default) or cedar")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="audio/narration.wav")
    a = ap.parse_args()
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (env or .env)")
    from openai import OpenAI
    client = OpenAI()

    paras = read_script()
    chunks = chunk(paras)
    print(f"{len(paras)} sentences, {sum(map(len, paras))} chars -> {len(chunks)} request(s)")
    frames, params = [], None
    for i, c in enumerate(chunks):
        resp = client.audio.speech.create(model=MODEL, voice=a.voice, input=c, instructions=INSTRUCTIONS,
                                          response_format="wav", speed=a.speed)
        with wave.open(io.BytesIO(resp.read())) as w:
            p = w.getparams()
            if params and (p.nchannels, p.sampwidth, p.framerate) != (params.nchannels, params.sampwidth, params.framerate):
                sys.exit("chunk audio formats differ")
            params = params or p
            frames.append(w.readframes(w.getnframes()))
        print(f"  request {i + 1}/{len(chunks)}: {len(c)} chars ok")
    out = ROOT / a.out
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(params.nchannels); w.setsampwidth(params.sampwidth); w.setframerate(params.framerate)
        for f in frames:
            w.writeframes(f)
    dur = sum(len(f) for f in frames) / (params.sampwidth * params.nchannels * params.framerate)
    meta = {"model": MODEL, "voice": a.voice, "speed": a.speed, "instructions": INSTRUCTIONS,
            "requests": len(chunks), "sample_rate": params.framerate, "duration_s": round(dur, 3)}
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "tts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {a.out}: {dur:.2f}s @ {params.framerate} Hz")


if __name__ == "__main__":
    main()
