#!/usr/bin/env python3
"""Generate the CNN max-pooling narration with ElevenLabs (voice: Adam), single take.

Outputs
  narration.mp3   - raw Adam voiceover (one continuous take)
  alignment.json  - raw character-level alignment returned by ElevenLabs
  timings.json    - words, sentences, blocks, named cues and the time-warp
                    (narration time -> video time) used by index.html / render.mjs

Usage
  export ELEVENLABS_API_KEY=sk_...
  python3 generate_audio.py            # call the API (one request)
  python3 generate_audio.py --retime   # rebuild timings.json from alignment.json, no API call

Only the Python standard library is required.
"""
import base64
import json
import math
import os
import re
import subprocess
import sys
import urllib.request

VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # ElevenLabs premade voice "Adam"
MODEL_ID = "eleven_multilingual_v2"
SPEED = 1.1   # ElevenLabs speed (0.7-1.2); 1.0 runs ~81 s, too long for 90 s with think time
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Time-warp policy (narration time -> video time) ----------------------
LEAD_IN = 1.0        # silence before narration
TARGET_TOTAL = 80.0  # target length (flexible)
MAX_TOTAL = 90.0     # hard cap
MIN_THINK = 9.0      # thinking period after narration (spec: 8-10 s)
# Extra silence inserted at the midpoint of the natural pause after the
# sentence that starts with the given prefix (gives animations room to land).
PAUSES_AFTER = [
    ("Your input image", 0.5),   # feature map finishes appearing
    ("You then apply", 1.3),     # window sweeps all four regions
    ("The feature map", 0.7),    # 4x4 -> 2x2 settles
    ("Interestingly", 0.8),      # shift demo lands
    ("Why does", 0.3),
    ("D. ", 0.3),
]

CUES = {
    "cnn": "convolutional neural network to recognize",
    "digits": "handwritten digits",
    "input": "Your input image",
    "conv_layer": "a convolutional layer",
    "feature_map": "producing a feature map",
    "edges": "edges and curves",
    "pool": "two-by-two max pooling",
    "stride": "stride of two",
    "half": "half its original height and width",
    "strongest": "strongest activations are preserved",
    "shift": "a feature shifts slightly",
    "same_region": "within the same pooling region",
    "unchanged": "remain unchanged",
    "question": "Why does max pooling behave",
    "purpose": "what is its main purpose",
    "opt_A": "A. Max pooling",
    "opt_B": "B. Max pooling",
    "opt_C": "C. Max pooling",
    "opt_D": "D. Max pooling",
    "final_q": "Which is it",
    "verify": "simple feature map",
}


def load_script():
    with open(os.path.join(HERE, "script.txt"), encoding="utf-8") as f:
        return f.read().strip()


def tts_with_timestamps(text, key):
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps"
           "?output_format=mp3_44100_128")
    body = json.dumps({
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.8, "style": 0.0,
                           "use_speaker_boost": True, "speed": SPEED},
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "xi-api-key": key, "Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def audio_duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "default=nw=1:nk=1", path])
    return float(out)


def span_times(al, start, end):
    """Start/end seconds of text[start:end] from character alignment, trimming whitespace."""
    chars = al["characters"]
    s, e = start, end - 1
    while s < e and chars[s].isspace():
        s += 1
    while e > s and chars[e].isspace():
        e -= 1
    return al["character_start_times_seconds"][s], al["character_end_times_seconds"][e]


SCENE_OF = [  # sentence prefix -> scene
    ("You are training", "s1_input"), ("Your input image", "s1_input"),
    ("You then apply", "s2_pool"), ("The feature map", "s2_pool"),
    ("Interestingly", "s3_shift"), ("Why does", "s4_question"),
    ("A. ", "s4_question"), ("B. ", "s4_question"), ("C. ", "s4_question"),
    ("D. ", "s4_question"), ("Which is it", "s5_think"),
]


def build_timing(text, al, duration):
    assert "".join(al["characters"]) == text, "alignment characters do not match the script"
    # sentences ("A." option letters are not sentence ends)
    sentences, st = [], 0
    for m in re.finditer(r"[.?!](?=\s|$)", text):
        en = m.end()
        if re.fullmatch(r"\s*[A-D]\.", text[st:en]):
            continue
        seg = text[st:en]
        a = st + (len(seg) - len(seg.lstrip()))
        s, e = span_times(al, a, en)
        sentences.append({"index": len(sentences), "text": text[a:en], "char_start": a,
                          "char_end": en, "start": round(s, 3), "end": round(e, 3)})
        st = en
    for x in sentences:
        x["scene"] = next(sc for p, sc in SCENE_OF if x["text"].startswith(p))
    # paragraphs / option blocks
    blocks, pos = [], 0
    for para in text.split("\n\n"):
        st = text.index(para, pos)
        en = st + len(para)
        pos = en
        s, e = span_times(al, st, en)
        b = {"text": para, "char_start": st, "char_end": en, "start": round(s, 3), "end": round(e, 3)}
        m = re.match(r"([A-D])\. ", para)
        if m:
            b["option"] = m.group(1)
        blocks.append(b)
    # words (straight from character alignment, not estimated)
    words = []
    for m in re.finditer(r"\S+", text):
        s, e = span_times(al, m.start(), m.end())
        words.append({"w": m.group(0), "char_start": m.start(), "start": round(s, 3), "end": round(e, 3)})

    def find(phrase):
        i = text.index(phrase)
        s, e = span_times(al, i, i + len(phrase))
        return {"phrase": phrase, "start": round(s, 3), "end": round(e, 3)}

    cues = {k: find(p) for k, p in CUES.items()}

    inserts = []
    for prefix, dur in PAUSES_AFTER:
        i = next(k for k, x in enumerate(sentences) if x["text"].startswith(prefix))
        a, b = sentences[i]["end"], sentences[i + 1]["start"]
        assert b - a > 0.12, f"no natural pause after {prefix!r}"
        inserts.append({"at": round((a + b) / 2, 3), "dur": dur, "after": prefix})

    def warp(n):
        return LEAD_IN + n + sum(x["dur"] for x in inserts if x["at"] <= n)

    for x in sentences:  # audio vs visual timing per sentence
        x["video_start"] = round(warp(x["start"]), 3)
        x["video_end"] = round(warp(x["end"]), 3)
    narr_end_v = LEAD_IN + duration + sum(x["dur"] for x in inserts)
    total = max(TARGET_TOTAL, math.ceil(narr_end_v + MIN_THINK))
    assert total <= MAX_TOTAL, f"video would be {total}s (> {MAX_TOTAL})"
    warp_info = {"lead_in": LEAD_IN, "inserts": inserts, "narration_end_video": round(narr_end_v, 3),
                 "total": total, "frames": int(round(total * 30)),
                 "think_seconds": round(total - narr_end_v, 3)}
    return {"audio_duration": round(duration, 3), "warp": warp_info, "voice": "Adam",
            "voice_id": VOICE_ID, "model_id": MODEL_ID, "speed": SPEED, "script": text, "blocks": blocks,
            "sentences": sentences, "cues": cues, "words": words}


def write_timing(timing):
    with open(os.path.join(HERE, "timings.json"), "w") as f:
        json.dump(timing, f, indent=1)
    print("warp", json.dumps(timing["warp"]))
    for s in timing["sentences"]:
        print(f"  audio {s['start']:6.2f}-{s['end']:6.2f}  video {s['video_start']:6.2f}-{s['video_end']:6.2f}"
              f"  [{s['scene']}] {s['text'][:50]}")


def retime():
    text = load_script()
    with open(os.path.join(HERE, "alignment.json")) as f:
        al = json.load(f)["alignment"]
    write_timing(build_timing(text, al, audio_duration(os.path.join(HERE, "narration.mp3"))))


def main():
    if "--retime" in sys.argv:
        return retime()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("Set ELEVENLABS_API_KEY")
    text = load_script()
    res = tts_with_timestamps(text, key)
    mp3 = os.path.join(HERE, "narration.mp3")
    with open(mp3, "wb") as f:
        f.write(base64.b64decode(res["audio_base64"]))
    with open(os.path.join(HERE, "alignment.json"), "w") as f:
        json.dump({"alignment": res.get("alignment"),
                   "normalized_alignment": res.get("normalized_alignment")}, f)
    al = res.get("alignment") or res.get("normalized_alignment")
    dur = audio_duration(mp3)
    print(f"audio {dur:.2f}s")
    write_timing(build_timing(text, al, dur))


if __name__ == "__main__":
    main()
