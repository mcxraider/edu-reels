#!/usr/bin/env python3
"""Generate the narration with ElevenLabs (Adam) in one continuous take.

Outputs:
  narration.mp3   - the narration audio
  alignment.json  - raw character-level alignment returned by ElevenLabs
  timing.json     - sentence-level / option-level / word-level timestamps

Usage:  ELEVENLABS_API_KEY=sk_... python3 generate_audio.py
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
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Time-warp policy (narration time -> video time) ----------------------
LEAD_IN = 1.0          # silence before the narration starts
TARGET_TOTAL = 80.0    # target video length
MAX_TOTAL = 90.0       # hard cap
MIN_THINK = 8.5        # minimum thinking time after the narration
# Extra silence inserted *inside existing pauses* after the paragraph whose
# text starts with the given prefix (gives the animation room to breathe).
PAUSES_AFTER = [
    ("You set K", 1.4),       # before "The first run"  (run 1 animation)
    ("The second run", 1.4),  # before "Both runs"       (run 2 settles)
    ("Both runs", 1.0),       # before the question
    ("Why does", 0.6),        # before option A
    ("A. ", 0.5), ("B. ", 0.5), ("C. ", 0.5),
    ("D. ", 0.6),             # before "Which is it"
]


def load_script():
    with open(os.path.join(HERE, "script.txt"), encoding="utf-8") as f:
        return f.read().strip()


def tts_with_timestamps(text, key):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps?output_format=mp3_44100_128"
    body = json.dumps({
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.8, "style": 0.0,
                           "use_speaker_boost": True, "speed": 1.0},
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
    """Start/end seconds of the characters text[start:end], skipping whitespace."""
    chars = al["characters"]
    s, e = start, end - 1
    while s < e and chars[s].isspace():
        s += 1
    while e > s and chars[e].isspace():
        e -= 1
    return al["character_start_times_seconds"][s], al["character_end_times_seconds"][e]


def build_timing(text, al, duration):
    assert "".join(al["characters"]) == text, "alignment characters do not match script"
    # Sentence split: every sentence-ending punctuation followed by whitespace/end.
    sentences, st = [], 0
    for m in re.finditer(r"[.?!](?=\s|$)", text):
        en = m.end()
        if re.fullmatch(r"\s*[A-D]\.", text[st:en]):  # option letter "A." is not a sentence
            continue
        seg = text[st:en]
        a = st + (len(seg) - len(seg.lstrip()))
        s, e = span_times(al, a, en)
        sentences.append({"index": len(sentences), "text": text[a:en], "char_start": a,
                          "char_end": en, "start": round(s, 3), "end": round(e, 3)})
        st = en
    # Paragraph-level blocks (options are their own paragraphs).
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
    # Words
    words = []
    for m in re.finditer(r"\S+", text):
        s, e = span_times(al, m.start(), m.end())
        words.append({"w": m.group(0), "char_start": m.start(), "start": round(s, 3), "end": round(e, 3)})

    def find(phrase):
        i = text.index(phrase)
        s, e = span_times(al, i, i + len(phrase))
        return {"phrase": phrase, "start": round(s, 3), "end": round(e, 3)}

    cues = {k: find(p) for k, p in {
        "three_groups": "three clearly visible groups",
        "k_three": "K equal to three",
        "random_seeds": "different random seeds",
        "first_run": "The first run",
        "second_run": "The second run",
        "splits": "splits one group into two clusters",
        "merges": "merges the other two groups together",
        "converge": "Both runs converge",
        "no_changes": "neither produces any further changes",
        "question": "Why does K-Means",
        "opt_A": "A. K-Means",
        "opt_B": "B. K-Means",
        "opt_C": "C. K-Means",
        "opt_D": "D. K-Means",
        "final_q": "Which is it",
        "inertia": "inertia scores",
    }.items()}
    # Time warp: cut each chosen pause at its midpoint and insert extra silence there.
    inserts = []
    for prefix, dur in PAUSES_AFTER:
        i = next(k for k, x in enumerate(sentences) if x["text"].startswith(prefix))
        a, b = sentences[i]["end"], sentences[i + 1]["start"]
        assert b - a > 0.15, f"no pause after {prefix!r}"
        inserts.append({"at": round((a + b) / 2, 3), "dur": dur, "after": prefix})
    narr_end_v = LEAD_IN + duration + sum(x["dur"] for x in inserts)
    total = max(TARGET_TOTAL, math.ceil(narr_end_v + MIN_THINK))
    assert total <= MAX_TOTAL, f"video would be {total}s"
    warp = {"lead_in": LEAD_IN, "inserts": inserts, "narration_end_video": round(narr_end_v, 3),
            "total": total, "think_seconds": round(total - narr_end_v, 3)}
    return {"audio_duration": round(duration, 3), "warp": warp, "voice": "Adam", "voice_id": VOICE_ID,
            "model_id": MODEL_ID, "blocks": blocks, "sentences": sentences, "cues": cues,
            "words": words}


def retime():
    """Rebuild timing.json from the saved alignment (no API call)."""
    text = load_script()
    with open(os.path.join(HERE, "alignment.json")) as f:
        al = json.load(f)["alignment"]
    timing = build_timing(text, al, audio_duration(os.path.join(HERE, "narration.mp3")))
    with open(os.path.join(HERE, "timing.json"), "w") as f:
        json.dump(timing, f, indent=1)
    print("warp", json.dumps(timing["warp"]))


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
    al = res.get("alignment") or res.get("normalized_alignment")
    with open(os.path.join(HERE, "alignment.json"), "w") as f:
        json.dump({"alignment": res.get("alignment"),
                   "normalized_alignment": res.get("normalized_alignment")}, f)
    dur = audio_duration(mp3)
    timing = build_timing(text, al, dur)
    with open(os.path.join(HERE, "timing.json"), "w") as f:
        json.dump(timing, f, indent=1)
    print(f"audio {dur:.2f}s, {len(timing['sentences'])} sentences, {len(timing['words'])} words")
    print("warp", json.dumps(timing["warp"]))
    for b in timing["blocks"]:
        print(f"  {b['start']:6.2f}-{b['end']:6.2f}  {b['text'][:60]}")


if __name__ == "__main__":
    main()
