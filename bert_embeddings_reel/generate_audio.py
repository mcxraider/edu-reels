#!/usr/bin/env python3
"""Generate the BERT-embeddings narration with ElevenLabs (voice: Adam), one continuous take.

Outputs
  narration.mp3   raw Adam voiceover from /v1/text-to-speech/{voice}/with-timestamps
  alignment.json  raw character-level alignment PLUS the derived timeline:
                  words, sentences (audio + video start/end, scene), subtitle phrases,
                  named cues and the time-warp used by index.html and render.mjs

Usage
  export ELEVENLABS_API_KEY=sk_...      (or put it in ../.env or ./.env, never in code)
  python3 generate_audio.py             one API request
  python3 generate_audio.py --retime    rebuild the timeline from the saved alignment, no API call

Standard library only.
"""
import base64
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_NAME = "Adam"
VOICE_ID = "pNInz6obpgDQGcFmaJgB"          # ElevenLabs premade voice "Adam"
MODEL_ID = "eleven_multilingual_v2"
SPEED = 1.12                                # slight speed-up keeps the reel under 120 s
VOICE_SETTINGS = {"stability": 0.6, "similarity_boost": 0.8, "style": 0.0,
                  "use_speaker_boost": True, "speed": SPEED}

# ---- time-warp policy (narration time -> video time) ------------------------
LEAD_IN = 1.0      # silent lead-in before the narration
TAIL = 3.0         # final visual hold / silence tail after the narration
MAX_TOTAL = 120.0  # hard cap
# extra silence inserted inside the natural pause AFTER the sentence with this prefix:
# it gives scene transitions and key animations room to land.
PAUSES_AFTER = [
    ("Here's how", 0.3),            # -> tokenization
    ("Some words", 0.4),            # -> input embeddings
    ("BERT then adds", 0.5),        # -> encoder stack
    ("Inside each layer", 0.3),     # -> attention centerpiece
    ("For example", 0.4),           # sentence A attention lands
    ("But in the second", 0.4),     # sentence B attention lands
    ("Across multiple", 0.3),       # -> contextual embeddings
    ("So even though", 0.5),        # -> sentence embeddings
    ("That's why", 0.4),            # -> final takeaway
]

# sentence prefix -> animation scene
SCENE_OF = [
    ("How does BERT", "s1"), ("Let's take", "s1"), ("'I deposited", "s1"), ("And, ", "s1"),
    ("The word 'bank'", "s1"), ("Here's how", "s1"),
    ("First, it splits", "s2"), ("Some words", "s2"),
    ("Each token", "s3"), ("BERT then adds", "s3"),
    ("These vectors", "s4"), ("Inside each layer", "s4"),
    ("Every token", "s5"), ("For example", "s5"), ("But in the second", "s5"),
    ("Attention uses", "s5"), ("Across multiple", "s5"),
    ("So even though", "s6"),
    ("But what if", "s7"), ("We can pool", "s7"), ("Another approach", "s7"),
    ("However, vanilla", "s7"), ("That's why", "s7"),
    ("And that's the key", "s8"), ("It constructs", "s8"),
]

# named phrases whose (warped) timestamps drive animation beats
CUES = {
    "vectors0": "vectors that actually understand context",
    "two": "two sentences",
    "sentA": "I deposited money at the bank",
    "sentB": "I sat by the river bank",
    "word_bank": "The word 'bank'",
    "different": "clearly means different things",
    "handles": "Here's how BERT handles that",
    "splits": "it splits each sentence",
    "tokens": "called tokens",
    "whole": "Some words remain whole",
    "subwords": "broken into subwords",
    "mapped": "Each token is mapped",
    "learned_vec": "a learned vector",
    "positional": "positional information",
    "tokentype": "token-type information",
    "applicable": "when applicable",
    "stack": "a stack of Transformer encoder layers",
    "inside": "Inside each layer",
    "interesting": "something interesting happens",
    "bidir": "bidirectional self-attention",
    "every": "Every token can attend",
    "both_sides": "on both sides of the sentence",
    "ex_bank": "For example, 'bank' can attend",
    "deposited": "'deposited'",
    "money": "'money'",
    "second": "But in the second",
    "river": "'river'",
    "sat": "and 'sat'",
    "qkv": "query, key, and value projections",
    "flows": "how information flows between tokens",
    "heads": "multiple attention heads",
    "layers2": "and encoder layers",
    "updated": "progressively updated",
    "identical": "the original word is identical",
    "different_vec": "different contextual vectors",
    "one_vec": "one vector for the entire sentence",
    "pool": "pool the token representations",
    "mean": "taking their mean",
    "cls": "the special CLS token",
    "vanilla": "vanilla BERT wasn't specifically trained",
    "similarity": "sentence similarity embeddings",
    "sbert": "Sentence-BERT",
    "sbert_useful": "more useful sentence-level representations",
    "key_idea": "And that's the key idea",
    "lookup": "doesn't simply look up a word's meaning",
    "constructs": "It constructs representations",
    "surrounding": "the surrounding context",
}

# technical keywords highlighted in terracotta in the captions
KEYWORDS = ["BERT", "Sentence-BERT", "vectors", "vector", "context", "contextual", "tokens", "token",
            "subwords", "learned", "positional", "token-type", "Transformer", "encoder",
            "bidirectional", "self-attention", "attend", "query,", "key,", "value", "projections",
            "attention", "heads", "representation", "representations", "bank", "pool", "mean",
            "CLS", "similarity", "sentence-level", "fine-tune"]


def load_env():
    for p in (os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*$", line)
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def load_script():
    with open(os.path.join(HERE, "narration.txt"), encoding="utf-8") as f:
        # paragraphs joined by single newlines (identical words; avoids long paragraph pauses)
        return "\n".join(p.strip() for p in f.read().strip().split("\n\n"))


def tts_with_timestamps(text, key):
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps"
           "?output_format=mp3_44100_128")
    body = json.dumps({"text": text, "model_id": MODEL_ID, "voice_settings": VOICE_SETTINGS}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "xi-api-key": key, "Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def audio_duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "default=nw=1:nk=1", path])
    return float(out)


def span_times(al, start, end):
    """Start/end seconds of text[start:end], trimming whitespace and quote marks."""
    chars = al["characters"]
    s, e = start, end - 1
    while s < e and (chars[s].isspace() or chars[s] in "'\""):
        s += 1
    while e > s and (chars[e].isspace() or chars[e] in "'\""):
        e -= 1
    return al["character_start_times_seconds"][s], al["character_end_times_seconds"][e]


def build_timeline(text, al, duration):
    assert "".join(al["characters"]) == text, "alignment characters do not match narration.txt"
    r3 = lambda x: round(x, 3)

    # ---- sentences: end at . ? ! (optionally followed by a closing quote)
    sentences, st = [], 0
    for m in re.finditer(r"[.?!]['\"]?(?=\s|$)", text):
        en = m.end()
        seg = text[st:en]
        a = st + (len(seg) - len(seg.lstrip()))
        s, e = span_times(al, a, en)
        sentences.append({"index": len(sentences), "text": text[a:en], "char_start": a,
                          "char_end": en, "start": r3(s), "end": r3(e)})
        st = en
    for x in sentences:
        x["scene"] = next(sc for p, sc in SCENE_OF if x["text"].startswith(p))

    # ---- words, straight from character spans
    words = []
    for m in re.finditer(r"\S+", text):
        s, e = span_times(al, m.start(), m.end())
        words.append({"w": m.group(0), "char_start": m.start(), "char_end": m.end(),
                      "start": r3(s), "end": r3(e)})

    def find(phrase):
        i = text.index(phrase)
        s, e = span_times(al, i, i + len(phrase))
        return {"phrase": phrase, "char_start": i, "start": r3(s), "end": r3(e)}
    cues = {k: find(p) for k, p in CUES.items()}

    # ---- time warp
    inserts = []
    for prefix, dur in PAUSES_AFTER:
        i = next(k for k, x in enumerate(sentences) if x["text"].startswith(prefix))
        a, b = sentences[i]["end"], sentences[i + 1]["start"]
        assert b - a > 0.1, f"no natural pause after {prefix!r} ({a}->{b})"
        inserts.append({"at": r3((a + b) / 2), "dur": dur, "after": prefix})

    def warp(n):
        return LEAD_IN + n + sum(x["dur"] for x in inserts if x["at"] <= n)

    for x in sentences + words:
        x["video_start"] = r3(warp(x["start"]))
        x["video_end"] = r3(warp(x["end"]))
    for c in cues.values():
        c["video_start"] = r3(warp(c["start"]))
        c["video_end"] = r3(warp(c["end"]))
    narr_end_v = LEAD_IN + duration + sum(x["dur"] for x in inserts)
    last_word_v = warp(sentences[-1]["end"])
    total = round(last_word_v + TAIL + 0.3, 2)
    total = max(total, round(narr_end_v + 0.2, 2))
    assert total <= MAX_TOTAL, f"video would be {total}s (> {MAX_TOTAL}s); raise SPEED or trim pauses"

    # ---- scene windows (video time): from the first sentence of a scene to the next scene
    order = []
    for x in sentences:
        if not order or order[-1]["scene"] != x["scene"]:
            order.append({"scene": x["scene"], "first": x["index"]})
    scenes = []
    for k, o in enumerate(order):
        first = sentences[o["first"]]
        start = 0.0 if k == 0 else (sentences[o["first"] - 1]["video_end"] + first["video_start"]) / 2
        scenes.append({"scene": o["scene"], "start": r3(start)})
    for k, s in enumerate(scenes):
        s["end"] = scenes[k + 1]["start"] if k + 1 < len(scenes) else total

    # ---- subtitle phrases: clauses (split at , : ;) then balanced chunks of <= MAXC chars;
    #      tiny fragments merge into a neighbour. Drawn on <= 2 lines in index.html.
    MAXC, MINC, MERGE = 40, 14, 46
    phrases = []
    tlen = lambda ch: len(" ".join(w["w"] for w in ch))
    for sen in sentences:
        ws = [w for w in words if sen["char_start"] <= w["char_start"] < sen["char_end"]]
        clauses, cur = [], []
        for w in ws:
            cur.append(w)
            if re.search(r"[,:;]['\"]?$", w["w"]):
                clauses.append(cur); cur = []
        if cur:
            clauses.append(cur)
        chunks = []
        for cl in clauses:
            n = max(1, -(-tlen(cl) // MAXC))
            # greedy balanced split into n parts by character length
            target = tlen(cl) / n
            part = []
            for w in cl:
                if part and tlen(part + [w]) > target + 6 and len(chunks) >= 0 and n > 1:
                    chunks.append(part); part = []; n -= 1
                    target = tlen(cl[cl.index(w):]) / max(1, n)
                part.append(w)
            chunks.append(part)
        merged = []
        for ch in chunks:
            if merged and (tlen(ch) < MINC or tlen(merged[-1]) < MINC) and tlen(merged[-1] + ch) <= MERGE:
                merged[-1] = merged[-1] + ch
            else:
                merged.append(ch)
        for cur in merged:
            phrases.append({"text": " ".join(c["w"] for c in cur), "words": [c["w"] for c in cur],
                            "word_times": [[c["video_start"], c["video_end"]] for c in cur],
                            "start": cur[0]["start"], "end": cur[-1]["end"],
                            "video_start": cur[0]["video_start"], "video_end": cur[-1]["video_end"],
                            "sentence": sen["index"], "scene": sen["scene"]})
    for k, p in enumerate(phrases):   # subtitle display window: until the next phrase (max +0.6 s)
        nxt = phrases[k + 1]["video_start"] if k + 1 < len(phrases) else total
        p["show_from"] = r3(p["video_start"] - 0.05)
        p["show_until"] = r3(min(nxt - 0.02, p["video_end"] + 0.6))

    return {
        "voice": VOICE_NAME, "voice_id": VOICE_ID, "model_id": MODEL_ID, "voice_settings": VOICE_SETTINGS,
        "script": text, "audio_duration": r3(duration),
        "warp": {"lead_in": LEAD_IN, "tail": TAIL, "inserts": inserts, "narration_end_video": r3(narr_end_v),
                 "total": total, "fps": 30, "frames": int(round(total * 30))},
        "scenes": scenes, "sentences": sentences, "subtitles": phrases, "cues": cues,
        "keywords": KEYWORDS, "words": words,
    }


def write_timeline(tl, raw):
    tl["raw_response_alignment"] = raw
    with open(os.path.join(HERE, "alignment.json"), "w") as f:
        json.dump(tl, f, indent=1)
    print("warp", json.dumps({k: v for k, v in tl["warp"].items() if k != "inserts"}))
    for s in tl["scenes"]:
        print(f"  scene {s['scene']}: {s['start']:6.2f} - {s['end']:6.2f}")
    for s in tl["sentences"]:
        print(f"  audio {s['start']:6.2f}-{s['end']:6.2f}  video {s['video_start']:6.2f}-{s['video_end']:6.2f}"
              f"  [{s['scene']}] {s['text'][:60]}")


def main():
    text = load_script()
    if "--retime" in sys.argv:
        with open(os.path.join(HERE, "alignment.json")) as f:
            raw = json.load(f)["raw_response_alignment"]
    else:
        load_env()
        key = os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            sys.exit("Set ELEVENLABS_API_KEY")
        res = tts_with_timestamps(text, key)
        with open(os.path.join(HERE, "narration.mp3"), "wb") as f:
            f.write(base64.b64decode(res["audio_base64"]))
        raw = {"alignment": res.get("alignment"), "normalized_alignment": res.get("normalized_alignment")}
        with open(os.path.join(HERE, "alignment.json"), "w") as f:   # saved before any validation
            json.dump({"raw_response_alignment": raw}, f)
    al = raw["alignment"] or raw["normalized_alignment"]
    dur = audio_duration(os.path.join(HERE, "narration.mp3"))
    print(f"narration audio {dur:.2f}s")
    write_timeline(build_timeline(text, al, dur), raw)


if __name__ == "__main__":
    main()
