import math, json, sys, subprocess, difflib, re, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, SS, FPS = 1080, 1920, 2, 30
BG = (10, 14, 30); GRID = (19, 26, 50); PANEL = (18, 25, 48)
WHITE = (240, 244, 255); DIM = (128, 140, 175); CYAN = (80, 220, 210); AMBER = (255, 184, 77)
PINK = (255, 110, 150); GREEN = (110, 230, 140); RED = (255, 92, 92); VIOLET = (165, 135, 255)
AV = "/System/Library/Fonts/Avenir Next.ttc"; MENLO = "/System/Library/Fonts/Menlo.ttc"
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- timeline ----------------
STARTS = [0.5, 3.3, 17.0, 31.0, 34.2, 57.4, 64.5, 69.7, 74.9]
DURS = [2.27, 13.24, 13.64, 2.69, 22.67, 6.69, 4.75, 4.75, 3.63]
TOTAL = 82.0
SCRIPT = [
    "Q9: Context extension",
    "Context: You extend a 4k-context model to 32k using linear position interpolation, dividing every position by 8, and then fine-tune briefly on long documents.",
    "Long-context retrieval works. But short-context benchmarks (under 2k tokens) drop noticeably, and the model gets worse at precise local tasks such as reproducing exact token order.",
    "Question: What is the most likely cause?",
    "A. Uniform interpolation also compresses the high-frequency RoPE dimensions. Those dimensions already resolved short distances well and never needed extrapolation, so nearby tokens now look closer together than they did in pretraining. NTK-by-parts and YaRN leave those dimensions mostly unscaled.",
    "B. Fine-tuning on long documents caused forgetting, and mixing short data back in would fully fix it.",
    "C. The attention softmax temperature is too high at short lengths.",
    "D. The RoPE base should have been decreased, not effectively increased.",
    "Which would you pick, and what ablation would isolate the cause?",
]

def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())

def align(authored, whisper):
    wt = []
    for w, a, b in whisper:
        if w.startswith("-") and wt: wt[-1] = [wt[-1][0] + w, wt[-1][1], b]
        else: wt.append([w, a, b])
    A = authored.split(); An = [norm(x) for x in A]; Wn = [norm(x[0]) for x in wt]
    times = [None] * len(A)
    for blk in difflib.SequenceMatcher(None, An, Wn, autojunk=False).get_matching_blocks():
        for k in range(blk.size): times[blk.a + k] = (wt[blk.b + k][1], wt[blk.b + k][2])
    # fill gaps by interpolation
    i = 0
    while i < len(A):
        if times[i] is None:
            j = i
            while j < len(A) and times[j] is None: j += 1
            t0 = times[i - 1][1] if i > 0 else 0.0
            t1 = times[j][0] if j < len(A) else wt[-1][2]
            lens = [max(1, len(An[k])) for k in range(i, j)]; tot = sum(lens); acc = t0
            for k, L in zip(range(i, j), lens):
                d = (t1 - t0) * L / tot; times[k] = (acc, acc + d); acc += d
            i = j
        else: i += 1
    return list(zip(A, times))

WORDS = json.load(open(os.path.join(HERE, "words.json")))
CHUNKS = []  # (t_start, t_end, [(word, ws, we)])
for si in range(9):
    al = align(SCRIPT[si], WORDS[str(si + 1)])
    al = [(w, a + STARTS[si], b + STARTS[si]) for w, (a, b) in al]
    cur = []
    for w in al:
        cur.append(w)
        chars = sum(len(x[0]) + 1 for x in cur)
        if chars > 20 or len(cur) >= 4 or w[0][-1] in ".,:?)":
            CHUNKS.append(cur); cur = []
    if cur: CHUNKS.append(cur)
CH = []
for k, c in enumerate(CHUNKS):
    st = c[0][1]
    nxt = CHUNKS[k + 1][0][1] if k + 1 < len(CHUNKS) else 1e9
    en = min(nxt, c[-1][2] + 0.6)
    CH.append((st, en, c))

# ---------------- helpers ----------------
_fc = {}
def F(size, w="bold", mono=False):
    idx = {"bold": 0, "demi": 2, "med": 5, "reg": 7, "heavy": 8}[w]
    k = (size, idx, mono)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(MENLO if mono else AV, int(size * SS), index=(1 if mono and w == "bold" else 0) if mono else idx)
    return _fc[k]
SYM = set("→✓✗↓÷⅛×·")
_fs = {}
def FS(size):
    if size not in _fs: _fs[size] = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", int(size * SS))
    return _fs[size]
def cl(x, a=0.0, b=1.0): return max(a, min(b, x))
def pr(u, a, b): return cl((u - a) / (b - a)) if b > a else float(u >= a)
def eo(x): return 1 - (1 - x) ** 3
def eio(x): return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def back(x):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def lerp(a, b, t): return a + (b - a) * t
def win(u, a, b, f=0.3): return min(pr(u, a, a + f), 1 - pr(u, b - f, b))

class C:
    def __init__(s, d, A=1.0): s.d = d; s.A = A
    def col(s, c, a=1.0, bg=BG):
        a = cl(a * s.A); return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))
    def line(s, pts, c, w=3, a=1.0):
        if a * s.A <= 0.01: return
        s.d.line([(x * SS, y * SS) for x, y in pts], fill=s.col(c, a), width=int(w * SS), joint="curve")
    def circ(s, x, y, r, fill=None, outline=None, w=3, a=1.0):
        if a * s.A <= 0.01 or r <= 0: return
        s.d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
                    fill=s.col(fill, a) if fill else None, outline=s.col(outline, a) if outline else None, width=int(w * SS))
    def rr(s, x0, y0, x1, y1, r=18, fill=None, outline=None, w=2, a=1.0, bg=BG):
        if a * s.A <= 0.01 or x1 - x0 < 1 or y1 - y0 < 1: return
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
        s.d.rounded_rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], radius=r * SS,
                              fill=s.col(fill, a, bg) if fill else None, outline=s.col(outline, a, bg) if outline else None, width=int(w * SS))
    def t(s, x, y, txt, size, c=WHITE, a=1.0, anchor="mm", w="bold", mono=False, bg=BG):
        if a * s.A <= 0.01 or not txt: return
        if any(ch in SYM for ch in txt):
            runs = []
            for ch in txt:
                f = FS(size) if ch in SYM else F(size, w, mono)
                if runs and runs[-1][1] is f: runs[-1][0] += ch
                else: runs.append([ch, f])
            tot = sum(f.getlength(r) for r, f in runs) / SS
            x0 = x - tot / 2 if anchor[0] == "m" else (x - tot if anchor[0] == "r" else x)
            for r, f in runs:
                s.d.text((x0 * SS, y * SS), r, font=f, fill=s.col(c, a, bg), anchor="l" + anchor[1])
                x0 += f.getlength(r) / SS
            return
        s.d.text((x * SS, y * SS), txt, font=F(size, w, mono), fill=s.col(c, a, bg), anchor=anchor)
    def tw(s, txt, size, w="bold", mono=False): return sum((FS(size) if ch in SYM else F(size, w, mono)).getlength(ch) for ch in txt) / SS
    def pie(s, x, y, r, a0, a1, c, a=1.0):
        if a * s.A <= 0.01: return
        s.d.pieslice([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], math.degrees(a0) - 90, math.degrees(a1) - 90, fill=s.col(c, a))
    def arc(s, x, y, r, a0, a1, c, w=4, a=1.0):
        if a * s.A <= 0.01: return
        s.d.arc([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], math.degrees(a0) - 90, math.degrees(a1) - 90, fill=s.col(c, a), width=int(w * SS))

def hand(c, x, y, r, ang, col, w=6, a=1.0, dot=True):
    ex, ey = x + r * math.sin(ang), y - r * math.cos(ang)
    c.line([(x, y), (ex, ey)], col, w, a)
    if dot: c.circ(ex, ey, w * 1.6, fill=col, a=a)

def dial(c, x, y, r, ang, col, a=1.0):
    c.circ(x, y, r, fill=PANEL, a=a)
    c.circ(x, y, r, outline=mix(DIM, 0.6), w=3, a=a)
    for k in range(12):
        q = k * math.pi / 6
        c.line([(x + r * .86 * math.sin(q), y - r * .86 * math.cos(q)), (x + r * .96 * math.sin(q), y - r * .96 * math.cos(q))], DIM, 2, a * .6)
    hand(c, x, y, r * .8, ang, col, 5, a)
    c.circ(x, y, 7, fill=WHITE, a=a)

def mix(cc, a, bg=BG): return tuple(int(bg[i] + (cc[i] - bg[i]) * a) for i in range(3))

def doc(c, x, y, w, h, col, a=1.0):
    c.rr(x, y, x + w, y + h, 10, fill=PANEL, outline=col, w=3, a=a)
    for k in range(4):
        yy = y + h * (0.25 + 0.17 * k)
        c.line([(x + w * .18, yy), (x + w * (0.82 if k < 3 else 0.55), yy)], col, 3, a * .8)

def badge(c, u, letter, col, title):
    p = eo(pr(u, -0.2, 0.25))
    c.circ(150, 455, 54 * back(pr(u, -0.2, 0.3)), fill=col, a=p)
    c.t(150, 458, letter, 58, BG, a=p, w="heavy", bg=col)
    c.t(228 + 30 * (1 - p), 455, title, 46, WHITE, a=p, anchor="lm")

def pill(c, x, y, txt, size, col, a=1.0, anchor="mm", mono=False, pad=22):
    tw = c.tw(txt, size, mono=mono); hh = size * 0.75 + pad
    if anchor == "mm": x0 = x - tw / 2 - pad
    elif anchor == "lm": x0 = x - pad * 0 
    else: x0 = x - tw - 2 * pad
    c.rr(x0, y - hh / 2 - 4, x0 + tw + 2 * pad, y + hh / 2 + 4, hh / 2 + 4, fill=mix(col, 0.16), outline=col, w=2, a=a)
    c.t(x0 + pad + tw / 2, y + 1, txt, size, col, a=a, mono=mono)

# ---------------- scenes ----------------
def s1(c, u):
    p = pr(u, 0, 0.45)
    sz = int(120 + 120 * back(p)) if p > 0 else 0
    if sz: c.t(540, 690, "Q9", sz, AMBER, a=eo(p), w="heavy")
    s = "CONTEXT EXTENSION"; n = int(len(s) * pr(u, 0.88, 1.8))
    c.t(540, 880, s[:n], 66, WHITE, w="heavy")
    sp = [2.0, 0.9, 0.4, 0.15, 0.05]
    for k in range(5):
        a = eo(pr(u, -0.3 + 0.12 * k, 0.2 + 0.12 * k))
        dial(c, 150 + 195 * k, 1100, 66, 2 * math.pi * sp[k] * (u + 1), [CYAN, CYAN, VIOLET, AMBER, AMBER][k], a)
    c.t(540, 1235, "RoPE  ·  position interpolation", 34, DIM, a=pr(u, 1.2, 1.7), w="demi")

def s2(c, u):
    c.t(90, 445, "THE SETUP", 34, CYAN, a=pr(u, -0.2, 0.2), anchor="lm", w="heavy")
    # context window bar
    a = pr(u, 1.5, 1.9)
    c.t(90, 520, "context window", 32, DIM, a=a, anchor="lm", w="demi")
    w4 = 112.5 * eo(pr(u, 2.0, 2.4)); wx = (900 - 112.5) * eio(pr(u, 3.86, 4.7))
    if w4 > 1:
        c.rr(90, 550, 90 + w4, 620, 12, fill=CYAN, a=a)
    if wx > 1:
        c.rr(90 + 112.5 + 4, 550, 90 + 112.5 + 4 + wx - 4, 620, 12, fill=AMBER, a=a)
    c.t(90 + 56, 585, "4k", 32, BG, a=pr(u, 2.2, 2.5), bg=CYAN)
    c.t(90 + 112.5 + wx - 20, 585, "32k", 32, BG, a=pr(u, 4.3, 4.7), anchor="rm", bg=AMBER)
    if u > 4.4:
        pill(c, 560, 660, "× 8 longer", 30, PINK, a=pr(u, 4.4, 4.8))
    # positions ruler
    a2 = pr(u, 5.3, 5.8)
    c.t(90, 760, "positions seen in pretraining:  0 → 4k", 32, DIM, a=a2, anchor="lm", w="demi")
    c.line([(90, 905), (990, 905)], GREEN, 8, a2)
    for xx, lb in [(90, "0"), (990, "4k")]:
        c.line([(xx, 890), (xx, 920)], GREEN, 4, a2)
        c.t(xx, 950, lb, 30, GREEN, a=a2)
    f = eio(pr(u, 7.9, 9.2))
    for k in range(64):
        x = 90 + k * 112.5 * (1 - f * 7 / 8)
        if x > 1075: continue
        inside = x <= 992
        ak = a2 * eo(pr(u, 5.4 + k * 0.02, 5.7 + k * 0.02))
        c.circ(x, 860, 9 if f < 0.5 else lerp(9, 6, f), fill=CYAN if inside else RED, a=ak)
    c.t(990, 815, "4k–32k: never seen", 28, RED, a=a2 * (1 - pr(u, 7.6, 8.0)), anchor="rm", w="demi")
    if u > 7.52:
        pill(c, 540, 1010, "position  p  →  p / 8", 34, AMBER, a=pr(u, 7.52, 7.9), mono=True)
    c.t(540, 815, "all 32k tokens squeezed into 0 → 4k", 28, CYAN, a=pr(u, 9.0, 9.4), w="demi")
    # fine-tune
    for k in range(3):
        p = eo(pr(u, 10.26 + 0.12 * k, 10.7 + 0.12 * k))
        doc(c, 100 + 115 * k + 200 * (1 - p), 1095, 90, 120, VIOLET, p)
    a3 = pr(u, 10.4, 10.8)
    c.t(470, 1125, "brief fine-tune on long docs", 32, WHITE, a=a3, anchor="lm", w="demi")
    c.rr(470, 1170, 990, 1190, 10, fill=PANEL, a=a3)
    fw = 520 * pr(u, 10.8, 12.7)
    if fw > 20: c.rr(470, 1170, 470 + fw, 1190, 10, fill=VIOLET, a=a3)

def s3(c, u):
    # panel 1 needle retrieval
    a1 = pr(u, -0.3, 0.1)
    c.t(90, 450, "long-context retrieval", 34, WHITE, a=a1, anchor="lm", w="demi")
    found = pr(u, 1.2, 1.5)
    c.t(990, 450, "✓ works", 34, GREEN, a=found, anchor="rm")
    n = 30; cw = 28; x0 = 90
    scan = 22 * eio(pr(u, 0.1, 1.2))
    for k in range(n):
        x = x0 + k * 30
        col = AMBER if k == 22 else mix(DIM, 0.45)
        c.rr(x, 500, x + cw, 560, 6, fill=col, a=a1)
    c.rr(x0 + scan * 30 - 4, 492, x0 + scan * 30 + cw + 4, 568, 8, outline=GREEN if found > 0 else CYAN, w=4, a=a1)
    c.t(x0 + 22 * 30 + 14, 600, "needle", 26, AMBER, a=found, w="demi")
    # panel 2 short benchmarks
    a2 = pr(u, 2.42, 2.8)
    c.t(90, 690, "short-context benchmarks  (< 2k tokens)", 34, WHITE, a=a2, anchor="lm", w="demi")
    base = 1000; hmax = 230
    c.line([(150, base), (930, base)], DIM, 3, a2)
    hb = hmax * eo(pr(u, 2.6, 3.2))
    if hb > 2: c.rr(250, base - hb, 430, base, 10, fill=CYAN, a=a2)
    drop = eio(pr(u, 5.7, 6.6))
    ha = hmax * eo(pr(u, 3.0, 3.6)) * (1 - 0.38 * drop)
    if ha > 2: c.rr(650, base - ha, 830, base, 10, fill=CYAN if drop < 0.5 else RED, a=a2)
    c.t(340, base + 32, "before", 28, DIM, a=a2, w="demi")
    c.t(740, base + 32, "after 32k extension", 28, DIM, a=a2, w="demi")
    c.t(740, base - ha - 36, "↓ drops", 32, RED, a=pr(u, 6.3, 6.7))
    # panel 3 token order
    a3 = pr(u, 7.18, 7.6)
    c.t(90, 1095, "precise local task: copy exact token order", 32, WHITE, a=a3, anchor="lm", w="demi")
    toks = ["the", "cat", "sat", "on", "the", "mat"]
    widths = [c.tw(t, 32) + 36 for t in toks]; gap = 14
    xs = [];
    x = 230
    for wdt in widths: xs.append(x); x += wdt + gap
    c.t(90, 1160, "input", 28, DIM, a=a3, anchor="lm", w="demi")
    for k, t in enumerate(toks):
        c.rr(xs[k], 1135, xs[k] + widths[k], 1185, 12, fill=PANEL, outline=CYAN, w=2, a=a3 * eo(pr(u, 7.3 + .08 * k, 7.6 + .08 * k)))
        c.t(xs[k] + widths[k] / 2, 1160, t, 32, WHITE, a=a3 * eo(pr(u, 7.3 + .08 * k, 7.6 + .08 * k)))
    a4 = pr(u, 11.0, 11.4)
    c.t(90, 1240, "output", 28, DIM, a=a4, anchor="lm", w="demi")
    sw = eio(pr(u, 11.9, 12.7))
    perm = {1: 2, 2: 1, 3: 4, 4: 3}
    for k, t in enumerate(toks):
        tgt = perm.get(k, k)
        cx = lerp(xs[k] + widths[k] / 2, xs[tgt] + widths[tgt] / 2, sw)
        wob = 10 * math.sin(sw * math.pi) * (1 if k % 2 else -1) if k in perm else 0
        bad = k in perm and sw > 0.5
        wdt = widths[k]
        c.rr(cx - wdt / 2, 1215 + wob, cx + wdt / 2, 1265 + wob, 12, fill=PANEL, outline=RED if bad else CYAN, w=2, a=a4)
        c.t(cx, 1240 + wob, t, 32, RED if bad else WHITE, a=a4)
    c.t(990, 1240, "✗", 44, RED, a=pr(u, 12.6, 12.9), anchor="rm")

def s4(c, u):
    p = pr(u, -0.1, 0.35)
    c.t(540, 700 + 30 * (1 - eo(p)), "What is the most", 86, WHITE, a=eo(p), w="heavy")
    p2 = pr(u, 0.5, 0.9)
    c.t(540, 810 + 30 * (1 - eo(p2)), "likely cause?", 86, AMBER, a=eo(p2), w="heavy")
    for k, (L, col) in enumerate(zip("ABCD", [CYAN, AMBER, PINK, VIOLET])):
        q = pr(u, 0.9 + 0.15 * k, 1.25 + 0.15 * k)
        r = 62 * back(q) if q > 0 else 0
        x = 255 + 190 * k
        c.circ(x, 1050, r, fill=col, a=eo(q))
        c.t(x, 1053, L, 56, BG, a=eo(q), w="heavy", bg=col)

TURNS = ["652", "65", "6.5", "0.65", "0.08"]
SPEEDS = [2.0, 0.9, 0.4, 0.15, 0.05]
def ueff(u):
    if u < 2.5: return u
    n = 24; tot = 0; hi = u
    for k in range(n):
        s = 2.5 + (hi - 2.5) * (k + .5) / n
        tot += (1 - 7 / 8 * eio(pr(s, 2.5, 3.3))) * (hi - 2.5) / n
    return 2.5 + tot

def s5(c, u):
    badge(c, u, "A", CYAN, "uniform interpolation")
    # phase 1: dial row
    a1 = win(u, -0.4, 11.9, 0.4)
    if a1 > 0:
        c.t(90, 555, "high freq", 30, CYAN, a=a1, anchor="lm", w="demi")
        c.t(990, 555, "low freq", 30, AMBER, a=a1, anchor="rm", w="demi")
        c.line([(250, 555), (820, 555)], DIM, 2, a1 * .6)
        c.line([(805, 545), (820, 555), (805, 565)], DIM, 2, a1 * .6)
        ue = ueff(u)
        cols = [CYAN, CYAN, VIOLET, AMBER, AMBER]
        for k in range(5):
            dial(c, 150 + 195 * k, 665, 72, 2 * math.pi * SPEEDS[k] * (ue + 3), cols[k], a1)
        st = win(u, 2.5, 6.0, 0.35)
        c.line([(80, 760), (80, 778), (1000, 778), (1000, 760)], PINK, 4, st * a1)
        if st > 0: pill(c, 540, 830, "÷ 8 on every dimension", 34, PINK, a=st * a1)
        ta = pr(u, 5.9, 6.4) * a1
        for k in range(5):
            c.t(150 + 195 * k, 780, TURNS[k], 34, WHITE, a=ta, w="heavy")
        c.t(540, 830, "full rotations inside the 4k pretraining window", 28, DIM, a=ta, w="demi")
        g = pr(u, 7.2, 7.6) * a1
        c.rr(64, 575, 432, 812, 22, outline=GREEN, w=4, a=g)
        c.t(248, 900, "short distances:", 30, GREEN, a=g, w="demi")
        c.t(248, 942, "already resolved ✓", 30, GREEN, a=g)
        am = pr(u, 10.3, 10.7) * a1
        c.rr(648, 575, 1016, 812, 22, outline=AMBER, w=4, a=am)
        c.t(832, 900, "long distances:", 30, AMBER, a=am, w="demi")
        c.t(832, 942, "need extrapolation", 30, AMBER, a=am)
        fa = pr(u, 0.6, 1.0) * a1
        c.t(540, 1060, "angle(p) = p · θᵢ      θᵢ = base^(−2i/d)", 30, DIM, a=fa, mono=True)
        c.t(540, 1110, "each dimension pair is a clock spinning at its own speed", 28, DIM, a=fa, w="med")
    # phase 2: neighbors on the fastest dial
    a2 = win(u, 11.7, 17.4, 0.4)
    if a2 > 0:
        c.t(540, 548, "fastest dimension  ·  neighbors p and p+1", 32, WHITE, a=a2, w="demi")
        cx, cy, r = 540, 860, 190
        c.circ(cx, cy, r, fill=PANEL, a=a2)
        c.circ(cx, cy, r, outline=DIM, w=3, a=a2 * .6)
        g0 = 1.0; g1 = 1.0 / 8
        sh = eio(pr(u, 13.9, 14.9))
        gap = lerp(g0, g1, sh)
        base = -0.9 + 0.15 * math.sin(u * 1.3)
        c.pie(cx, cy, r * .7, base, base + gap, mix(PINK, 0.35), a=a2)
        hand(c, cx, cy, r * .82, base, CYAN, 8, a2)
        hand(c, cx, cy, r * .82, base + gap, AMBER, 8, a2)
        c.circ(cx, cy, 10, fill=WHITE, a=a2)
        c.t(cx + (r + 48) * math.sin(base), cy - (r + 48) * math.cos(base), "p", 34, CYAN, a=a2)
        c.t(cx + (r + 60) * math.sin(base + gap + 0.12), cy - (r + 60) * math.cos(base + gap + 0.12), "p+1", 34, AMBER, a=a2)
        deg = int(round(math.degrees(gap)))
        c.t(540, 1110, f"{deg}° apart", 60, PINK if sh > 0.5 else WHITE, a=a2, w="heavy")
        c.t(540, 1175, "pretraining: 57°", 30, DIM, a=a2 * pr(u, 14.6, 15.0), w="demi")
        c.t(540, 1218, "now neighbors look ~8× closer", 30, PINK, a=a2 * pr(u, 14.9, 15.3), w="demi")
    # phase 3: per-dimension scale graph
    a3 = win(u, 17.0, 30, 0.4)
    if a3 > 0:
        ox, oy, xr, ytop, ylo = 150, 1110, 990, 640, 1060
        c.t(540, 565, "how much each dimension is scaled", 32, WHITE, a=a3, w="demi")
        c.line([(ox, ytop - 30), (ox, oy), (xr, oy)], DIM, 3, a3)
        c.t(ox - 20, ytop, "1×", 30, DIM, a=a3, anchor="rm")
        c.t(ox - 20, ylo, "⅛", 34, DIM, a=a3, anchor="rm")
        c.t(ox, oy + 40, "high freq", 28, CYAN, a=a3, anchor="lm", w="demi")
        c.t(xr, oy + 40, "low freq", 28, AMBER, a=a3, anchor="rm", w="demi")
        pl = eo(pr(u, 17.3, 18.0))
        if pl > 0:
            c.line([(ox, ylo), (lerp(ox, xr, pl), ylo)], RED, 7, a3)
        c.t(xr, ylo + 26, "linear PI: ÷8 everywhere", 28, RED, a=a3 * pr(u, 17.7, 18.1), anchor="rm", w="demi")
        py = eo(pr(u, 18.8, 20.0))
        def yv(x):
            if x <= 430: return ytop
            if x >= 790: return ylo
            q = (x - 430) / 360; q = q * q * (3 - 2 * q); return lerp(ytop, ylo, q)
        if py > 0:
            xe = lerp(ox, xr, py); pts = [(x, yv(x)) for x in np.linspace(ox, xe, 60)]
            c.line(pts, GREEN, 8, a3)
        c.t(ox + 10, ytop - 40, "NTK-by-parts / YaRN", 30, GREEN, a=a3 * pr(u, 18.9, 19.3), anchor="lm")
        hl = pr(u, 20.1, 20.6) * a3
        c.rr(ox + 6, ytop + 20, 430, oy - 8, 14, fill=mix(GREEN, 0.10), outline=GREEN, w=2, a=hl)
        c.t(290, 800, "left", 30, GREEN, a=hl, w="demi")
        c.t(290, 842, "unscaled", 30, GREEN, a=hl)

def s6(c, u):
    badge(c, u, "B", AMBER, "forgetting from long-doc tuning")
    c.t(540, 580, "short-context skills", 32, DIM, a=pr(u, 0.0, 0.4), w="demi")
    lost = {1, 4, 6, 9, 10, 13, 15, 16}
    x0, y0, s, g = 215, 630, 90, 22
    f1 = pr(u, 2.2, 3.1); f2 = eo(pr(u, 4.2, 5.3))
    for k in range(18):
        i, j = k % 6, k // 6
        x, y = x0 + i * (s + g), y0 + j * (s + g)
        a = eo(pr(u, 0.1 + 0.02 * k, 0.4 + 0.02 * k))
        col = CYAN
        if k in lost:
            q = f1 * (1 - f2)
            col = mix(CYAN, 1 - 0.85 * q)
            if f2 > 0 and f2 < 1: col = mix(GREEN, 0.4 + 0.6 * f2) if f1 >= 1 else col
            if f2 >= 1: col = GREEN
        c.rr(x, y, x + s, y + s, 14, fill=col, a=a)
    for k in range(4):
        p = eo(pr(u, 0.5 + 0.15 * k, 1.0 + 0.15 * k)) * (1 - pr(u, 2.6, 3.0))
        doc(c, 1000 - 70 * p + 30 * k - 150, 980 - 20 * k, 70, 95, VIOLET, p)
    c.t(540, 1010, "long docs in → short skills fade", 30, VIOLET, a=pr(u, 1.0, 1.4) * (1 - pr(u, 3.4, 3.7)), w="demi")
    c.t(540, 1010, "+ mix short data back in", 32, GREEN, a=pr(u, 3.8, 4.2), w="demi")
    q = pr(u, 5.4, 5.8)
    c.t(540, 1110, "fully fix it?", 56 + 4 * math.sin(u * 6) * q, AMBER, a=q, w="heavy")

LOG = np.array([0.2, 0.6, 3.0, 1.0, 0.3, 2.1, 0.1, 0.5])
def s7(c, u):
    badge(c, u, "C", PINK, "softmax temperature")
    c.t(540, 590, "attention = softmax( q·k / T )", 34, WHITE, a=pr(u, 0.0, 0.4), mono=True)
    T = lerp(1.0, 3.5, eio(pr(u, 1.9, 3.4)))
    p = np.exp(LOG / T); p /= p.sum(); p1 = np.exp(LOG); p1 /= p1.sum()
    c.t(540, 660, f"T = {T:.1f}", 40, PINK, a=pr(u, 0.2, 0.6), w="heavy")
    base = 1080; hm = 320
    for k in range(8):
        a = eo(pr(u, 0.2 + 0.05 * k, 0.5 + 0.05 * k))
        h = hm * p[k] / p1.max() * a
        x = 170 + 95 * k
        c.rr(x, base - max(h, 4), x + 70, base, 10, fill=VIOLET if T < 2 else PINK, a=a)
        c.t(x + 35, base + 30, f"t{k+1}", 24, DIM, a=a, w="demi")
    c.line([(150, base), (930, base)], DIM, 3, pr(u, 0.2, 0.5))
    c.t(540, 1170, "too high → attention flattens out", 30, DIM, a=pr(u, 3.2, 3.6), w="demi")

def s8(c, u):
    badge(c, u, "D", VIOLET, "the RoPE base")
    cx, cy, r = 540, 800, 190
    a = pr(u, 0.0, 0.4)
    c.arc(cx, cy, r, -2.36, 2.36, mix(DIM, .5), 14, a)
    c.t(cx - r - 10, cy + r * .8, "decrease", 28, CYAN, a=a, anchor="rm", w="demi")
    c.t(cx + r + 10, cy + r * .8, "increase", 28, AMBER, a=a, anchor="lm", w="demi")
    ang = 0.0
    ang = lerp(ang, -1.9, eio(pr(u, 2.2, 2.8)))
    ang = lerp(ang, 1.9, eio(pr(u, 3.3, 4.0)))
    col = CYAN if ang < -0.3 else (AMBER if ang > 0.3 else WHITE)
    if abs(ang) > 0.05: c.arc(cx, cy, r, min(0, ang), max(0, ang), col, 14, a)
    hand(c, cx, cy, r * .78, ang, col, 8, a)
    c.circ(cx, cy, 16, fill=WHITE, a=a)
    c.t(cx, cy + 90, "base b", 32, DIM, a=a, w="demi")
    c.t(540, 1080, "θᵢ = b^(−2i/d)", 36, WHITE, a=pr(u, 0.5, 0.9), mono=True)
    c.t(540, 1140, "bigger b → slower clocks, longer wavelengths", 28, DIM, a=pr(u, 0.9, 1.3), w="demi")
    c.t(540, 590, "should it go down, or up?", 34, WHITE, a=pr(u, 1.0, 1.4), w="demi")

def s9(c, u):
    c.t(540, 470, "Which would you pick?", 62, WHITE, a=eo(pr(u, -0.2, 0.3)), w="heavy")
    cards = [("A", CYAN, "high-freq dims", "compressed"), ("B", AMBER, "forgetting from", "long-doc tuning"),
             ("C", PINK, "softmax temp", "too high"), ("D", VIOLET, "RoPE base should", "be decreased")]
    for k, (L, col, l1, l2) in enumerate(cards):
        q = eo(pr(u, 0.0 + 0.15 * k, 0.4 + 0.15 * k))
        x0 = 90 if k % 2 == 0 else 555; y0 = 560 if k < 2 else 830
        dy = 40 * (1 - q)
        c.rr(x0, y0 + dy, x0 + 435, y0 + 240 + dy, 26, fill=PANEL, outline=col, w=3, a=q)
        c.circ(x0 + 70, y0 + 70 + dy, 40, fill=col, a=q)
        c.t(x0 + 70, y0 + 72 + dy, L, 44, BG, a=q, w="heavy", bg=col)
        c.t(x0 + 40, y0 + 150 + dy, l1, 32, WHITE, a=q, anchor="lm", w="demi", bg=PANEL)
        c.t(x0 + 40, y0 + 192 + dy, l2, 32, WHITE, a=q, anchor="lm", w="demi", bg=PANEL)
    if u > 1.4: pill(c, 540, 1150, "+ which ablation isolates the cause?", 34, AMBER, a=pr(u, 1.4, 1.8))
    q = pr(u, 3.6, 4.0)
    c.t(540, 1260, "comment your pick + ablation ↓", 36 + 2 * math.sin(u * 5) * q, PINK, a=q, w="heavy")

SCENES = [s1, s2, s3, s4, s5, s6, s7, s8, s9]

# ---------------- frame ----------------
BASE = None
def make_base():
    im = Image.new("RGB", (W * SS, H * SS), BG); d = ImageDraw.Draw(im)
    for x in range(0, W, 60): d.line([(x * SS, 0), (x * SS, H * SS)], fill=GRID, width=SS)
    for y in range(0, H, 60): d.line([(0, y * SS), (W * SS, y * SS)], fill=GRID, width=SS)
    # soft vignette: darken top/bottom bands
    arr = np.asarray(im).astype(np.float32)
    yy = np.linspace(-1, 1, H * SS)[:, None]; xx = np.linspace(-1, 1, W * SS)[None, :]
    v = 1 - 0.35 * np.clip((xx ** 2 * 0.6 + yy ** 2) - 0.3, 0, 1)
    arr = arr * v[..., None]
    return Image.fromarray(arr.astype(np.uint8))

def captions(c, t):
    for st, en, words in CH:
        if st - 0.05 <= t < en:
            fade = pr(t, st - 0.05, st + 0.1) * (1 - pr(t, en - 0.12, en))
            size = 54; sp = c.tw(" ", size, "heavy")
            ws = [c.tw(w[0], size, "heavy") for w in words]
            lines = [[]]; lw = [0]
            for k, wd in enumerate(ws):
                if lines[-1] and lw[-1] + sp + wd > 920: lines.append([]); lw.append(0)
                lw[-1] += (sp if lines[-1] else 0) + wd; lines[-1].append(k)
            y = 1450 - (len(lines) - 1) * 36 + 10 * (1 - eo(pr(t, st - 0.05, st + 0.15)))
            for li, ln in enumerate(lines):
                x = 540 - lw[li] / 2
                for k in ln:
                    w, ws_, we_ = words[k]
                    active = ws_ - 0.03 <= t
                    col = AMBER if (ws_ - 0.03 <= t < we_ + 0.1) else (WHITE if active else DIM)
                    c.t(x, y + li * 72, w, size, col, a=fade, anchor="lm", w="heavy")
                    x += ws[k] + sp
            return

def header(c, t):
    a = pr(t, 3.0, 3.4)
    c.t(90, 262, "Q9", 32, AMBER, a=a, anchor="lm", w="heavy")
    c.t(150, 262, "CONTEXT EXTENSION", 30, WHITE, a=a, anchor="lm", w="demi")
    c.line([(90, 305), (990, 305)], PANEL, 6, a)
    c.line([(90, 305), (90 + 900 * cl(t / 78.6), 305)], CYAN, 6, a)

def frame(t):
    im = BASE.copy(); d = ImageDraw.Draw(im)
    header(C(d), t)
    for i in range(9):
        b = 0 if i == 0 else STARTS[i] - 0.35
        e = STARTS[i + 1] - 0.35 if i < 8 else TOTAL + 1
        if b <= t < e:
            A = pr(t, b, b + 0.3) if i else 1.0
            A *= 1 - pr(t, e - 0.28, e)
            SCENES[i](C(d, A), t - STARTS[i])
    captions(C(d), t)
    return im.resize((W, H), Image.LANCZOS)

def init():
    global BASE; BASE = make_base()

def render(i): return frame(i / FPS).tobytes()

if __name__ == "__main__":
    import multiprocessing as mp
    if sys.argv[1] == "preview":
        init()
        for ts in sys.argv[2:]:
            frame(float(ts)).save(os.path.join(HERE, f"prev_{ts}.png"))
        sys.exit()
    N = int(TOTAL * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(HERE, "video.mp4")], stdin=subprocess.PIPE)
    with mp.Pool(9, initializer=init) as pool:
        for k, b in enumerate(pool.imap(render, range(N), chunksize=6)):
            ff.stdin.write(b)
            if k % 300 == 0: print(k, N, flush=True)
    ff.stdin.close(); ff.wait(); print("done")
