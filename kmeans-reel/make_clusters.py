#!/usr/bin/env python3
"""Build clustering_data.json: one dataset, K=3, two Lloyd K-Means runs from
different random seeds (Forgy init: K distinct data points chosen by the seed).

Run 1 must recover the three natural groups (global optimum); run 2 must
converge to a genuine local minimum that splits one group and merges two.
Every iteration (assignment + mean update) is recorded and re-validated.
"""
import json
import math
import random

K = 3
N_PER = 40
CENTERS = [(2.4, 7.4), (5.6, 7.6), (6.6, 2.6)]
SPREAD = [(0.6, 0.55), (0.6, 0.55), (0.65, 0.6)]
DATA_SEED = 7


def make_data():
    rng = random.Random(DATA_SEED)
    pts, labels = [], []
    for g, ((cx, cy), (sx, sy)) in enumerate(zip(CENTERS, SPREAD)):
        n = 0
        while n < N_PER:  # truncated Gaussian: keeps each group visibly compact
            x, y = rng.gauss(0, 1), rng.gauss(0, 1)
            if x * x + y * y > 1.8 ** 2:
                continue
            pts.append((round(cx + sx * x, 4), round(cy + sy * y, 4)))
            labels.append(g)
            n += 1
    return pts, labels


def d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def assign(pts, cents):
    return [min(range(len(cents)), key=lambda k: d2(p, cents[k])) for p in pts]


def means(pts, lab, cents):
    out = []
    for k in range(len(cents)):
        mem = [p for p, l in zip(pts, lab) if l == k]
        if not mem:
            out.append(cents[k])
        else:
            out.append((sum(p[0] for p in mem) / len(mem), sum(p[1] for p in mem) / len(mem)))
    return out


def inertia(pts, lab, cents):
    return sum(d2(p, cents[l]) for p, l in zip(pts, lab))


def kmeans(pts, seed, max_iter=50):
    rng = random.Random(seed)
    idx = rng.sample(range(len(pts)), K)
    cents = [pts[i] for i in idx]
    hist = []
    lab = None
    for it in range(max_iter):
        new = assign(pts, cents)
        before = list(cents)
        moved = means(pts, new, cents)
        hist.append({"iter": it + 1, "centroids_before": [list(c) for c in before],
                     "assign": new, "centroids_after": [list(c) for c in moved],
                     "inertia": inertia(pts, new, moved),
                     "changed": None if lab is None else sum(a != b for a, b in zip(lab, new))})
        if new == lab:
            break
        lab, cents = new, moved
    return idx, hist


def purity(lab, true):
    """For each found cluster, which natural groups it contains."""
    comp = {}
    for l, t in zip(lab, true):
        comp.setdefault(l, set()).add(t)
    return comp


def classify(lab, true):
    comp = purity(lab, true)
    natural = all(len(v) == 1 for v in comp.values()) and len(comp) == 3 and \
        len({next(iter(v)) for v in comp.values()}) == 3
    groups_per = {}
    for l, t in zip(lab, true):
        groups_per.setdefault(t, set()).add(l)
    split = [g for g, s in groups_per.items() if len(s) == 2]
    merged = [l for l, v in comp.items() if len(v) == 2]
    return natural, split, merged, comp


def main():
    pts, true = make_data()
    good = bad = None
    for seed in range(1, 500):
        idx, hist = kmeans(pts, seed)
        final = hist[-1]["assign"]
        natural, split, merged, comp = classify(final, true)
        n_it = len(hist)
        if natural and good is None and 4 <= n_it <= 6:
            # want a visibly 'wrong-looking' start that still recovers
            good = (seed, idx, hist)
        if (not natural and len(split) == 1 and len(merged) == 1 and bad is None
                and 4 <= n_it <= 6 and split[0] == 2):
            bad = (seed, idx, hist)
        if good and bad:
            break
    assert good and bad, "no suitable seeds"

    runs = []
    for name, (seed, idx, hist) in (("run1", good), ("run2", bad)):
        # ---- validation: every step is a true Lloyd step; final state is a fixed point
        for h in hist:
            assert h["assign"] == assign(pts, [tuple(c) for c in h["centroids_before"]])
            assert all(abs(a - b) < 1e-12 for c1, c2 in zip(h["centroids_after"],
                       means(pts, h["assign"], h["centroids_before"])) for a, b in zip(c1, c2))
        fc = [tuple(c) for c in hist[-1]["centroids_after"]]
        fl = hist[-1]["assign"]
        assert assign(pts, fc) == fl, "not converged"
        assert all(math.dist(a, b) < 1e-12 for a, b in zip(means(pts, fl, fc), fc))
        # Re-run from scratch -> identical (reproducibility)
        assert kmeans(pts, seed)[1] == hist
        runs.append({"name": name, "seed": seed, "init_indices": idx,
                     "init_centroids": [list(pts[i]) for i in idx],
                     "iterations": hist, "final_centroids": [list(c) for c in fc],
                     "final_assign": fl, "final_inertia": inertia(pts, fl, fc),
                     "n_iterations": len(hist),
                     "composition": {str(k): sorted(v) for k, v in purity(fl, true).items()}})

    # Remap run2 cluster ids for nicer, consistent colours: the merged cluster gets the
    # colour id 0 (groups 0+1 lived there in run1 as ids..). Keep it simple: colour by id.
    out = {"k": K, "data_seed": DATA_SEED, "points": [list(p) for p in pts],
           "true_group": true, "bounds": [0, 10, 0, 10], "runs": runs,
           "note": "Lloyd's algorithm, Forgy init (random data points chosen by seed). "
                   "Each iteration: assign to nearest centroid, then move centroid to mean."}
    with open("clustering_data.json", "w") as f:
        json.dump(out, f, indent=1)
    for r in runs:
        print(r["name"], "seed", r["seed"], "iters", r["n_iterations"], "inertia",
              round(r["final_inertia"], 2), "composition", r["composition"],
              "changed per iter", [h["changed"] for h in r["iterations"]])


if __name__ == "__main__":
    main()
