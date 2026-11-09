"""
Quick test: can the commentary (Whisper transcript) tell us WHO acted, for hidden-player events?

  1. Action words : does a spoken "cross", "header"... fall just after a real event of that class?
  2. Name -> player: learned from VISIBLE events only, no lineups: a name is linked to the (team, shirt)
                     that acts most often just before the name is spoken (lift over that player's base rate).
  3. Hidden events : is the actor's linked name spoken just after a hidden event?
                     Also for the hidden events HPI v6 misses. Control: words shifted by +60 s.

Commentary lags the action, so a word at time w is linked to events in [w - BEFORE, w + AFTER].

Usage:
  python whisper_probe.py --whisper data/whisper_val.json --game game_47
"""
import argparse
import collections
import json
import os
import re

import numpy as np

from verify_hidden_v2 import match_counts

BEFORE, AFTER = 5.0, 1.0
CLASS_WORDS = {2: ["pass", "passes", "passed"], 3: ["cross", "crosses", "crossed"], 4: ["throw", "throw-in"],
               5: ["shot", "shoots", "shoot"], 6: ["header", "heads", "headed"], 7: ["tackle", "tackled", "tackles"],
               8: ["block", "blocked", "blocks"]}
NAMES = ["", "drive", "pass", "cross", "throw-in", "shot", "header", "tackle", "block"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--whisper", default=os.path.join("data", "whisper_val.json"))
    p.add_argument("--gt_val", default=os.path.join("playbyplay_GT", "playbyplay_val.json"))
    p.add_argument("--run", default=os.path.join("runs_hidden", "v6_w250"))
    p.add_argument("--taad_json", default="playbyplay_PRED_model40_audio.json")
    p.add_argument("--game", default="game_47")
    p.add_argument("--hpi_thr", type=float, default=0.2)
    p.add_argument("--min_count", type=int, default=3)
    p.add_argument("--min_lift", type=float, default=2.0)
    return p.parse_args()


def clean(w):
    return w.strip(".,!?;:'\"()-")


def spoken_after(t, times):
    """True if a word time lies in [t - AFTER, t + BEFORE] (the word follows the event)."""
    return bool(times.size) and bool(((times >= t - AFTER) & (times <= t + BEFORE)).any())


def main():
    a = parse_args()
    d = json.load(open(a.whisper, encoding="utf-8"))[a.game]
    gt_all = json.load(open(a.gt_val))["events"]
    keys = [k for k in gt_all if k.rsplit("_", 1)[0] == a.game]
    # "team" in the data is the side of the pitch, which swaps at half-time: use one team id for the whole game
    ev = [dict(t=g[0] / 25, p=(int(g[1]) ^ int(k.endswith("H2")), int(g[2])), c=int(g[3]), vis=int(g[4]) == 1, k=k, i=i)
          for k in keys for i, g in enumerate(gt_all[k])]
    words = [(w[0], clean(w[2])) for w in d["words"] if clean(w[2])]
    print(f"{a.game}: {len(words)} words, {len(ev)} events, {sum(not e['vis'] for e in ev)} hidden")

    print("\n=== 1. ACTION WORDS (event of that class in the 5 s before the word) ===")
    rng = np.random.default_rng(0)
    et = {c: np.array([e["t"] for e in ev if e["c"] == c]) for c in CLASS_WORDS}
    for c, ws in CLASS_WORDS.items():
        wt = np.array([t for t, w in words if w.lower() in ws])
        if wt.size == 0 or et[c].size == 0:
            print(f"  {NAMES[c]:9s} spoken {wt.size:3d}x | events {et[c].size}")
            continue
        hit = lambda x: np.mean([((et[c] >= t - BEFORE) & (et[c] <= t + AFTER)).any() for t in x])
        chance = np.mean([hit(wt + s) for s in rng.uniform(60, 600, 20)])
        rec = np.mean([spoken_after(t, wt) for t in et[c]])
        print(f"  {NAMES[c]:9s} spoken {wt.size:3d}x | word follows a real event {100 * hit(wt):5.1f}% (chance {100 * chance:5.1f}%) | "
              f"events with the word {100 * rec:5.1f}% of {et[c].size}")

    lower = {w for _, w in words if w.islower()}
    cnt = collections.Counter(w for _, w in words if re.match(r"^[A-Z][a-zà-ÿ]{2,}$", w) and w.lower() not in lower)
    cand = [w for w, n in cnt.items() if n >= a.min_count]
    vis = [e for e in ev if e["vis"]]
    base = collections.Counter(e["p"] for e in vis)
    n_vis = len(vis)
    vt = np.array([e["t"] for e in vis])
    link = {}
    for n in cand:
        times = [t for t, w in words if w == n]
        co = collections.Counter()
        for t in times:
            for j in np.nonzero((vt >= t - BEFORE) & (vt <= t + AFTER))[0]:
                co[vis[j]["p"]] += 1
        if not co:
            continue
        tot = sum(co.values())
        lift = {p: (c / tot) / (base[p] / n_vis) for p, c in co.items() if c >= a.min_count}
        if lift:
            p = max(lift, key=lift.get)
            if lift[p] >= a.min_lift:
                link[n] = (p, round(lift[p], 1), co[p], len(times))
    print(f"\n=== 2. NAME -> PLAYER, learned from visible events only ({len(cand)} candidate names, {len(link)} linked) ===")
    for n, (p, lift, c, m) in sorted(link.items(), key=lambda x: -x[1][3]):
        print(f"  {n:14s} -> team {p[0]} shirt {p[1]:2d} | lift {lift:4.1f} | together {c:3d} | spoken {m:3d}x")

    hj = json.load(open(os.path.join(a.run, "hpi_val_events.json")))
    taad = json.load(open(a.taad_json))["events"]
    found = {}
    for k in keys:
        pred = [e[:4] + [2.0] for e in taad.get(k, []) if e[4] >= 0.70] + [e for e in hj["events"][k] if e[4] >= a.hpi_thr]
        found[k] = match_counts(pred, gt_all[k])[0]
    by_player = collections.defaultdict(list)
    for n, (p, *_ ) in link.items():
        by_player[p] += [t for t, w in words if w == n]
    by_player = {p: np.array(v) for p, v in by_player.items()}

    print("\n=== 3. HIDDEN EVENTS: is the actor's linked name spoken in the 5 s after? ===")
    hid = [e for e in ev if not e["vis"]]
    groups = {"all hidden": hid, "missed by HPI v6": [e for e in hid if not found[e["k"]][e["i"]]],
              "actor has a linked name": [e for e in hid if e["p"] in by_player]}
    for name, es in groups.items():
        real = np.mean([spoken_after(e["t"], by_player.get(e["p"], np.array([]))) for e in es])
        ctrl = np.mean([spoken_after(e["t"], by_player.get(e["p"], np.array([])) + 60) for e in es])
        print(f"  {name:24s} n={len(es):3d} | name spoken {100 * real:5.1f}% | words shifted 60 s {100 * ctrl:5.1f}%")
    allv = np.concatenate(list(by_player.values())) if by_player else np.array([])
    print(f"  linked-name mentions: {allv.size}; hidden events in the game: {len(hid)}")


if __name__ == "__main__":
    main()
