"""
Does audio make the highlight reel better? (no video rendering, uses generate_highlights.py's own selection code)

Reference (independent of audio): every ground-truth SHOT of the 3 val games; TAAD did not train on these games.
A shot is covered when it falls inside a clip of the reel. Clip precision = clips containing a real shot.

Events : TAAD@0.70, and TAAD + HPI v6 + one-action rule.
Audio  : none | energy (default) | energy with tau = 0 | energy from the WRONG game | energy shifted 60 s |
         PANNs shot detector (logistic regression trained on the other 2 val games).
Weights and reel length are the script's defaults, fixed before looking at these results.

Usage:
  python highlight_comparison.py
"""
import argparse
import json
import os
from types import SimpleNamespace

import h5py
import numpy as np

from audio_probe_panns import logreg
from generate_highlights import group_moments, load_excitement_signal, load_tau_seconds, score_events, select_moments
from hpi_exclusivity import apply_rule

GAMES = [18, 24, 47]
SHOT = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_val", default=os.path.join("playbyplay_GT", "playbyplay_val.json"))
    p.add_argument("--taad_json", default="playbyplay_PRED_model40_audio.json")
    p.add_argument("--run", default=os.path.join("runs_hidden", "v6_w250"))
    p.add_argument("--hpi_thr", type=float, default=0.2)
    p.add_argument("--audio_h5", default=os.path.join("data", "audio_features.h5"))
    p.add_argument("--panns", default=os.path.join("data", "panns_raw.h5"))
    p.add_argument("--checkpoint", default=os.path.join("runs", "TAAD_audiofusion_kaggle", "checkpoints", "curr_model_40.pt"))
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", default=os.path.join("runs_hidden", "highlight_comparison.json"))
    return p.parse_args()


def rows_for(game, taad, hpi, hpi_thr):
    rows = []
    for k in (f"game_{game}_H1", f"game_{game}_H2"):
        tk = [e for e in taad.get(k, []) if e[4] >= 0.70]
        hk = [e for e in (hpi or {}).get(k, []) if e[4] >= hpi_thr]
        if hpi is not None:
            tk, hk = apply_rule(tk, hk, "global", 12)
        rows += [(int(e[0]), int(e[1]), int(e[2]), int(e[3]), float(e[4])) for e in tk + hk]
    return sorted(rows)


def panns_shot_signal(a, shots):
    """Leave-one-game-out shot detector on PANNs embeddings -> per-frame excitement, on the same scale as energy."""
    pan = h5py.File(a.panns, "r")
    X, Y = {}, {}
    for g in GAMES:
        e = pan[f"game_{g}"]["emb"][:].astype(np.float32)
        fr = np.arange(len(e)) * 10
        s = np.sort(shots[g])
        i = np.clip(np.searchsorted(s, fr), 1, len(s) - 1)
        X[g], Y[g] = e, (np.minimum(np.abs(fr - s[i - 1]), np.abs(fr - s[i])) <= 50).astype(np.float32)
    out = {}
    for g in GAMES:
        tr = [x for x in GAMES if x != g]
        P = np.concatenate([X[x] for x in tr])
        mu = P.mean(0)
        vt = np.linalg.svd(P[::4] - mu, full_matrices=False)[2][:64].T
        z = logreg((P - mu) @ vt, np.concatenate([Y[x] for x in tr]), (X[g] - mu) @ vt)
        z = (z - np.median(z)) / (1.4826 * np.median(np.abs(z - np.median(z))) + 1e-9)
        out[g] = np.repeat(z, 10)
    return out


def reel(rows, tau, exc, weight, target=180.0):
    args = SimpleNamespace(conf_thresh=0.0, fps=25, max_reaction=4.0, audio_weight=weight, merge_gap=10.0,
                           min_class_weight=0.5, pre_roll=5.0, post_roll=3.0, target_seconds=target)
    return select_moments(group_moments(score_events(rows, args, tau, exc), args), args)


def main():
    a = parse_args()
    rng = np.random.default_rng(0)
    gt = json.load(open(a.gt_val))["events"]
    taad = json.load(open(a.taad_json))["events"]
    hpi = json.load(open(os.path.join(a.run, "hpi_val_events.json")))["events"]
    shots = {g: np.array([e[0] for k in (f"game_{g}_H1", f"game_{g}_H2") for e in gt[k] if int(e[3]) == SHOT]) for g in GAMES}
    print("GT shots per game:", {g: len(s) for g, s in shots.items()})
    tau = load_tau_seconds(a.checkpoint, 25)
    zero = np.zeros_like(tau)
    energy = {g: load_excitement_signal(a.audio_h5, g) for g in GAMES}
    panns = panns_shot_signal(a, shots)

    audio = {"no audio": lambda g: (None, tau, 0.0),
             "energy + tau": lambda g: (energy[g], tau, 2.0),
             "energy, tau = 0": lambda g: (energy[g], zero, 2.0),
             "energy, WRONG game": lambda g: (energy[GAMES[(GAMES.index(g) + 1) % 3]], tau, 2.0),
             "energy, shifted 60 s": lambda g: (np.roll(energy[g], 1500), tau, 2.0),
             "PANNs shot detector": lambda g: (panns[g], tau, 2.0)}
    res, flags = {}, {}
    print(f"\n{'events':16s} {'audio':22s} {'shots covered':>15s} {'clips':>6s} {'clips with shot':>16s} {'seconds':>8s}")
    for ename, src in [("TAAD", None), ("TAAD+HPI+rule", hpi)]:
        for aname, f in audio.items():
            cov, nclip, good, secs = [], 0, 0, 0.0
            for g in GAMES:
                exc, t, w = f(g)
                clips = reel(rows_for(g, taad, src, a.hpi_thr), t, exc, w)
                st = np.array([c["start"] for c in clips]); en = np.array([c["end"] for c in clips])
                sec = shots[g] / 25
                cov += [bool(((st <= s) & (s <= en)).any()) for s in sec]
                nclip += len(clips)
                good += sum(bool(((sec >= c["start"]) & (sec <= c["end"])).any()) for c in clips)
                secs += float((en - st).sum())
            cov = np.array(cov)
            flags[ename, aname] = cov
            res[f"{ename} | {aname}"] = dict(covered=int(cov.sum()), shots=len(cov), clips=nclip, clips_with_shot=good, seconds=secs)
            print(f"{ename:16s} {aname:22s} {cov.sum():5d}/{len(cov)} ({100 * cov.mean():4.1f}%) {nclip:6d} "
                  f"{good:6d} ({100 * good / max(nclip, 1):4.1f}%) {secs:8.0f}")

    print("\nPaired bootstrap over shots (difference in shots covered, 95% CI):")
    pairs = [(("TAAD", "energy + tau"), ("TAAD", "no audio")), (("TAAD", "energy + tau"), ("TAAD", "energy, WRONG game")),
             (("TAAD+HPI+rule", "energy + tau"), ("TAAD+HPI+rule", "no audio")),
             (("TAAD+HPI+rule", "energy + tau"), ("TAAD+HPI+rule", "energy, WRONG game")),
             (("TAAD+HPI+rule", "energy + tau"), ("TAAD+HPI+rule", "energy, tau = 0")),
             (("TAAD+HPI+rule", "PANNs shot detector"), ("TAAD+HPI+rule", "no audio")),
             (("TAAD+HPI+rule", "no audio"), ("TAAD", "no audio"))]
    res["bootstrap"] = {}
    for x, y in pairs:
        d = flags[x].astype(float) - flags[y].astype(float)
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.n_boot)]
        ci = np.percentile(b, [2.5, 97.5])
        print(f"  {' / '.join(x):42s} vs {' / '.join(y):42s} {100 * d.mean():+5.1f} pts  CI [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]")
        res["bootstrap"][f"{x} vs {y}"] = dict(diff=float(d.mean()), ci=ci.tolist())
    json.dump(res, open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
