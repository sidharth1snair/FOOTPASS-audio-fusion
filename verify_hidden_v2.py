"""
Verification of HPI v2 results (run after train_hidden_v2.py with several seeds).

  1. Seeds        : overall F1 / hidden recall of each run, mean +- std
  2. Significance : paired bootstrap over val halves AND over events (TAAD alone vs TAAD + HPI)
  3. Breakdown    : hidden recall per class, and "no player visible on screen" vs "some players visible"
  4. Controls     : the trained model is re-run with corrupted inputs; a genuine model must DROP
                      - time_shift : tracking features shifted by +5 s relative to labels / visibility / TAAD context
                      - no_context : TAAD context removed (visible players' probabilities and global context zeroed)
                      - shuffle_slots : player identities shuffled inside each team (positions no longer match slots)

Usage:
  python verify_hidden_v2.py --runs runs_hidden/v2_full runs_hidden/v2_seed1 runs_hidden/v2_seed2 --val_logits TAAD_h5/preds_val.h5
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch

from train_hidden import merged_json, evaluate, events_from_probs, CLASS_NAMES, C
from train_hidden_v2 import load_half, HPIv2, predict_half, M


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--val_h5", default=os.path.join("hidden_data", "val.h5"))
    p.add_argument("--val_logits", default=os.path.join("kaggle_preds", "val", "TAAD_predictions"))
    p.add_argument("--gt_val", default=os.path.join("playbyplay_GT", "playbyplay_val.json"))
    p.add_argument("--taad_json", default="playbyplay_PRED_model40_audio.json")
    p.add_argument("--taad_thr", type=float, default=0.70)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", default=os.path.join("runs_hidden", "verification.json"))
    return p.parse_args()


def match_counts(pred, gt, delta=12):
    """Greedy matching as evaluation.py (same team, shirt, class; nearest frame within delta; by score).
    Returns per-GT matched flags and per-pred correct flags."""
    used = np.zeros(len(gt), bool)
    ok = np.zeros(len(pred), bool)
    groups = {}
    for i, g in enumerate(gt):
        groups.setdefault((int(g[1]), int(g[2]), int(g[3])), []).append(i)
    for j in sorted(range(len(pred)), key=lambda j: -pred[j][4]):
        f, t, s, c = (int(v) for v in pred[j][:4])
        best, bdt = -1, delta + 1
        for i in groups.get((t, s, c), []):
            dt = abs(int(gt[i][0]) - f)
            if not used[i] and dt <= delta and dt < bdt:
                best, bdt = i, dt
        if best >= 0:
            used[best], ok[j] = True, True
    return used, ok


def f1_from(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def per_half_counts(taad, hpi, gt, keys, taad_thr, hpi_thr):
    """(tp, fp, fn) per half for TAAD alone and TAAD + HPI, plus matched flags for breakdowns."""
    out = {}
    for k in keys:
        base = [e for e in taad.get(k, []) if e[4] >= taad_thr]
        add = [e for e in (hpi.get(k, []) if hpi else []) if e[4] >= hpi_thr]
        res = {}
        for name, pred in [("taad", base), ("merged", [e[:4] + [2.0] for e in base] + add)]:
            used, ok = match_counts(pred, gt[k])
            res[name] = (int(ok.sum()), int((~ok).sum()), int((~used).sum()), used)
        out[k] = res
    return out


def bootstrap(counts, keys, n, rng):
    """Paired bootstrap over halves: P(F1 gain <= 0) and 95% CI of the gain."""
    gains = []
    for _ in range(n):
        ks = rng.choice(keys, len(keys))
        tb = np.sum([counts[k]["taad"][:3] for k in ks], 0)
        tm = np.sum([counts[k]["merged"][:3] for k in ks], 0)
        gains.append(f1_from(*tm) - f1_from(*tb))
    gains = np.array(gains)
    return float(np.mean(gains <= 0)), np.percentile(gains, [2.5, 97.5]).tolist()


def bootstrap_events(counts, keys, gt, n, rng):
    """Paired bootstrap over GT events + predictions pooled (finer-grained, complements the half-level test)."""
    rows = []  # per GT event: (taad matched, merged matched)
    for k in keys:
        rows += list(zip(counts[k]["taad"][3], counts[k]["merged"][3]))
    rows = np.array(rows, dtype=np.int64)
    fp_b = sum(counts[k]["taad"][1] for k in keys)
    fp_m = sum(counts[k]["merged"][1] for k in keys)
    N = len(rows)
    gains = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        tp_b, tp_m = rows[idx, 0].sum(), rows[idx, 1].sum()
        # scale false positives with the resampled event mass (approximation)
        gains.append(f1_from(tp_m, fp_m, N - tp_m) - f1_from(tp_b, fp_b, N - tp_b))
    gains = np.array(gains)
    return float(np.mean(gains <= 0)), np.percentile(gains, [2.5, 97.5]).tolist()


@torch.no_grad()
def control_events(model, val, device, mode, rng, nms, W=100):
    evs = {}
    for k, h in val.items():
        hh = dict(h)
        x = h["x"].copy()
        if mode == "time_shift":
            sh = 125  # 5 s
            x[:, :, :12] = np.roll(h["x"][:, :, :12], sh, axis=0)  # tracking only; labels, visibility, TAAD context stay
            x[:, :, 11] = h["x"][:, :, 11]  # keep the true visibility flag aligned with evaluation
        elif mode == "no_context":
            if x.shape[-1] > 12:
                x[:, :, 12:] = 0.0
            hh["ctx"] = np.zeros_like(h["ctx"])
            hh["ctx"][:, :2] = h["ctx"][:, :2]
        elif mode == "shuffle_slots":
            for team in (0, 1):
                perm = rng.permutation(13) + 13 * team
                x[:, 13 * team:13 * team + 13] = x[:, perm]
        hh["x"] = x
        evs[k] = events_from_probs(predict_half(model, hh, device, W), hh, nms)
    return evs


def main():
    a = parse_args()
    rng = np.random.default_rng(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = json.load(open(a.gt_val))["events"]
    taad = json.load(open(a.taad_json))["events"]
    report = {"runs": {}}

    # ---------------- 1 + 2: seeds and significance
    print("=== 1. SEEDS (dev-chosen threshold) ===")
    f1s, rec = [], []
    for r in a.runs:
        hj = json.load(open(os.path.join(r, "hpi_val_events.json")))
        keys, thr = hj["keys"], hj["dev_thr"]
        counts = per_half_counts(taad, hj["events"], gt, keys, a.taad_thr, thr)
        tb = np.sum([counts[k]["taad"][:3] for k in keys], 0)
        tm = np.sum([counts[k]["merged"][:3] for k in keys], 0)
        p_half, ci_half = bootstrap(counts, keys, a.n_boot, rng)
        p_ev, ci_ev = bootstrap_events(counts, keys, gt, a.n_boot, rng)
        hid = sum(1 for k in keys for i, g in enumerate(gt[k]) if int(g[4]) == 0 and counts[k]["merged"][3][i])
        nh = sum(1 for k in keys for g in gt[k] if int(g[4]) == 0)
        f1s.append(f1_from(*tm)); rec.append(hid / nh)
        print(f"{r}: thr {thr}  F1 {f1_from(*tb):.3f} -> {f1_from(*tm):.3f} (gain {f1_from(*tm) - f1_from(*tb):+.3f}) | "
              f"hidden recall {hid}/{nh} ({100 * hid / nh:.1f}%)")
        print(f"     bootstrap over halves: P(gain<=0)={p_half:.3f} 95%CI {np.round(ci_half, 3)} | over events: P={p_ev:.3f} CI {np.round(ci_ev, 3)}")
        report["runs"][r] = dict(thr=thr, f1_taad=f1_from(*tb), f1_merged=f1_from(*tm), hidden_recall=hid / nh,
                                 p_half=p_half, ci_half=ci_half, p_events=p_ev, ci_events=ci_ev)
    print(f"mean over seeds: F1 {np.mean(f1s):.3f} +- {np.std(f1s):.3f} | hidden recall {100 * np.mean(rec):.1f}% +- {100 * np.std(rec):.1f}")

    # ---------------- 3: breakdown for the first run
    r0 = a.runs[0]
    hj = json.load(open(os.path.join(r0, "hpi_val_events.json")))
    keys, thr = hj["keys"], hj["dev_thr"]
    counts = per_half_counts(taad, hj["events"], gt, keys, a.taad_thr, thr)
    print(f"\n=== 3. BREAKDOWN ({r0}) — hidden events only ===")
    by_cls = np.zeros((C, 2), int)
    by_screen = {"no player visible": [0, 0], "some players visible": [0, 0]}
    for k in keys:
        for i, g in enumerate(gt[k]):
            if int(g[4]) != 0:
                continue
            m = int(counts[k]["merged"][3][i])
            by_cls[int(g[3])] += (m, 1)
            key = "no player visible" if int(g[5]) == 1 else "some players visible"
            by_screen[key][0] += m
            by_screen[key][1] += 1
    for c in range(1, C):
        if by_cls[c, 1]:
            print(f"  {CLASS_NAMES[c]:9s} {by_cls[c, 0]:4d}/{by_cls[c, 1]:4d} ({100 * by_cls[c, 0] / by_cls[c, 1]:5.1f}%)")
    for name, (m, n) in by_screen.items():
        print(f"  {name:22s} {m:4d}/{n:4d} ({100 * m / max(n, 1):5.1f}%)")
    report["breakdown"] = dict(per_class={CLASS_NAMES[c]: by_cls[c].tolist() for c in range(1, C)}, screen=by_screen)

    # ---------------- 4: controls with the first run's model
    ck = torch.load(os.path.join(r0, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    use_taad = not args.get("no_taad_context", False)
    with h5py.File(a.val_h5, "r") as f:
        val = {k: load_half(f[k], a.val_logits, ck["mean"], ck["std"], use_taad, args.get("focus", False), args.get("chain", False)) for k in keys}
    fp, fg = next(iter(val.values()))["x"].shape[-1], next(iter(val.values()))["ctx"].shape[-1]
    model = HPIv2(fp, fg, args.get("d", 96)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"\n=== 4. CONTROLS ({r0}, threshold {thr}) — a real model must DROP ===")
    report["controls"] = {}
    for mode in ["none", "time_shift", "no_context", "shuffle_slots"]:
        evs = hj["events"] if mode == "none" else control_events(model, val, device, mode, rng, args.get("nms_window", 25), args.get("window", 100))
        c2 = per_half_counts(taad, evs, gt, keys, a.taad_thr, thr)
        tm = np.sum([c2[k]["merged"][:3] for k in keys], 0)
        hid = sum(1 for k in keys for i, g in enumerate(gt[k]) if int(g[4]) == 0 and c2[k]["merged"][3][i])
        nh = sum(1 for k in keys for g in gt[k] if int(g[4]) == 0)
        print(f"  {mode:14s} F1 {f1_from(*tm):.3f}  hidden recall {100 * hid / nh:5.1f}%")
        report["controls"][mode] = dict(f1=f1_from(*tm), hidden_recall=hid / nh)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2, default=float)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
