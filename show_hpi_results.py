"""
Prints the HPI results on the 6 validation halves from the saved predictions (no training, about 1 minute).

Usage:
  python show_hpi_results.py
"""
import json
import os

import numpy as np

from hpi_exclusivity import apply_rule, score
from verify_hidden_v2 import match_counts

RUNS = ["v6_w250", "v6_w250_seed1", "v6_w250_seed2"]
NAMES = ["", "drive", "pass", "cross", "throw-in", "shot", "header", "tackle", "block"]


def macro_f1(taad, hpi, gt, keys, thr, rule):
    tp, fp, fn = np.zeros(9), np.zeros(9), np.zeros(9)
    for k in keys:
        tk, hk = apply_rule([e for e in taad.get(k, []) if e[4] >= 0.70], [e for e in hpi.get(k, []) if e[4] >= thr], rule, 12)
        pred = [e[:4] + [2.0] for e in tk] + hk
        used, ok = match_counts(pred, gt[k])
        for e, o in zip(pred, ok):
            (tp if o else fp)[int(e[3])] += 1
        for g, u in zip(gt[k], used):
            if not u:
                fn[int(g[3])] += 1
    return float((2 * tp[1:] / np.maximum(2 * tp[1:] + fp[1:] + fn[1:], 1)).mean())


def main():
    gt = json.load(open(os.path.join("playbyplay_GT", "playbyplay_val.json")))["events"]
    taad = json.load(open("playbyplay_PRED_model40_audio.json"))["events"]
    keys = list(gt)
    r = score(taad, {}, gt, keys, 0.70, 0.0, "none", 0)[1]
    n = r["hidden_n"]
    print(f"Validation: {len(keys)} halves, {sum(len(v) for v in gt.values())} events, {n} by hidden players (no bounding box)\n")
    print(f"{'system':32s} {'micro F1':>9s} {'macro F1':>9s} {'precision':>10s} {'recall':>7s} {'hidden found':>18s}")
    line = lambda name, x, m: print(f"{name:32s} {x['f1']:9.3f} {m:9.3f} {x['precision']:10.3f} {x['recall']:7.3f} "
                                    f"{x['hidden_tp']:6d}/{x['hidden_n']} ({100 * x['hidden_recall']:4.1f}%)")
    line("TAAD alone", r, macro_f1(taad, {}, gt, keys, 0.0, "none"))
    plain, rule = [], []
    for run in RUNS:
        p = os.path.join("runs_hidden", run)
        hj = json.load(open(os.path.join(p, "hpi_val_events.json")))
        c = json.load(open(os.path.join(p, "exclusivity.json")))["chosen"]
        x = score(taad, hj["events"], gt, keys, 0.70, hj["dev_thr"], "none", 0)[1]
        y = score(taad, hj["events"], gt, keys, 0.70, c["thr"], c["rule"], c["w"])[1]
        mx, my = macro_f1(taad, hj["events"], gt, keys, hj["dev_thr"], "none"), macro_f1(taad, hj["events"], gt, keys, c["thr"], c["rule"])
        line(f"TAAD + HPI ({run})", x, mx)
        line(f"  + one-action rule", y, my)
        plain.append((x["f1"], mx, x["hidden_recall"]))
        rule.append((y["f1"], my, y["hidden_recall"]))
    for name, v in [("TAAD + HPI", plain), ("TAAD + HPI + rule", rule)]:
        v = np.array(v)
        print(f"\n{name}, mean of 3 seeds: micro F1 {v[:, 0].mean():.3f} +- {v[:, 0].std():.3f} | macro F1 {v[:, 1].mean():.3f} +- {v[:, 1].std():.3f} | "
              f"hidden found {100 * v[:, 2].mean():.1f}% +- {100 * v[:, 2].std():.1f}", end="")
    print("\n\nExamples of hidden-player events found by HPI (run v6_w250):")
    hj = json.load(open(os.path.join("runs_hidden", RUNS[0], "hpi_val_events.json")))
    shown = 0
    for k in keys:
        pred = [e for e in hj["events"][k] if e[4] >= hj["dev_thr"]]
        _, ok = match_counts(pred, gt[k])
        for e, o in sorted(zip(pred, ok), key=lambda x: -x[0][4]):
            hit = o and any(int(g[4]) == 0 and abs(g[0] - e[0]) <= 12 and (g[1], g[2], g[3]) == (e[1], e[2], e[3]) for g in gt[k])
            if hit:
                print(f"  {k}  {int(e[0] / 25 // 60):3d}:{int(e[0] / 25 % 60):02d}  shirt {int(e[2]):2d}  {NAMES[int(e[3])]:8s}  confidence {e[4]:.2f}")
                shown += 1
                break
    print(f"\n({shown} examples, the most confident correct one per half; F1 scored with the same matching as evaluation.py)")


if __name__ == "__main__":
    main()
