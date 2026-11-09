"""
Where does the final system (TAAD + HPI + rule, seed 0) still lose F1 on validation?

Every wrong prediction is put into one group, by what is closest to it among the events it failed to match:
  wrong player   - same team, class and time (+-12 frames), other shirt number
  wrong team     - same class and time, other team
  wrong class    - same player and time, other class
  late / early   - same player and class, 13-50 frames away
  duplicate      - the event it would match was already taken by another prediction
  nothing there  - no event of any of these kinds nearby
For each group, "F1 if fixed" is the F1 if all of its predictions were turned into correct ones.
Diagnostic only: nothing here is tuned.

Usage:
  python error_breakdown.py
"""
import json
import os

import numpy as np

from hpi_exclusivity import apply_rule
from verify_hidden_v2 import match_counts

RUN = os.path.join("runs_hidden", "v7_33g_seed0")
gt_all = json.load(open(os.path.join("playbyplay_GT", "playbyplay_val.json")))["events"]
keys = list(gt_all)
taad_all = json.load(open("playbyplay_PRED_model40_audio.json"))["events"]
hpi_all = json.load(open(os.path.join(RUN, "hpi_val_events.json")))["events"]
c = json.load(open(os.path.join(RUN, "exclusivity.json")))["chosen"]

groups = ["wrong player", "wrong team", "wrong class", "late / early", "duplicate", "nothing there"]
fp_count = {g: 0 for g in groups}
fp_src = {g: [0, 0] for g in groups}  # TAAD, HPI
tp = fn = 0
fn_vis = fn_hid = 0
cls_fn = np.zeros(9, int)
cls_gt = np.zeros(9, int)
for k in keys:
    gt = gt_all[k]
    tk, hk = apply_rule([e for e in taad_all.get(k, []) if e[4] >= 0.70],
                        [e for e in hpi_all.get(k, []) if e[4] >= c["thr"]], c["rule"], c["w"])
    pred = [e[:4] + [2.0] for e in tk] + hk
    used, ok = match_counts(pred, gt)
    tp += int(ok.sum())
    for i, g in enumerate(gt):
        cls_gt[int(g[3])] += 1
        if not used[i]:
            fn += 1
            cls_fn[int(g[3])] += 1
            if int(g[4]) == 0:
                fn_hid += 1
            else:
                fn_vis += 1
    free = [i for i in range(len(gt)) if not used[i]]
    for j, p in enumerate(pred):
        if ok[j]:
            continue
        f, t, s, cl = (int(v) for v in p[:4])
        near = lambda i, d=12: abs(int(gt[i][0]) - f) <= d
        same = lambda i: (int(gt[i][1]), int(gt[i][2]), int(gt[i][3])) == (t, s, cl)
        if any(near(i) and int(gt[i][1]) == t and int(gt[i][3]) == cl for i in free):
            g = "wrong player"
        elif any(near(i) and int(gt[i][3]) == cl for i in free):
            g = "wrong team"
        elif any(near(i) and int(gt[i][1]) == t and int(gt[i][2]) == s for i in free):
            g = "wrong class"
        elif any(near(i, 50) and same(i) for i in free):
            g = "late / early"
        elif any(near(i) and same(i) for i in range(len(gt)) if used[i]):
            g = "duplicate"
        else:
            g = "nothing there"
        fp_count[g] += 1
        fp_src[g][0 if j < len(tk) else 1] += 1

fp = sum(fp_count.values())
f1 = lambda a, b, d: 2 * a / (2 * a + b + d)
print(f"TAAD + HPI + rule (seed 0) on validation: TP {tp}  FP {fp}  FN {fn}  micro F1 {f1(tp, fp, fn):.3f}\n")
print(f"{'wrong prediction':16s} {'count':>6s} {'share':>6s} {'TAAD':>5s} {'HPI':>5s} {'F1 if fixed':>12s}")
out = {"tp": tp, "fp": fp, "fn": fn, "groups": {}}
for g in groups:
    n = fp_count[g]
    fixed = f1(tp + n, fp - n, fn - n) if g not in ("duplicate", "nothing there") else f1(tp, fp - n, fn)
    print(f"{g:16s} {n:6d} {100 * n / fp:5.1f}% {fp_src[g][0]:5d} {fp_src[g][1]:5d} {fixed:12.3f}")
    out["groups"][g] = dict(count=n, taad=fp_src[g][0], hpi=fp_src[g][1], f1_if_fixed=fixed)
print("  (for 'duplicate' and 'nothing there', 'fixed' means the prediction is removed)\n")
print(f"Missed events: {fn} = {fn_vis} visible + {fn_hid} hidden")
print("Missed share by class:", {i: f"{cls_fn[i]}/{cls_gt[i]}" for i in range(1, 9) if cls_gt[i]})
out.update(fn_visible=fn_vis, fn_hidden=fn_hid, class_missed=cls_fn.tolist(), class_total=cls_gt.tolist())
json.dump(out, open(os.path.join("runs_hidden", "error_breakdown_val.json"), "w"), indent=2)
print("saved runs_hidden/error_breakdown_val.json")
