"""
tau = 0 ablation on validation (games 18, 24, 47): same TAAD model and audio, but every audio delay tau fixed at 0.
Compared with the learned-tau model using the same one-pass predictions and the same post-processing.

Usage:
  python score_tau0.py
"""
import json
import os

from hpi_exclusivity import score
from hpi_final_test import taad_events
import numpy as np
from hpi_exclusivity import match_counts

GT = json.load(open(os.path.join("playbyplay_GT", "playbyplay_val.json")))["events"]
KEYS = list(GT)
H5 = os.path.join("data", "val_tactical_data.h5")
RUNS = {
    "learned tau": (os.path.join("kaggle_preds", "val", "TAAD_predictions"), os.path.join("runs_hidden", "taad_val_check_events.json")),
    "tau = 0": (os.path.join("kaggle_preds", "val_tau0", "TAAD_predictions"), os.path.join("runs_hidden", "taad_val_tau0_events.json")),
}



def macro(ev, thr):
    tp, fp, fn = np.zeros(9), np.zeros(9), np.zeros(9)
    for k in KEYS:
        pred = [e[:4] + [2.0] for e in ev.get(k, []) if e[4] >= thr]
        used, ok = match_counts(pred, GT[k])
        for e, o in zip(pred, ok):
            (tp if o else fp)[int(e[3])] += 1
        for g, u in zip(GT[k], used):
            if not u:
                fn[int(g[3])] += 1
    return float((2 * tp[1:] / np.maximum(2 * tp[1:] + fp[1:] + fn[1:], 1)).mean())


out = {}
print(f"{'model':12s} {'thr':>5s} {'micro':>6s} {'macro':>6s} {'prec':>6s} {'recall':>6s}")
for name, (logits, path) in RUNS.items():
    ev = taad_events(logits, H5, path)
    out[name] = {}
    for thr in [0.15, 0.3, 0.5, 0.7, 0.8]:
        r = score(ev, {}, GT, KEYS, thr, 0.0, "none", 0)[1]
        m = macro(ev, thr)
        print(f"{name:12s} {thr:5.2f} {r['f1']:6.3f} {m:6.3f} {r['precision']:6.3f} {r['recall']:6.3f}")
        out[name][thr] = dict(f1=r["f1"], macro=m, precision=r["precision"], recall=r["recall"])
json.dump(out, open(os.path.join("runs_hidden", "tau0_val.json"), "w"), indent=2, default=float)
print("saved runs_hidden/tau0_val.json")
