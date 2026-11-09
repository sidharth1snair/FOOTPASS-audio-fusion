"""
Improvement 1 for HPI: "one ball action at a time" post-processing (no retraining).

Only one player acts on the ball at any moment, but TAAD (visible players) and HPI (hidden players)
predict every player independently, so they produce simultaneous events on different players.
This script removes such duplicates and checks whether it helps.

  1. GT check     : how often do real events of two different players fall within +-w frames?
  2. Rules        : none        - TAAD@0.70 + HPI as before
                    taad_prio   - drop an HPI event within +-w of a kept TAAD event
                    hpi_nms     - among HPI events, keep only the most confident within +-w (any player)
                    both        - taad_prio + hpi_nms
                    global      - one greedy NMS over all TAAD + HPI events by score (also removes TAAD duplicates)
  3. Selection    : rule, window w and HPI threshold chosen on the 6 held-out train halves (dev),
                    then reported once on val, with a paired bootstrap against the old setting.

Caveat: TAAD was trained on all train halves, so its dev predictions are optimistic.

Usage:
  python hpi_exclusivity.py --run runs_hidden/v2_full
"""
import argparse
import json
import os
import shutil

import h5py
import numpy as np
import torch

from train_hidden import merged_json, evaluate, events_from_probs
from train_hidden_v2 import load_half, HPIv2, predict_half
from verify_hidden_v2 import match_counts, f1_from, bootstrap, bootstrap_events
from utils.metric_utils import post_processing_TAAD_preds_and_export

RULES = ["none", "taad_prio", "hpi_nms", "both", "global"]
WINDOWS = [6, 12, 25]
THRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=os.path.join("runs_hidden", "v2_full"))
    p.add_argument("--train_h5", default=os.path.join("hidden_data", "train.h5"))
    p.add_argument("--train_logits", default=os.path.join("kaggle_preds", "TAAD_predictions"))
    p.add_argument("--train_tactical", default=os.path.join("data", "train_tactical_data.h5"))
    p.add_argument("--gt_train", default=os.path.join("playbyplay_GT", "playbyplay_train.json"))
    p.add_argument("--gt_val", default=os.path.join("playbyplay_GT", "playbyplay_val.json"))
    p.add_argument("--taad_json", default="playbyplay_PRED_model40_audio.json")
    p.add_argument("--taad_thr", type=float, default=0.70)
    p.add_argument("--dev_halves", type=int, default=6)
    p.add_argument("--dev_games", default=None, help="Comma-separated game numbers held out as dev")
    p.add_argument("--n_boot", type=int, default=2000)
    return p.parse_args()


# ----------------------------------------------------------------------------- dev predictions (cached)

def dev_hpi_events(a, dev_keys):
    path = os.path.join(a.run, "hpi_dev_events.json")
    if os.path.exists(path):
        return json.load(open(path))["events"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(a.run, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    use_taad = not args.get("no_taad_context", False)
    with h5py.File(a.train_h5, "r") as f:
        dev = {k: load_half(f[k], a.train_logits, ck["mean"], ck["std"], use_taad, args.get("focus", False), args.get("chain", False)) for k in dev_keys}
    fp, fg = next(iter(dev.values()))["x"].shape[-1], next(iter(dev.values()))["ctx"].shape[-1]
    model = HPIv2(fp, fg, args.get("d", 96)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    ev = {k: events_from_probs(predict_half(model, h, device, args.get("window", 100)), h, args.get("nms_window", 25))
          for k, h in dev.items()}
    json.dump({"keys": dev_keys, "events": ev}, open(path, "w"))
    return ev


def dev_taad_events(a, dev_keys):
    path = os.path.join("runs_hidden", "taad_dev_events.json")
    if os.path.exists(path):
        return json.load(open(path))["events"]
    tmp = os.path.join("runs_hidden", "_taad_dev_tmp")
    os.makedirs(os.path.join(tmp, "TAAD_predictions"), exist_ok=True)
    for k in dev_keys:
        shutil.copy(os.path.join(a.train_logits, f"avg_logits_{k}.npy"), os.path.join(tmp, "TAAD_predictions"))
    post_processing_TAAD_preds_and_export(nms_window=25, root_path=tmp, save_path=path, h5file_path=a.train_tactical)
    shutil.rmtree(tmp)
    return json.load(open(path))["events"]


# ----------------------------------------------------------------------------- rules

def nms_across_players(events, w):
    """Greedy by score: keep an event only if no kept event lies within +-w frames (any player)."""
    kept = []
    for e in sorted(events, key=lambda e: -e[4]):
        if all(abs(e[0] - k[0]) > w for k in kept):
            kept.append(e)
    return kept


def apply_rule(taad, hpi, rule, w):
    """taad / hpi: thresholded event lists of one half. Returns (taad_kept, hpi_kept)."""
    if rule == "none":
        return taad, hpi
    if rule in ("hpi_nms", "both"):
        hpi = nms_across_players(hpi, w)
    if rule in ("taad_prio", "both"):
        tf = np.array(sorted(e[0] for e in taad))
        if tf.size:
            hpi = [e for e in hpi if np.abs(tf - e[0]).min() > w]
    if rule == "global":
        tagged = [e + ["T"] for e in taad] + [e + ["H"] for e in hpi]
        kept = nms_across_players(tagged, w)
        taad = [e[:5] for e in kept if e[5] == "T"]
        hpi = [e[:5] for e in kept if e[5] == "H"]
    return taad, hpi


def score(taad_all, hpi_all, gt, keys, taad_thr, hpi_thr, rule, w):
    """Per-half counts in the verify_hidden_v2 format + summary numbers."""
    counts, n_hpi, ok_hpi, hid_tp, hid_n = {}, 0, 0, 0, 0
    for k in keys:
        base = [e for e in taad_all.get(k, []) if e[4] >= taad_thr]
        add = [e for e in hpi_all.get(k, []) if e[4] >= hpi_thr]
        tk, hk = apply_rule(base, add, rule, w)
        pred = [e[:4] + [2.0] for e in tk] + hk  # TAAD first in matching, as in verify_hidden_v2
        used, ok = match_counts(pred, gt[k])
        counts[k] = {"merged": (int(ok.sum()), int((~ok).sum()), int((~used).sum()), used)}
        n_hpi += len(hk)
        ok_hpi += int(ok[len(tk):].sum())
        for i, g in enumerate(gt[k]):
            if int(g[4]) == 0:
                hid_n += 1
                hid_tp += int(used[i])
    tp, fp, fn = np.sum([counts[k]["merged"][:3] for k in keys], 0)
    return counts, dict(f1=f1_from(tp, fp, fn), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                        hidden_recall=hid_tp / max(hid_n, 1), hidden_tp=hid_tp, hidden_n=hid_n,
                        n_hpi=n_hpi, hpi_precision=ok_hpi / max(n_hpi, 1))


def gt_cooccurrence(gt, keys):
    print("=== 1. GROUND TRUTH: events with another real event within +-w frames ===")
    for w in WINDOWS:
        n = other = same = 0
        for k in keys:
            ev = sorted(gt[k], key=lambda g: g[0])
            fr = np.array([g[0] for g in ev])
            for i, g in enumerate(ev):
                near = [j for j in np.nonzero(np.abs(fr - g[0]) <= w)[0] if j != i]
                n += 1
                if any((ev[j][1], ev[j][2]) != (g[1], g[2]) for j in near):
                    other += 1
                if any((ev[j][1], ev[j][2]) == (g[1], g[2]) for j in near):
                    same += 1
        print(f"  w=+-{w:2d} frames ({w / 25:.2f} s): different player {100 * other / n:5.1f}% | same player {100 * same / n:5.1f}%")


def main():
    a = parse_args()
    rng = np.random.default_rng(0)
    with h5py.File(a.train_h5, "r") as f:
        dev_keys = sorted(f.keys())[-a.dev_halves:]
        if a.dev_games:
            dev_keys = [k for k in sorted(f.keys()) if k.split("_")[1] in a.dev_games.split(",")]
    gt_tr = json.load(open(a.gt_train))["events"]
    gt_va = json.load(open(a.gt_val))["events"]
    gt_cooccurrence(gt_tr, list(gt_tr))

    hpi_dev = dev_hpi_events(a, dev_keys)
    taad_dev = dev_taad_events(a, dev_keys)
    hj = json.load(open(os.path.join(a.run, "hpi_val_events.json")))
    val_keys, hpi_val, old_thr = hj["keys"], hj["events"], hj["dev_thr"]
    taad_val = json.load(open(a.taad_json))["events"]

    print(f"\n=== 2. DEV ({len(dev_keys)} held-out train halves): best HPI threshold per rule / window ===")
    best = None
    table = []
    for rule in RULES:
        for w in ([0] if rule == "none" else WINDOWS):
            res = [(thr, score(taad_dev, hpi_dev, gt_tr, dev_keys, a.taad_thr, thr, rule, w)[1]) for thr in THRS]
            thr, r = max(res, key=lambda x: x[1]["f1"])
            table.append((rule, w, thr))
            print(f"  {rule:9s} w={w:2d}: thr {thr:.1f}  F1 {r['f1']:.3f}  hidden recall {100 * r['hidden_recall']:4.1f}%  "
                  f"HPI preds {r['n_hpi']:5d} (prec {r['hpi_precision']:.2f})")
            if best is None or r["f1"] > best[3]["f1"]:
                best = (rule, w, thr, r)
    rule, w, thr = best[:3]
    print(f"  -> chosen on dev: rule={rule}, w=+-{w}, HPI thr={thr}")

    print("\n=== 3. VALIDATION (settings fixed on dev) ===")
    print(f"  {'setting':40s} {'F1':>6} {'P':>6} {'R':>6} {'hidden recall':>14} {'HPI preds':>10} {'HPI prec':>9}")
    out = {"chosen": dict(rule=rule, w=w, thr=thr), "val": {}}
    rows = [("TAAD alone", {}, 0.0, "none", 0), (f"TAAD + HPI (old, thr {old_thr})", hpi_val, old_thr, "none", 0)]
    rows += [(f"{r_} w={w_} thr {t_}", hpi_val, t_, r_, w_) for r_, w_, t_ in table if r_ != "none"]
    counts = {}
    for name, hpi, t_, r_, w_ in rows:
        c, r = score(taad_val, hpi, gt_va, val_keys, a.taad_thr, t_, r_, w_)
        counts[name] = c
        mark = "  <- chosen" if (r_, w_, t_) == (rule, w, thr) else ""
        print(f"  {name:40s} {r['f1']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['hidden_tp']:5d}/{r['hidden_n']} ({100 * r['hidden_recall']:4.1f}%) {r['n_hpi']:10d} {r['hpi_precision']:9.2f}{mark}")
        out["val"][name] = r

    chosen = f"{rule} w={w} thr {thr}" if rule != "none" else f"TAAD + HPI (old, thr {old_thr})"
    print(f"\n=== 4. SIGNIFICANCE (paired bootstrap, {a.n_boot} resamples) ===")
    out["bootstrap"] = {}
    for ref in ["TAAD alone", f"TAAD + HPI (old, thr {old_thr})"]:
        if ref == chosen:
            continue
        pair = {k: {"taad": counts[ref][k]["merged"], "merged": counts[chosen][k]["merged"]} for k in val_keys}
        p_h, ci_h = bootstrap(pair, val_keys, a.n_boot, rng)
        p_e, ci_e = bootstrap_events(pair, val_keys, gt_va, a.n_boot, rng)
        print(f"  chosen vs {ref:30s}: over halves P(gain<=0)={p_h:.3f} CI {np.round(ci_h, 3)} | "
              f"over events P={p_e:.3f} CI {np.round(ci_e, 3)}")
        out["bootstrap"][ref] = dict(p_half=p_h, ci_half=ci_h, p_events=p_e, ci_events=ci_e)

    # cross-check the chosen setting with the official evaluation code
    ev = {}
    for k in val_keys:
        tk, hk = apply_rule([e for e in taad_val.get(k, []) if e[4] >= a.taad_thr],
                            [e for e in hpi_val.get(k, []) if e[4] >= thr], rule, w)
        ev[k] = tk + hk
    p = os.path.join(a.run, "val_exclusive.json")
    merged_json(ev, None, val_keys, 0.0, 0.0, p)
    f1, pr, rc, tpn, gtn = evaluate(p, a.gt_val)
    print(f"\n  official evaluation.py on chosen setting: F1 {f1:.3f} P {pr:.3f} R {rc:.3f} hidden {tpn}/{gtn}")
    out["official"] = dict(f1=f1, precision=pr, recall=rc, hidden_tp=tpn, hidden_gt=gtn)
    json.dump(out, open(os.path.join(a.run, "exclusivity.json"), "w"), indent=2, default=float)
    print("saved", os.path.join(a.run, "exclusivity.json"))


if __name__ == "__main__":
    main()
