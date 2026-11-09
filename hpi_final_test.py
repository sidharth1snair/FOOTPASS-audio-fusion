"""
Final one-time internal test of TAAD + HPI v7 on the 6 frozen test games (hidden_data/split_v2.json).

All settings are fixed before scoring: the 3 v7 models, and each seed's rule and HPI threshold from exclusivity.json
(chosen on dev games 6, 8, 52). Nothing here is tuned on the test games.

Before any test game is scored, the same code is run on validation and must reproduce the saved numbers.
HPI has never seen the test games. TAAD did train on them, so overall F1 here is not a fully unseen score for TAAD.

Usage:
  python build_hidden_dataset.py --set train --halves_from_logits kaggle_preds/TAAD_predictions_test --out hidden_data/test6.h5
  python hpi_final_test.py
"""
import json
import os
import shutil
import sys

import h5py
import numpy as np
import torch

from hpi_exclusivity import score
from show_hpi_results import macro_f1
from train_hidden import events_from_probs
from train_hidden_v2 import load_half, HPIv2, predict_half
from utils.metric_utils import post_processing_TAAD_preds_and_export
from verify_hidden_v2 import bootstrap, bootstrap_events

RUNS = ["v7_33g_seed0", "v7_33g_seed1", "v7_33g_seed2"]
TEST_H5 = os.path.join("hidden_data", "test6.h5")
TEST_LOGITS = os.path.join("kaggle_preds", "TAAD_predictions_test")
VAL_H5 = os.path.join("hidden_data", "val.h5")
VAL_LOGITS = os.path.join("kaggle_preds", "val", "TAAD_predictions")
TAAD_THR = 0.70
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hpi_events(run, h5, logits):
    ck = torch.load(os.path.join("runs_hidden", run, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    ev, model = {}, None
    with h5py.File(h5, "r") as f:
        for k in sorted(f.keys()):
            h = load_half(f[k], logits, ck["mean"], ck["std"], not args.get("no_taad_context", False))
            if model is None:
                model = HPIv2(h["x"].shape[-1], h["ctx"].shape[-1], args.get("d", 96)).to(device)
                model.load_state_dict(ck["model"])
                model.eval()
            ev[k] = events_from_probs(predict_half(model, h, device, args["window"]), h, args["nms_window"])
    return ev


def taad_events(logits, tactical, path):
    if os.path.exists(path):
        return json.load(open(path))["events"]
    tmp = path + "_tmp"
    os.makedirs(os.path.join(tmp, "TAAD_predictions"), exist_ok=True)
    for n in os.listdir(logits):
        shutil.copy(os.path.join(logits, n), os.path.join(tmp, "TAAD_predictions"))
    post_processing_TAAD_preds_and_export(nms_window=25, root_path=tmp, save_path=path, h5file_path=tactical)
    shutil.rmtree(tmp)
    return json.load(open(path))["events"]


def check_on_validation():
    gt = json.load(open(os.path.join("playbyplay_GT", "playbyplay_val.json")))["events"]
    keys = list(gt)
    saved_taad = json.load(open("playbyplay_PRED_model40_audio.json"))["events"]
    new_taad = taad_events(VAL_LOGITS, os.path.join("data", "val_tactical_data.h5"), os.path.join("runs_hidden", "taad_val_check_events.json"))
    a = score(saved_taad, {}, gt, keys, TAAD_THR, 0.0, "none", 0)[1]
    b = score(new_taad, {}, gt, keys, TAAD_THR, 0.0, "none", 0)[1]
    print(f"[check] TAAD on val: saved {a['f1']:.4f} | rebuilt from logits {b['f1']:.4f}", flush=True)
    ok = abs(a["f1"] - b["f1"]) < 0.002
    for run in RUNS:
        p = os.path.join("runs_hidden", run)
        c = json.load(open(os.path.join(p, "exclusivity.json")))["chosen"]
        saved = json.load(open(os.path.join(p, "hpi_val_events.json")))["events"]
        new = hpi_events(run, VAL_H5, VAL_LOGITS)
        x = score(saved_taad, saved, gt, keys, TAAD_THR, c["thr"], c["rule"], c["w"])[1]
        y = score(new_taad, new, gt, keys, TAAD_THR, c["thr"], c["rule"], c["w"])[1]
        print(f"[check] {run} on val: saved F1 {x['f1']:.4f} hidden {x['hidden_tp']} | rebuilt F1 {y['f1']:.4f} hidden {y['hidden_tp']}", flush=True)
        ok &= abs(x["f1"] - y["f1"]) < 0.002 and abs(x["hidden_tp"] - y["hidden_tp"]) <= 5
    return ok


def main():
    split = json.load(open(os.path.join("hidden_data", "split_v2.json")))
    want = {f"game_{g}_H{h}" for g in split["hpi_test_games"] for h in (1, 2)}
    with h5py.File(TEST_H5, "r") as f:
        keys = sorted(f.keys())
    assert set(keys) == want, "test file does not hold exactly the frozen test halves"
    with h5py.File(os.path.join("hidden_data", "train33.h5"), "r") as f:
        assert not want & set(f.keys()), "a test half is in the HPI training file"
    if not check_on_validation():
        sys.exit("[STOP] validation check failed: test games NOT scored")
    print("[check] passed, scoring the test games\n", flush=True)

    gt = json.load(open(os.path.join("playbyplay_GT", "playbyplay_train.json")))["events"]
    gt = {k: gt[k] for k in keys}
    taad = taad_events(TEST_LOGITS, os.path.join("data", "train_tactical_data.h5"), os.path.join("runs_hidden", "taad_test_events.json"))
    rng = np.random.default_rng(0)
    base_c, r = score(taad, {}, gt, keys, TAAD_THR, 0.0, "none", 0)
    n_ev, n_hid = sum(len(v) for v in gt.values()), r["hidden_n"]
    print(f"Test: {len(keys)} halves, {n_ev} events, {n_hid} by hidden players\n")
    print(f"{'system':34s} {'micro':>6s} {'macro':>6s} {'prec':>6s} {'recall':>6s} {'hidden found':>18s} {'HPI prec':>8s}")
    row = lambda n, x, m: print(f"{n:34s} {x['f1']:6.3f} {m:6.3f} {x['precision']:6.3f} {x['recall']:6.3f} "
                                f"{x['hidden_tp']:6d}/{x['hidden_n']} ({100 * x['hidden_recall']:4.1f}%) {x['hpi_precision']:8.2f}", flush=True)
    out = {"taad": dict(r, macro=macro_f1(taad, {}, gt, keys, 0.0, "none")), "runs": {}}
    row("TAAD alone", r, out["taad"]["macro"])
    r2 = score(taad, {}, gt, keys, TAAD_THR, 0.0, "global", 12)[1]
    out["taad_rule"] = dict(r2, macro=macro_f1(taad, {}, gt, keys, 0.0, "global"))
    row("TAAD + rule", r2, out["taad_rule"]["macro"])
    plain, rule, same = [], [], []
    for run in RUNS:
        p = os.path.join("runs_hidden", run)
        c = json.load(open(os.path.join(p, "exclusivity.json")))["chosen"]
        dev_thr = json.load(open(os.path.join(p, "hpi_val_events.json")))["dev_thr"]
        ev = hpi_events(run, TEST_H5, TEST_LOGITS)
        json.dump({"keys": keys, "events": ev}, open(os.path.join(p, "hpi_test_events.json"), "w"))
        x = score(taad, ev, gt, keys, TAAD_THR, dev_thr, "none", 0)[1]
        cnt, y = score(taad, ev, gt, keys, TAAD_THR, c["thr"], c["rule"], c["w"])
        z = score(taad, ev, gt, keys, TAAD_THR, 0.2, c["rule"], c["w"])[1]
        mx, my = macro_f1(taad, ev, gt, keys, dev_thr, "none"), macro_f1(taad, ev, gt, keys, c["thr"], c["rule"])
        row(f"TAAD + HPI ({run})", x, mx)
        row("  + one-action rule", y, my)
        pair = {k: {"taad": base_c[k]["merged"], "merged": cnt[k]["merged"]} for k in keys}
        ph, ch = bootstrap(pair, keys, 2000, rng)
        pe, ce = bootstrap_events(pair, keys, gt, 2000, rng)
        print(f"    vs TAAD alone: halves P={ph:.3f} CI {np.round(ch, 3)} | events P={pe:.3f} CI {np.round(ce, 3)}", flush=True)
        out["runs"][run] = dict(plain=dict(x, macro=mx), rule=dict(y, macro=my), thr02=z, p_half=ph, ci_half=ch, p_events=pe, ci_events=ce)
        plain.append((x["f1"], mx, x["hidden_recall"]))
        rule.append((y["f1"], my, y["hidden_recall"]))
        same.append((z["f1"], z["hidden_recall"]))
    for n, v in [("TAAD + HPI", plain), ("TAAD + HPI + rule", rule)]:
        v = np.array(v)
        print(f"\n{n}, mean of 3 seeds: micro {v[:, 0].mean():.3f} +- {v[:, 0].std():.3f} | macro {v[:, 1].mean():.3f} +- {v[:, 1].std():.3f} | "
              f"hidden {100 * v[:, 2].mean():.1f}% +- {100 * v[:, 2].std():.1f}")
    v = np.array(same)
    print(f"\nSame threshold 0.2 + rule: micro {v[:, 0].mean():.3f} | hidden {100 * v[:, 1].mean():.1f}%")
    json.dump(out, open(os.path.join("runs_hidden", "final_test.json"), "w"), indent=2, default=float)
    print("saved runs_hidden/final_test.json")


if __name__ == "__main__":
    main()
