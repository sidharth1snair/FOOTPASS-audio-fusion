"""
Event Verifier: a second opinion from tracking for every TAAD prediction.

1. train : the HPI network (spatial transformer over 26 players + biGRU) is trained to spot the actions of ALL
           tracked players (HPI only learned hidden players). By default it does not see TAAD, so its opinion is
           independent of TAAD.
2. fuse  : for each TAAD candidate (score >= 0.5) features are read from the Verifier around the event frame
           (does the named player act, with this class, more than anyone else?). A logistic regression on dev
           matches 6, 8, 52 turns TAAD score + Verifier features into one score, and the threshold of the full
           system (verified TAAD + HPI + one-action rule) is chosen on the same dev matches. Validation is
           scored once, with everything fixed.
   The test matches are NOT touched here (use --test at the very end only).

Usage:
  python event_verifier.py train --run ver_seed0 --seed 0
  python event_verifier.py fuse  --run ver_seed0 --hpi_run v7_33g_seed0
"""
import argparse
import json
import os
import shutil
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hpi_exclusivity import score, apply_rule
from train_hidden import soft_targets, C, merged_json, evaluate
from train_hidden_v2 import load_half, HPIv2, predict_half, M
from utils.metric_utils import post_processing_TAAD_preds_and_export
from verify_hidden_v2 import match_counts

DEV_GAMES = ["6", "8", "52"]
TRAIN_H5 = os.path.join("hidden_data", "train33.h5")
VAL_H5 = os.path.join("hidden_data", "val.h5")
TEST_H5 = os.path.join("hidden_data", "test6.h5")
TRAIN_LOGITS = os.path.join("kaggle_preds", "TAAD_predictions")
VAL_LOGITS = os.path.join("kaggle_preds", "val", "TAAD_predictions")
TEST_LOGITS = os.path.join("kaggle_preds", "TAAD_predictions_test")
CAND_MIN = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["train", "fuse"])
    p.add_argument("--run", default="ver_seed0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--taad_context", action="store_true", help="variant: Verifier also sees TAAD's probabilities")
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--pos_frac", type=float, default=0.5)
    p.add_argument("--hpi_run", default="v7_33g_seed0")
    p.add_argument("--test", action="store_true", help="FINAL ONLY: also score the 6 test matches")
    p.add_argument("--control", choices=["none", "shift", "shuffle"], default="none",
                   help="sanity check: corrupt the Verifier output on validation only (dev fit is unchanged)")
    return p.parse_args()


# ----------------------------------------------------------------------------- training

def sample_batch(halves, keys, pos, n, W, sigma, pos_frac, rng):
    xs, cs, ps, ys, ms = [], [], [], [], []
    for _ in range(n):
        if rng.random() < pos_frac:
            k, t = pos[rng.integers(len(pos))]
            t = t + rng.integers(-W // 3, W // 3)
        else:
            k = keys[rng.integers(len(keys))]
            t = rng.integers(halves[k]["x"].shape[0])
        h = halves[k]
        T = h["x"].shape[0]
        t0 = int(np.clip(t - W // 2, 0, max(T - W, 0)))
        sl = slice(t0, t0 + W)
        xs.append(h["x"][sl].astype(np.float32))
        cs.append(h["ctx"][sl].astype(np.float32))
        ps.append(h["present"][sl])
        lab = h["labels"][sl]
        ys.append(np.stack([soft_targets(lab[:, s], sigma) for s in range(M)], 1))
        ms.append(h["present"][sl].astype(np.float32))  # ALL tracked players, visible or not
    return [torch.from_numpy(np.stack(a)) for a in (xs, cs, ps, ys, ms)]


def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = os.path.join("runs_hidden", a.run)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with h5py.File(TRAIN_H5, "r") as f:
        keys = sorted(k for k in f.keys() if k.split("_")[1] not in DEV_GAMES)
        smp = np.concatenate([f[k]["feats"][::50][f[k]["present"][::50]].astype(np.float32) for k in keys])
        mean, std = smp.mean(0), smp.std(0) + 1e-6
        mean[-2:], std[-2:] = 0.0, 1.0
        train = {k: load_half(f[k], TRAIN_LOGITS, mean, std, a.taad_context) for k in keys}
    pos = [(k, t) for k, h in train.items() for t in np.nonzero(((h["labels"] > 0) & h["present"]).any(1))[0]]
    counts = np.zeros(C)
    for h in train.values():
        counts += np.bincount(h["labels"][h["present"]], minlength=C)
    cw = np.ones(C)
    cw[1:] = np.clip((counts[1:].sum() / np.maximum(counts[1:], 1) / (C - 1)) ** 0.5, 0.5, 5.0)
    cw = torch.tensor(cw, dtype=torch.float32, device=device)
    fp, fg = train[keys[0]]["x"].shape[-1], train[keys[0]]["ctx"].shape[-1]
    model = HPIv2(fp, fg).to(device)
    print(f"[INFO] {len(train)} training halves (dev games {DEV_GAMES} held out) | {len(pos)} event frames | "
          f"TAAD context {a.taad_context} | {sum(p.numel() for p in model.parameters()):,} parameters | "
          f"loaded in {time.time() - t0:.0f}s | {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.max_steps, pct_start=0.05)
    model.train()
    for step in range(1, a.max_steps + 1):
        x, c, pr, y, m = (t.to(device) for t in sample_batch(train, keys, pos, a.batch_size, a.window, a.sigma, a.pos_frac, rng))
        lf = -(y * F.log_softmax(model(x, c, pr), -1) * cw).sum(-1)
        loss = (lf * m).sum() / m.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 1:
            print(f"step {step:6d}/{a.max_steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "args": vars(a)}, os.path.join(out, "model.pt"))
    print("[INFO] saved", os.path.join(out, "model.pt"), flush=True)


# ----------------------------------------------------------------------------- fusion

def taad_events_for(keys, logits, tactical, path):
    """TAAD candidate events (all exported peaks) for the given halves, cached."""
    if os.path.exists(path):
        return json.load(open(path))["events"]
    tmp = path + "_tmp"
    os.makedirs(os.path.join(tmp, "TAAD_predictions"), exist_ok=True)
    for k in keys:
        shutil.copy(os.path.join(logits, f"avg_logits_{k}.npy"), os.path.join(tmp, "TAAD_predictions"))
    post_processing_TAAD_preds_and_export(nms_window=25, root_path=tmp, save_path=path, h5file_path=tactical)
    shutil.rmtree(tmp)
    return json.load(open(path))["events"]


def features(ev, P, h):
    """Verifier evidence for one TAAD candidate [frame, team, shirt, class, score]."""
    f, team, shirt, cls, s = int(ev[0]), int(ev[1]), int(ev[2]), int(ev[3]), float(ev[4])
    T = P.shape[0]
    t = int(np.clip(f - h["frame0"], 0, T - 1))
    lo, hi = max(t - 12, 0), min(t + 13, T)
    slots = [j for j in range(team * 13, team * 13 + 13) if int(h["shirt"][t, j]) == shirt]
    act = 1.0 - P[lo:hi, :, 0]                       # (w,26) any action
    if slots and h["present"][t, slots[0]]:
        j = slots[0]
        v_c, v_act = P[lo:hi, j, cls].max(), act[:, j].max()
        others = np.ones(M, bool)
        others[j] = False
        o_c, o_act = P[lo:hi][:, others, cls].max(), act[:, others].max()
        vis, found = float(h["visible"][t, j]), 1.0
    else:
        v_c = v_act = 0.0
        o_c, o_act = P[lo:hi, :, cls].max(), act.max()
        vis, found = 0.0, 0.0
    lg = lambda p: np.log(np.clip(p, 1e-4, 1.0))
    one = np.zeros(8)
    one[cls - 1] = 1.0
    return np.concatenate([[np.log(s / (1 - min(s, 0.9999))), lg(v_c), lg(v_act), lg(o_c), lg(o_act),
                            lg(v_c) - lg(o_c), lg(v_act) - lg(o_act), float(v_act >= o_act), vis, found], one])


def candidates(h5, logits_dir, taad, model, ck, use_taad, keys, control="none"):
    out = {}
    rng = np.random.default_rng(0)
    with h5py.File(h5, "r") as f:
        for k in keys:
            h = load_half(f[k], logits_dir, ck["mean"], ck["std"], use_taad)
            P = predict_half(model, h, device, ck["args"]["window"])
            if control == "shift":      # Verifier opinion taken from 5 s later: timing broken
                P = np.roll(P, 125, axis=0)
            elif control == "shuffle":  # players swapped within each team: identity broken
                perm = np.concatenate([rng.permutation(13), 13 + rng.permutation(13)])
                P = P[:, perm]
            ev = [e for e in taad.get(k, []) if e[4] >= CAND_MIN]
            out[k] = (ev, np.array([features(e, P, h) for e in ev], np.float32).reshape(len(ev), -1))
            print(f"  {k}: {len(ev)} candidates", flush=True)
    return out


def labels(cands, gt):
    y = {}
    for k, (ev, _) in cands.items():
        pred = [e[:4] + [e[4]] for e in ev]
        y[k] = match_counts(pred, gt[k])[1].astype(np.float32)
    return y


def fit_lr(X, y, l2=1e-2, steps=3000):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd)
    yt = torch.tensor(y)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    W, B = w.detach().numpy(), float(b.detach())
    return lambda Z: 1 / (1 + np.exp(-(((Z - mu) / sd) @ W + B))), W


def macro_f1(taad, hpi, gt, keys, taad_thr, hpi_thr, rule, w):
    tp, fp, fn = np.zeros(C), np.zeros(C), np.zeros(C)
    for k in keys:
        tk, hk = apply_rule([e for e in taad.get(k, []) if e[4] >= taad_thr], [e for e in hpi.get(k, []) if e[4] >= hpi_thr], rule, w)
        pred = [e[:4] + [2.0] for e in tk] + hk
        used, ok = match_counts(pred, gt[k])
        for e, o in zip(pred, ok):
            (tp if o else fp)[int(e[3])] += 1
        for g, u in zip(gt[k], used):
            if not u:
                fn[int(g[3])] += 1
    return float((2 * tp[1:] / np.maximum(2 * tp[1:] + fp[1:] + fn[1:], 1)).mean())


CAL_COLS = [0] + list(range(10, 18))  # TAAD score + class only: per-class calibration, no Verifier


def rescored(cands, prob, cols=None):
    cols = cols or slice(None)
    return {k: [e[:4] + [float(q)] for e, q in zip(ev, prob(X[:, cols]))] if len(ev) else []
            for k, (ev, X) in cands.items()}


def choose_thr(name, t_dev, hpi_dev, gtd, dev_keys, c):
    best = None
    for thr in np.round(np.arange(0.20, 0.81, 0.05), 2):
        r = score(t_dev, hpi_dev, gtd, dev_keys, thr, c["thr"], c["rule"], c["w"])[1]
        if best is None or r["f1"] > best[1]:
            best = (float(thr), r["f1"])
    print(f"[dev] {name}: chosen threshold {best[0]} (dev F1 with HPI + rule {best[1]:.3f})", flush=True)
    return best[0]


def fuse(a):
    ck = torch.load(os.path.join("runs_hidden", a.run, "model.pt"), map_location="cpu", weights_only=False)
    use_taad = ck["args"]["taad_context"]
    with h5py.File(TRAIN_H5, "r") as f:
        dev_keys = sorted(k for k in f.keys() if k.split("_")[1] in DEV_GAMES)
    with h5py.File(VAL_H5, "r") as f:
        val_keys = sorted(f.keys())
    probe = load_half(h5py.File(VAL_H5, "r")[val_keys[0]], VAL_LOGITS, ck["mean"], ck["std"], use_taad)
    model = HPIv2(probe["x"].shape[-1], probe["ctx"].shape[-1]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    gt_tr = json.load(open(os.path.join("playbyplay_GT", "playbyplay_train.json")))["events"]
    gt_va = json.load(open(os.path.join("playbyplay_GT", "playbyplay_val.json")))["events"]
    taad_dev = taad_events_for(dev_keys, TRAIN_LOGITS, os.path.join("data", "train_tactical_data.h5"),
                               os.path.join("runs_hidden", "taad_dev_events.json"))
    taad_val = json.load(open("playbyplay_PRED_model40_audio.json"))["events"]
    hrun = os.path.join("runs_hidden", a.hpi_run)
    c = json.load(open(os.path.join(hrun, "exclusivity.json")))["chosen"]
    hpi_dev = json.load(open(os.path.join(hrun, "hpi_dev_events.json")))
    hpi_dev = hpi_dev.get("events", hpi_dev)
    hpi_val = json.load(open(os.path.join(hrun, "hpi_val_events.json")))["events"]

    print("[dev] Verifier on dev halves", flush=True)
    cd = candidates(TRAIN_H5, TRAIN_LOGITS, taad_dev, model, ck, use_taad, dev_keys)
    yd = labels(cd, gt_tr)
    X = np.concatenate([cd[k][1] for k in dev_keys])
    y = np.concatenate([yd[k] for k in dev_keys])
    prob, W = fit_lr(X, y)
    prob_cal, _ = fit_lr(X[:, CAL_COLS], y)
    names = ["TAAD score", "V: player+class", "V: player acts", "V: others class", "V: others act",
             "V: class margin", "V: act margin", "V: top actor", "visible", "player found"] + [f"class {i}" for i in range(1, 9)]
    print("[dev] logistic regression weights (standardised):", {n: round(float(w), 2) for n, w in zip(names, W)})

    gtd = {k: gt_tr[k] for k in dev_keys}
    base = score(taad_dev, hpi_dev, gtd, dev_keys, 0.70, c["thr"], c["rule"], c["w"])[1]
    print(f"[dev] baseline TAAD@0.70 + HPI + rule: F1 {base['f1']:.3f}")
    thr_cal = choose_thr("calibrated TAAD (score + class)", rescored(cd, prob_cal, CAL_COLS), hpi_dev, gtd, dev_keys, c)
    thr = choose_thr("verified TAAD (score + class + Verifier)", rescored(cd, prob), hpi_dev, gtd, dev_keys, c)

    print("\n[val] validation halves (scored once, settings fixed)", flush=True)
    cv = candidates(VAL_H5, VAL_LOGITS, taad_val, model, ck, use_taad, val_keys, a.control)
    if a.control != "none":
        print(f"[CONTROL] Verifier output corrupted on validation: {a.control}")
    tcal, tval = rescored(cv, prob_cal, CAL_COLS), rescored(cv, prob)
    rows = [("TAAD alone", taad_val, {}, 0.70, 0.0, "none"),
            ("TAAD + HPI + rule", taad_val, hpi_val, 0.70, c["thr"], c["rule"]),
            ("Calibrated TAAD alone", tcal, {}, thr_cal, 0.0, "none"),
            ("Calibrated TAAD + HPI + rule", tcal, hpi_val, thr_cal, c["thr"], c["rule"]),
            ("Verified TAAD alone", tval, {}, thr, 0.0, "none"),
            ("Verified TAAD + HPI + rule", tval, hpi_val, thr, c["thr"], c["rule"])]
    res = {}
    print(f"{'system':30s} {'micro':>6s} {'macro':>6s} {'prec':>6s} {'recall':>6s} {'hidden':>10s}")
    for name, t_, h_, tt, ht, rule in rows:
        r = score(t_, h_, gt_va, val_keys, tt, ht, rule, c["w"])[1]
        r["macro"] = macro_f1(t_, h_, gt_va, val_keys, tt, ht, rule, c["w"])
        print(f"{name:30s} {r['f1']:6.3f} {r['macro']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} {r['hidden_tp']:5d}/{r['hidden_n']}")
        res[name] = r
    for name, t_, tt in [("Calibrated TAAD + HPI + rule", tcal, thr_cal), ("Verified TAAD + HPI + rule", tval, thr)]:
        ev = {}
        for k in val_keys:
            tk, hk = apply_rule([e for e in t_.get(k, []) if e[4] >= tt], [e for e in hpi_val.get(k, []) if e[4] >= c["thr"]], c["rule"], c["w"])
            ev[k] = tk + hk
        p = os.path.join("runs_hidden", a.run, "official_tmp.json")
        merged_json(ev, None, val_keys, 0.0, 0.0, p)
        f1, pr_, rc, tpn, gtn = evaluate(p, os.path.join("playbyplay_GT", "playbyplay_val.json"))
        print(f"[official evaluation.py] {name}: F1 {f1:.3f} P {pr_:.3f} R {rc:.3f} hidden {tpn}/{gtn}")
        res[name]["official_f1"] = f1
    tag = "" if a.control == "none" else f"_{a.control}"
    json.dump({"weights": dict(zip(names, map(float, W))), "thr": thr, "thr_cal": thr_cal, "dev_base": base, "val": res,
               "hpi_run": a.hpi_run, "control": a.control},
              open(os.path.join("runs_hidden", a.run, f"fusion_val{tag}.json"), "w"), indent=2, default=float)
    if a.control == "none":
        json.dump({"thr": thr, "events": tval}, open(os.path.join("runs_hidden", a.run, "verified_taad_val.json"), "w"))
    print("saved", os.path.join("runs_hidden", a.run, f"fusion_val{tag}.json"))
    if a.test and a.control == "none":
        final_test(a, model, ck, use_taad, prob, prob_cal, thr, thr_cal, c, hrun)


def final_test(a, model, ck, use_taad, prob, prob_cal, thr, thr_cal, c, hrun):
    """ONE-TIME score on the 6 frozen test matches. Weights and thresholds come from dev only."""
    from verify_hidden_v2 import bootstrap, bootstrap_events
    with h5py.File(TEST_H5, "r") as f:
        keys = sorted(f.keys())
    gt = json.load(open(os.path.join("playbyplay_GT", "playbyplay_train.json")))["events"]
    gt = {k: gt[k] for k in keys}
    taad = json.load(open(os.path.join("runs_hidden", "taad_test_events.json")))["events"]
    hpi = json.load(open(os.path.join(hrun, "hpi_test_events.json")))["events"]
    print("\n[TEST] 6 frozen test matches (scored once, all settings from dev)", flush=True)
    ct = candidates(TEST_H5, TEST_LOGITS, taad, model, ck, use_taad, keys)
    tcal, tver = rescored(ct, prob_cal, CAL_COLS), rescored(ct, prob)
    rows = [("TAAD alone", taad, {}, 0.70, 0.0, "none"),
            ("TAAD + HPI + rule", taad, hpi, 0.70, c["thr"], c["rule"]),
            ("Calibrated TAAD + HPI + rule", tcal, hpi, thr_cal, c["thr"], c["rule"]),
            ("Verified TAAD alone", tver, {}, thr, 0.0, "none"),
            ("Verified TAAD + HPI + rule", tver, hpi, thr, c["thr"], c["rule"])]
    res, cnt = {}, {}
    print(f"{'system':30s} {'micro':>6s} {'macro':>6s} {'prec':>6s} {'recall':>6s} {'hidden':>10s}")
    for name, t_, h_, tt, ht, rule in rows:
        cnt[name], r = score(t_, h_, gt, keys, tt, ht, rule, c["w"])
        r["macro"] = macro_f1(t_, h_, gt, keys, tt, ht, rule, c["w"])
        print(f"{name:30s} {r['f1']:6.3f} {r['macro']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} {r['hidden_tp']:5d}/{r['hidden_n']}")
        res[name] = r
    rng = np.random.default_rng(0)
    for ref in ["TAAD + HPI + rule", "Calibrated TAAD + HPI + rule"]:
        pair = {k: {"taad": cnt[ref][k]["merged"], "merged": cnt["Verified TAAD + HPI + rule"][k]["merged"]} for k in keys}
        ph, ch = bootstrap(pair, keys, 2000, rng)
        pe, ce = bootstrap_events(pair, keys, gt, 2000, rng)
        print(f"  Verified vs {ref}: 95% interval of the F1 gain, halves {np.round(ch, 3)} | events {np.round(ce, 3)}")
        res[f"ci_vs_{ref}"] = dict(halves=ch, events=ce)
    json.dump(res, open(os.path.join("runs_hidden", a.run, "final_test.json"), "w"), indent=2, default=lambda x: np.asarray(x).tolist())
    print("saved", os.path.join("runs_hidden", a.run, "final_test.json"))


if __name__ == "__main__":
    a = parse_args()
    train(a) if a.stage == "train" else fuse(a)
