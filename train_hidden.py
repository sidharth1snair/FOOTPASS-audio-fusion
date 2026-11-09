"""
Pilot: Hidden-Player event Inference (HPI).

TAAD only scores players that have a bounding box. This model scores the OTHER player-frames
(present in tracking, no box) from each player's own tracking features over time (+ optional audio),
then its events are merged with the TAAD+audio events and evaluated with the official matcher.

Data: hidden_data/{train,val}.h5 from build_hidden_dataset.py

Usage:
  python train_hidden.py --run_name pilot_tracking
  python train_hidden.py --run_name pilot_tracking_audio --use_audio
  python train_hidden.py --run_name smoke --max_steps 50 --eval_halves 1     (quick test)
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metric_utils import evaluate_events_from_json

CLASS_NAMES = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']
C = 9


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_h5", default=os.path.join("hidden_data", "train.h5"))
    p.add_argument("--val_h5", default=os.path.join("hidden_data", "val.h5"))
    p.add_argument("--gt_val", default=os.path.join("playbyplay_GT", "playbyplay_val.json"))
    p.add_argument("--taad_json", default="playbyplay_PRED_model40_audio.json")
    p.add_argument("--taad_thr", type=float, default=0.70)
    p.add_argument("--run_name", default="pilot")
    p.add_argument("--use_audio", action="store_true")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--sigma", type=float, default=4.0, help="Gaussian label width (frames)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos_frac", type=float, default=0.5, help="fraction of windows centred on a hidden event")
    p.add_argument("--dev_halves", type=int, default=6, help="train halves held out to choose the HPI threshold")
    p.add_argument("--eval_halves", type=int, default=None, help="only evaluate on the first N val halves (testing)")
    p.add_argument("--nms_window", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------------- data

def load_split(path, keys=None):
    halves = {}
    with h5py.File(path, "r") as f:
        for k in (keys or list(f.keys())):
            g = f[k]
            halves[k] = dict(feats=g["feats"][:], present=g["present"][:], visible=g["visible"][:],
                             labels=g["labels"][:], shirt=g["shirt"][:], audio=g["audio"][:].astype(np.float32),
                             frame0=int(g.attrs["frame0"]))
    return halves


def input_tensor(h, use_audio, mean, std):
    """(T, 26, F[+2]) normalised inputs for one half."""
    T = h["feats"].shape[0]
    F_in = h["feats"].shape[-1] + (2 if use_audio else 0)
    x = np.zeros((T, 26, F_in), np.float16)
    a = np.log1p(h["audio"] * 100.0) if use_audio else None  # compress loudness
    for s in range(0, T, 20000):
        xs = (h["feats"][s:s + 20000].astype(np.float32) - mean) / std
        xs[~h["present"][s:s + 20000]] = 0.0
        if use_audio:
            xs = np.concatenate([xs, np.broadcast_to(a[s:s + 20000, None, :], (xs.shape[0], 26, 2))], -1)
        x[s:s + 20000] = xs
    return x


def soft_targets(lab, sigma):
    """lab (L,) event class at event frame for one player window -> (L,C) soft target (Gaussian in time)."""
    L = lab.shape[0]
    tgt = np.zeros((L, C), np.float32)
    r = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    for t in np.nonzero(lab)[0]:
        lo, hi = max(t - r, 0), min(t + r + 1, L)
        tgt[lo:hi, lab[t]] = np.maximum(tgt[lo:hi, lab[t]], w[lo - (t - r):hi - (t - r)])
    tgt[:, 0] = np.clip(1.0 - tgt[:, 1:].sum(-1), 0.0, 1.0)
    return tgt


class Sampler:
    """Windows of one player slot; half centred on hidden events, half on random hidden frames."""

    def __init__(self, halves, X, sigma, window, pos_frac, rng):
        self.h, self.X, self.sigma, self.W, self.pos_frac, self.rng = halves, X, sigma, window, pos_frac, rng
        self.keys = list(halves)
        self.pos = []   # (key, t, s) hidden events
        self.hid = {}   # key -> (t, s) arrays of hidden present frames
        for k, h in halves.items():
            hidden = h["present"] & ~h["visible"]
            ts, ss = np.nonzero(hidden & (h["labels"] > 0))
            self.pos += [(k, t, s) for t, s in zip(ts, ss)]
            self.hid[k] = np.nonzero(hidden)

    def batch(self, n):
        xs, ys, ms = [], [], []
        for _ in range(n):
            if self.rng.random() < self.pos_frac:
                k, t, s = self.pos[self.rng.integers(len(self.pos))]
                t = t + self.rng.integers(-self.W // 3, self.W // 3)
            else:
                k = self.keys[self.rng.integers(len(self.keys))]
                i = self.rng.integers(len(self.hid[k][0]))
                t, s = self.hid[k][0][i], self.hid[k][1][i]
            h = self.h[k]
            T = h["labels"].shape[0]
            t0 = int(np.clip(t - self.W // 2, 0, max(T - self.W, 0)))
            sl = slice(t0, t0 + self.W)
            xs.append(self.X[k][sl, s].astype(np.float32))
            ys.append(soft_targets(h["labels"][sl, s], self.sigma))
            ms.append((h["present"][sl, s] & ~h["visible"][sl, s]).astype(np.float32))  # loss only on hidden frames
        return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ms)))


# ----------------------------------------------------------------------------- model

class HPIPilot(nn.Module):
    """Per-player sequence model: Linear -> 2-layer bidirectional GRU -> per-frame class logits."""

    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * hidden, C)

    def forward(self, x):  # x (B, T, F)
        h, _ = self.gru(self.inp(x))
        return self.out(h)  # (B, T, C)


@torch.no_grad()
def predict_half(model, X, device, chunk=2000, overlap=100):
    """Run the whole half for all 26 slots in overlapping chunks -> probabilities (T, 26, C)."""
    T = X.shape[0]
    probs = np.zeros((T, 26, C), np.float32)
    x = torch.from_numpy(X).permute(1, 0, 2).contiguous()  # (26, T, F) float16
    for s in range(0, T, chunk):
        lo, hi = max(s - overlap, 0), min(s + chunk + overlap, T)
        p = model(x[:, lo:hi].to(device).float()).softmax(-1).cpu().numpy()  # (26, L, C)
        keep = slice(s - lo, s - lo + min(chunk, T - s))
        probs[s:s + chunk] = p[:, keep].transpose(1, 0, 2)
    return probs


def events_from_probs(probs, h, nms_window):
    """Peak picking on hidden player-frames only -> [frame, team, shirt, class, score]."""
    hidden = h["present"] & ~h["visible"]
    events = []
    half = nms_window // 2
    for s in range(26):
        for c in range(1, C):
            sc = np.where(hidden[:, s], probs[:, s, c], 0.0)
            cand = np.nonzero(sc > 0.05)[0]
            if cand.size == 0:
                continue
            order = cand[np.argsort(-sc[cand])]
            taken = np.zeros(sc.shape[0], bool)
            for t in order:
                if taken[t]:
                    continue
                taken[max(t - half, 0):t + half + 1] = True
                shirt = int(h["shirt"][t, s])
                events.append([int(t + h["frame0"]), int(s // 13), shirt, int(c), float(sc[t])])
    return events


def merged_json(taad_events, hpi_events, keys, taad_thr, hpi_thr, path):
    out = {"keys": keys, "events": {}}
    for k in keys:
        ev = [e[:4] + [1.0] for e in taad_events.get(k, []) if e[4] >= taad_thr]
        if hpi_events is not None:
            ev += [e[:4] + [1.0] for e in hpi_events.get(k, []) if e[4] >= hpi_thr]
        out["events"][k] = ev
    with open(path, "w") as f:
        json.dump(out, f)


def evaluate(path, gt, keys=None):
    import contextlib, io
    gt_used = gt
    if keys is not None:  # restrict GT to these halves
        g = json.load(open(gt))
        gt_used = path + ".gt.json"
        json.dump({"keys": keys, "events": {k: g["events"][k] for k in keys}}, open(gt_used, "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        m = evaluate_events_from_json(gt_json_path=gt_used, pred_json_path=path, class_names=CLASS_NAMES,
                                      delta=12, conf_thresh=0.5, print_per_game=False)
    tp_nobb = sum(m["tp_splits"]["no_bbox"][1:]) if isinstance(m["tp_splits"]["no_bbox"], (list, np.ndarray)) else sum(m["tp_splits"]["no_bbox"].values())
    gt_nobb = sum(m["gt_splits"]["no_bbox"][1:]) if isinstance(m["gt_splits"]["no_bbox"], (list, np.ndarray)) else sum(m["gt_splits"]["no_bbox"].values())
    return m["overall"]["f1"], m["overall"]["precision"], m["overall"]["recall"], tp_nobb, gt_nobb


# ----------------------------------------------------------------------------- main

def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join("runs_hidden", a.run_name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    with h5py.File(a.train_h5, "r") as f:
        all_train = sorted(f.keys())
    dev_keys = all_train[-a.dev_halves:] if a.dev_halves else []
    tr_keys = [k for k in all_train if k not in dev_keys]
    train = load_split(a.train_h5, tr_keys)
    dev = load_split(a.train_h5, dev_keys) if dev_keys else {}
    val = load_split(a.val_h5)
    if a.eval_halves:
        val = {k: val[k] for k in list(val)[:a.eval_halves]}

    sample = np.concatenate([h["feats"][h["present"]][::20].astype(np.float32) for h in train.values()])
    mean, std = sample.mean(0), sample.std(0) + 1e-6
    mean[-2:], std[-2:] = 0.0, 1.0  # keep present / visible flags as 0/1
    del sample
    Xtr = {}
    for k, h in train.items():
        Xtr[k] = input_tensor(h, a.use_audio, mean, std)
        del h["feats"]  # free raw features once normalised
    print(f"[INFO] train {len(train)} halves, dev {len(dev)}, val {len(val)} | loaded in {time.time() - t0:.0f}s | device {device}", flush=True)

    sampler = Sampler(train, Xtr, a.sigma, a.window, a.pos_frac, rng)
    print(f"[INFO] {len(sampler.pos)} hidden training events", flush=True)

    # class weights from hidden event counts (background weight 1)
    counts = np.bincount([train[k]["labels"][t, s] for k, t, s in sampler.pos], minlength=C).astype(np.float64)
    cw = np.ones(C)
    cw[1:] = np.clip((counts[1:].sum() / np.maximum(counts[1:], 1) / (C - 1)) ** 0.5, 0.5, 5.0)
    cw = torch.tensor(cw, dtype=torch.float32, device=device)

    in_dim = next(iter(Xtr.values())).shape[-1]
    model = HPIPilot(in_dim, a.hidden_dim).to(device)
    print(f"[INFO] model parameters: {sum(p.numel() for p in model.parameters()):,} | input dim {in_dim}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.max_steps, pct_start=0.1)

    model.train()
    for step in range(1, a.max_steps + 1):
        x, y, m = (t.to(device) for t in sampler.batch(a.batch_size))
        logp = F.log_softmax(model(x), -1)
        loss_f = -(y * logp * cw).sum(-1)                  # soft cross-entropy, class-weighted
        loss = (loss_f * m).sum() / m.sum().clamp(min=1.0)  # hidden frames only
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            print(f"step {step:5d}/{a.max_steps}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "args": vars(a)}, os.path.join(out_dir, "model.pt"))

    # ---- predict
    model.eval()
    def run(split):
        evs = {}
        for k, h in split.items():
            probs = predict_half(model, input_tensor(h, a.use_audio, mean, std), device)
            evs[k] = events_from_probs(probs, h, a.nms_window)
        return evs
    hpi_dev, hpi_val = run(dev), run(val)

    # ---- choose the HPI threshold on held-out TRAIN halves (no tuning on val); dev has no TAAD events, so use hidden-only F1
    gt_train = os.path.join("playbyplay_GT", "playbyplay_train.json")
    best_thr, best_f1 = 0.5, -1
    if dev:
        for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            p = os.path.join(out_dir, "dev_tmp.json")
            merged_json({}, hpi_dev, list(dev), 0.0, thr, p)
            f1, pr, rc, tpn, gtn = evaluate(p, gt_train, list(dev))
            hid_r = tpn / max(gtn, 1)
            n_pred = sum(1 for k in dev for e in hpi_dev[k] if e[4] >= thr)
            hid_p = tpn / max(n_pred, 1)
            hid_f1 = 2 * hid_p * hid_r / max(hid_p + hid_r, 1e-9)
            print(f"[dev] thr {thr:.1f}: hidden recall {hid_r:.3f} precision {hid_p:.3f} hidden-F1 {hid_f1:.3f}")
            if hid_f1 > best_f1:
                best_thr, best_f1 = thr, hid_f1
    print(f"[INFO] chosen HPI threshold (on held-out train halves): {best_thr}")

    # ---- validation: TAAD alone vs TAAD + HPI
    taad = json.load(open(a.taad_json))["events"]
    keys = list(val)
    rows = []
    for name, hpi, thr in [("TAAD+audio alone", None, None), ("TAAD+audio + HPI", hpi_val, best_thr)]:
        p = os.path.join(out_dir, f"val_{'merged' if hpi else 'taad'}.json")
        merged_json(taad, hpi, keys, a.taad_thr, thr or 0, p)
        rows.append((name,) + evaluate(p, a.gt_val, keys if a.eval_halves else None))
    n_hpi = sum(1 for k in keys for e in hpi_val[k] if e[4] >= best_thr)
    print("\n=== VALIDATION ===")
    print(f"{'model':<20} {'F1':>6} {'P':>6} {'R':>6} {'hidden found':>14}")
    for name, f1, pr, rc, tpn, gtn in rows:
        print(f"{name:<20} {f1:6.3f} {pr:6.3f} {rc:6.3f} {tpn:6d}/{gtn} ({100 * tpn / max(gtn, 1):.1f}%)")
    print(f"HPI predicted {n_hpi} hidden events at threshold {best_thr}")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"rows": rows, "hpi_thr": best_thr, "n_hpi_events": n_hpi, "args": vars(a)}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
