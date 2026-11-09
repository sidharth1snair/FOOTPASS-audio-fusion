"""
Quick test: do PANNs audio features carry information about HIDDEN-player events, beyond what HPI already knows?

Samples : one point every 10 frames (the PANNs hop) in the 6 val halves, only where at least one player is hidden.
Target  : a hidden-player event (any class) within +-12 frames.
Features: H  = HPI v6 score (max over hidden players of 1 - p_background, max-pooled over +-12 frames)
          E  = our crowd / commentary energy (2 s window statistics)
          P  = PANNs CNN14 embedding (2048 -> PCA 64, fitted on the training games of each fold)
Model   : logistic regression, leave-one-GAME-out over the 3 val games, pooled ROC-AUC and average precision.
Controls: P from the WRONG game (another val game, same window index) and P shifted by 60 s.
          A real audio effect must disappear in both.

Usage:
  python audio_probe_panns.py --panns data/panns_raw.h5 --run runs_hidden/v6_w250
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch

from train_hidden_v2 import load_half, HPIv2, predict_half


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--panns", default=os.path.join("data", "panns_raw.h5"))
    p.add_argument("--run", default=os.path.join("runs_hidden", "v6_w250"))
    p.add_argument("--val_h5", default=os.path.join("hidden_data", "val.h5"))
    p.add_argument("--val_logits", default=os.path.join("kaggle_preds", "val", "TAAD_predictions"))
    p.add_argument("--pca", type=int, default=64)
    p.add_argument("--out", default=os.path.join("runs_hidden", "audio_probe_panns.json"))
    return p.parse_args()


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * (len(y) - n1), 1)


def avg_precision(y, s):
    o = np.argsort(-s)
    y = y[o]
    tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1))[y == 1].mean())


def logreg(Xtr, ytr, Xte, l2=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pw = (1 - y).sum() / y.sum().clamp(min=1)  # balance classes
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xtr @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=pw) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (Xte @ w + b).numpy()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(a.run, "model.pt"), map_location="cpu", weights_only=False)
    args = ck["args"]
    pan = h5py.File(a.panns, "r")

    rows = {}  # game -> dict of arrays
    with h5py.File(a.val_h5, "r") as f:
        halves = {k: load_half(f[k], a.val_logits, ck["mean"], ck["std"], not args.get("no_taad_context", False),
                               args.get("focus", False), args.get("chain", False)) for k in f.keys()}
        energy = {k: f[k]["audio"][:].astype(np.float32) for k in f.keys()}
    model = HPIv2(next(iter(halves.values()))["x"].shape[-1], next(iter(halves.values()))["ctx"].shape[-1], args.get("d", 96)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    for k, h in halves.items():
        game = k.rsplit("_", 1)[0]
        hidden = h["present"] & ~h["visible"]
        pr = predict_half(model, h, device, args.get("window", 100))
        sc = np.where(hidden, 1 - pr[..., 0], 0.0).max(1)  # (T,)
        T, f0 = len(sc), h["frame0"]
        pool = np.array([sc[max(t - 12, 0):t + 13].max() for t in range(T)])
        ev = ((h["labels"] > 0) & hidden).any(1)
        evn = np.convolve(ev.astype(float), np.ones(25), "same") > 0  # event within +-12 frames
        hop = int(pan[game].attrs["hop_frames"])
        idx = np.arange((f0 + hop - 1) // hop, (f0 + T - 1) // hop + 1)  # PANNs windows inside this half
        rel = idx * hop - f0
        ok = (rel >= 0) & (rel < T) & hidden[np.clip(rel, 0, T - 1)].any(1)
        idx, rel = idx[ok], rel[ok]
        e = energy[k]
        e = (e - e.mean(0)) / (e.std(0) + 1e-6)
        E = []
        for t in rel:
            w0, w1 = e[max(t - 25, 0):t + 26], e[max(t - 50, 0):max(t - 25, 1)]
            E.append(np.concatenate([w0.mean(0), w0.max(0), w0.std(0), w0.mean(0) - w1.mean(0)]))
        d = rows.setdefault(game, {"H": [], "E": [], "idx": [], "y": [], "half": []})
        d["H"].append(pool[rel]); d["E"].append(np.array(E)); d["idx"].append(idx); d["y"].append(evn[rel])
        d["half"].append(np.full(len(rel), k))
    for g, d in rows.items():
        for n in d:
            d[n] = np.concatenate(d[n])
        emb = pan[g]["emb"][:].astype(np.float32)
        d["P"] = emb[np.clip(d["idx"], 0, len(emb) - 1)]
        d["P_shift"] = emb[np.clip(d["idx"] + 150, 0, len(emb) - 1)]  # +60 s (150 windows of 0.4 s)
    games = sorted(rows)
    for i, g in enumerate(games):  # wrong game: next val game's audio at the same window index
        other = pan[games[(i + 1) % len(games)]]["emb"][:].astype(np.float32)
        rows[g]["P_wrong"] = other[np.clip(rows[g]["idx"], 0, len(other) - 1)]
    print("samples per game:", {g: (len(rows[g]["y"]), int(rows[g]["y"].sum())) for g in games})

    def feats(d, name, pcs):
        parts = []
        for p in name.split("+"):
            if p == "H":
                parts.append(d["H"][:, None])
            elif p == "E":
                parts.append(d["E"])
            else:  # P, P_wrong, P_shift
                parts.append((d[p] - pcs[0]) @ pcs[1])
        return np.concatenate(parts, 1)

    sets = ["H", "E", "P", "H+E", "H+P", "H+P_wrong", "H+P_shift", "P_wrong"]
    res = {s: {"y": [], "s": []} for s in sets}
    for g in games:
        tr = [x for x in games if x != g]
        Ptr = np.concatenate([rows[x]["P"] for x in tr])
        mu = Ptr.mean(0)
        _, _, vt = np.linalg.svd(Ptr[::4] - mu, full_matrices=False)
        pcs = (mu, vt[:a.pca].T)
        for s in sets:
            Xtr = np.concatenate([feats(rows[x], s.replace("_wrong", "").replace("_shift", ""), pcs) for x in tr])
            ytr = np.concatenate([rows[x]["y"] for x in tr]).astype(np.float32)
            res[s]["s"].append(logreg(Xtr, ytr, feats(rows[g], s, pcs)))
            res[s]["y"].append(rows[g]["y"])
    print(f"\n{'features':12s} {'AUC':>6} {'AP':>6}   (positives = hidden event within +-0.5 s; leave-one-game-out)")
    out = {}
    for s in sets:
        y, sc = np.concatenate(res[s]["y"]).astype(int), np.concatenate(res[s]["s"])
        out[s] = dict(auc=float(auc(y, sc)), ap=avg_precision(y, sc), base_rate=float(y.mean()))
        print(f"{s:12s} {out[s]['auc']:6.3f} {out[s]['ap']:6.3f}")
    print(f"(AP of a random guess = {out['H']['base_rate']:.3f})")
    json.dump(out, open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
