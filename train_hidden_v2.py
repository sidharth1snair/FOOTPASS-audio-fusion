"""
HPI v2 (quick improvement round over the pilot in train_hidden.py).

Changes vs the pilot:
  1. Team context  : all 26 player slots are modelled together; at every frame a small
                     transformer lets players attend to each other (spatial attention),
                     then a bidirectional GRU runs over time for each player.
  2. Play context  : per-player TAAD(+audio) class probabilities for VISIBLE players (0 if hidden),
                     plus per-frame global context: number of visible players, any-visible flag,
                     max TAAD probability per class over visible players.
  3. Longer training.

Loss is still applied only on hidden player-frames (present in tracking, no bounding box).

Usage:
  python train_hidden_v2.py --run_name v2
  python train_hidden_v2.py --run_name smoke2 --max_steps 30 --eval_halves 1 --dev_halves 2
  python train_hidden_v2.py --run_name v3_focus --focus      (improvement 2: camera-focus features)
  python train_hidden_v2.py --run_name v4_chain_w250 --chain --window 250 --batch_size 16   (improvements 3 + 4)
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

from train_hidden import soft_targets, events_from_probs, merged_json, evaluate, C

M = 26


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_h5", default=os.path.join("hidden_data", "train.h5"))
    p.add_argument("--val_h5", default=os.path.join("hidden_data", "val.h5"))
    p.add_argument("--train_logits", default=os.path.join("kaggle_preds", "TAAD_predictions"), help="folder of avg_logits_*.npy or a preds_train.h5")
    p.add_argument("--val_logits", default=os.path.join("kaggle_preds", "val", "TAAD_predictions"), help="folder of avg_logits_*.npy or a preds_val.h5")
    p.add_argument("--gt_val", default=os.path.join("playbyplay_GT", "playbyplay_val.json"))
    p.add_argument("--gt_train", default=os.path.join("playbyplay_GT", "playbyplay_train.json"))
    p.add_argument("--taad_json", default="playbyplay_PRED_model40_audio.json")
    p.add_argument("--taad_thr", type=float, default=0.70)
    p.add_argument("--run_name", default="v2")
    p.add_argument("--no_taad_context", action="store_true", help="ablation: tracking + team context only")
    p.add_argument("--focus", action="store_true", help="add camera-focus (ball-area proxy) features")
    p.add_argument("--chain", action="store_true", help="add event-chain features (last / next visible event TAAD detected)")
    p.add_argument("--chain_drop", type=float, default=0.25, help="fraction of TAAD events dropped on TRAIN halves")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--d", type=int, default=96)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos_frac", type=float, default=0.5)
    p.add_argument("--dev_halves", type=int, default=6)
    p.add_argument("--dev_games", default=None, help="Comma-separated game numbers held out as dev, instead of the last --dev_halves halves")
    p.add_argument("--eval_halves", type=int, default=None)
    p.add_argument("--nms_window", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def softmax0(x):
    x = x - x.max(0, keepdims=True)
    e = np.exp(x)
    return e / e.sum(0, keepdims=True)


def focus_features(feats, present, visible):
    """Camera-focus proxy (the broadcast camera follows the ball).
    Focus = centroid of the visible players; during gaps (replays, close-ups) the last and next known
    focus are linearly interpolated over the gap.
    Returns per-player (T,26,3) [dist to interpolated / last / next focus] and context (T,4)
    [focus x, focus y, time since last visible frame, time until next visible frame]."""
    T = feats.shape[0]
    xy = feats[..., :2].astype(np.float32)
    nvis = visible.sum(1)
    cen = np.where(nvis[:, None] > 0, (xy * visible[..., None]).sum(1) / np.maximum(nvis, 1)[:, None], 0.0)
    idx = np.where(nvis > 0, np.arange(T), -1)
    last = np.maximum.accumulate(idx)
    nxt = np.where(nvis > 0, np.arange(T), T)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    has_l, has_n = last >= 0, nxt < T
    c_last = np.where(has_l[:, None], cen[np.clip(last, 0, T - 1)], 0.5)
    c_next = np.where(has_n[:, None], cen[np.clip(nxt, 0, T - 1)], 0.5)
    ds = np.where(has_l, np.arange(T) - last, 10000).astype(np.float32)
    du = np.where(has_n, nxt - np.arange(T), 10000).astype(np.float32)
    wl = du / np.maximum(ds + du, 1)  # weight of the last known focus
    c_int = wl[:, None] * c_last + (1 - wl[:, None]) * c_next
    dist = lambda c: np.sqrt(((xy - c[:, None, :]) ** 2).sum(-1))
    pp = np.stack([dist(c_int), dist(c_last), dist(c_next)], -1)
    pp[~present] = 0.0
    tt = lambda d: np.log1p(np.minimum(d, 2500) / 25) / 4.6  # 0 .. 1 for 0 .. 100 s
    ctx = np.stack([c_int[:, 0], c_int[:, 1], tt(ds), tt(du)], -1)
    return pp, ctx


def detect_visible_events(best, thr=0.5, nms=12):
    """best (T,3) = [score, slot, class] of the most confident visible TAAD action per frame.
    Peak picking across ALL players (one ball action at a time) -> event frames sorted by time."""
    cand = np.nonzero(best[:, 0] >= thr)[0]
    taken = np.zeros(best.shape[0], bool)
    keep = []
    for t in cand[np.argsort(-best[cand, 0])]:
        if not taken[t]:
            taken[max(t - nms, 0):t + nms + 1] = True
            keep.append(t)
    return np.sort(np.array(keep, dtype=np.int64))


def chain_features(feats, best, drop_p=0.0, rng=None):
    """Event-chain context from the visible-player events TAAD detected.
    For every frame: the last and the next detected event (class, time gap, actor team / slot / position).
    Returns per-player (T,26,6) [dist to last / next actor position, same team as last / next actor (+-1),
    is last / next actor] and context (T,18) [time since last, time until next, class one-hots of last / next]."""
    T = feats.shape[0]
    ev = detect_visible_events(best)
    if drop_p > 0 and ev.size:  # TAAD is near-perfect on its own training halves: drop some events to mimic val
        ev = ev[rng.random(ev.size) >= drop_p]
    pp = np.zeros((T, M, 6), np.float32)
    ctx = np.zeros((T, 18), np.float32)
    pp[..., :2] = 2.0
    ctx[:, :2] = 1.0
    if ev.size == 0:
        return pp, ctx
    slot, cls = best[ev, 1].astype(np.int64), best[ev, 2].astype(np.int64)
    pos = feats[ev, slot, :2].astype(np.float32)  # actor position at the event frame
    t = np.arange(T)
    team = np.arange(M) // 13
    xy = feats[..., :2].astype(np.float32)
    tt = lambda d: np.log1p(np.minimum(d, 2500) / 25) / 4.6
    for j, idx in enumerate([np.searchsorted(ev, t, "right") - 1, np.searchsorted(ev, t, "right")]):
        ok = (idx >= 0) & (idx < ev.size)
        i = np.clip(idx, 0, ev.size - 1)
        gap = np.abs(t - ev[i])
        pp[ok, :, j] = np.sqrt(((xy[ok] - pos[i[ok]][:, None, :]) ** 2).sum(-1))
        pp[ok, :, 2 + j] = np.where(team[None] == (slot[i[ok]] // 13)[:, None], 1.0, -1.0)
        pp[np.nonzero(ok)[0], slot[i[ok]], 4 + j] = 1.0
        ctx[ok, j] = tt(gap[ok])
        ctx[np.nonzero(ok)[0], 2 + 8 * j + cls[i[ok]] - 1] = 1.0
    return pp, ctx


def load_half(g, logits_dir, mean, std, use_taad, use_focus=False, use_chain=False, chain_drop=0.0, rng=None):
    """Returns dict with x (T,26,Fp) float16, ctx (T,Fg) float16, masks, labels, shirt, frame0."""
    feats = g["feats"][:]
    present, visible = g["present"][:], g["visible"][:]
    f0 = int(g.attrs["frame0"])
    T = feats.shape[0]
    Fp = feats.shape[-1] + (9 if use_taad else 0)
    x = np.zeros((T, M, Fp), np.float16)
    ctx = np.zeros((T, 2 + (9 if use_taad else 0)), np.float16)
    nvis = visible.sum(1)
    ctx[:, 0] = nvis / 10.0
    ctx[:, 1] = nvis > 0
    lg = None
    best = np.zeros((T, 3), np.float32)  # most confident visible action per frame (for the event chain)
    if use_taad:
        k = g.name.strip("/")
        if logits_dir.endswith(".h5"):  # preds_train.h5 / preds_val.h5 (same arrays as the .npy files)
            lg = h5py.File(logits_dir, "r")[k]
        else:
            lg = np.load(os.path.join(logits_dir, f"avg_logits_{k}.npy"), mmap_mode="r")  # (9,26,Tabs)
    for s in range(0, T, 20000):
        sl = slice(s, s + 20000)
        xs = (feats[sl].astype(np.float32) - mean) / std
        xs[~present[sl]] = 0.0
        if use_taad:
            a0, a1 = f0 + s, f0 + min(s + 20000, T)
            seg = np.zeros((9, M, a1 - a0), np.float32)
            hi = min(a1, lg.shape[2])
            if hi > a0:
                seg[:, :, :hi - a0] = np.asarray(lg[:, :, a0:hi], dtype=np.float32)
            pr = softmax0(seg).transpose(2, 1, 0)  # (t,26,9)
            pr[~visible[sl]] = 0.0                 # TAAD is meaningless for hidden players
            xs = np.concatenate([xs, pr], -1)
            ctx[sl, 2:] = pr.max(1)
            act = pr[..., 1:].reshape(pr.shape[0], -1)
            j = act.argmax(1)
            best[sl] = np.stack([act[np.arange(len(j)), j], j // 8, j % 8 + 1], -1)
        x[sl] = xs
    if use_focus:
        pp, fc = focus_features(feats, present, visible)
        x = np.concatenate([x, pp.astype(np.float16)], -1)
        ctx = np.concatenate([ctx, fc.astype(np.float16)], -1)
    if use_chain:
        assert use_taad, "the event chain is built from TAAD detections"
        pp, cc = chain_features(feats, best, chain_drop, rng)
        x = np.concatenate([x, pp.astype(np.float16)], -1)
        ctx = np.concatenate([ctx, cc.astype(np.float16)], -1)
    return dict(x=x, ctx=ctx, present=present, visible=visible, labels=g["labels"][:], shirt=g["shirt"][:], frame0=f0)


class HPIv2(nn.Module):
    def __init__(self, fp, fg, d=96):
        super().__init__()
        self.inp = nn.Linear(fp, d)
        self.ctx = nn.Linear(fg, d)
        self.slot = nn.Embedding(M, d)  # team + role identity
        self.spatial = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True), 2)
        self.gru = nn.GRU(d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * d, C)

    def forward(self, x, ctx, present):
        # x (B,W,26,Fp) ctx (B,W,Fg) present (B,W,26) bool
        B, W, _, _ = x.shape
        h = self.inp(x) + self.ctx(ctx)[:, :, None, :] + self.slot.weight[None, None]
        h = h.reshape(B * W, M, -1)
        pad = ~present.reshape(B * W, M)
        pad = pad & ~pad.all(1, keepdim=True)  # frames with nobody tracked: attend to all (avoid NaN)
        h = self.spatial(h, src_key_padding_mask=pad)
        h = h.reshape(B, W, M, -1).permute(0, 2, 1, 3).reshape(B * M, W, -1)
        h, _ = self.gru(h)
        return self.out(h).reshape(B, M, W, C).permute(0, 2, 1, 3)  # (B,W,26,C)


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
        ms.append((h["present"][sl] & ~h["visible"][sl]).astype(np.float32))
    return [torch.from_numpy(np.stack(a)) for a in (xs, cs, ps, ys, ms)]


@torch.no_grad()
def predict_half(model, h, device, W=100, stride=50, bs=64):
    T = h["x"].shape[0]
    probs = np.zeros((T, M, C), np.float32)
    starts = list(range(0, max(T - W, 0) + 1, stride))
    if starts[-1] + W < T:
        starts.append(T - W)
    for i in range(0, len(starts), bs):
        st = starts[i:i + bs]
        x = torch.from_numpy(np.stack([h["x"][s:s + W] for s in st]).astype(np.float32)).to(device)
        c = torch.from_numpy(np.stack([h["ctx"][s:s + W] for s in st]).astype(np.float32)).to(device)
        pr = torch.from_numpy(np.stack([h["present"][s:s + W] for s in st])).to(device)
        p = model(x, c, pr).softmax(-1).cpu().numpy()
        q = W // 4  # keep the centre of each window (edges have less context)
        for j, s in enumerate(st):
            lo = 0 if s == 0 else q
            hi = W if s + W >= T else W - q
            probs[s + lo:s + hi] = p[j, lo:hi]
    return probs


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_taad = not a.no_taad_context
    out_dir = os.path.join("runs_hidden", a.run_name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    with h5py.File(a.train_h5, "r") as f:
        all_train = sorted(f.keys())
        if a.dev_games:
            dev_keys = [k for k in all_train if k.split("_")[1] in a.dev_games.split(",")]
        else:
            dev_keys = all_train[-a.dev_halves:] if a.dev_halves else []
        tr_keys = [k for k in all_train if k not in dev_keys]
        smp = np.concatenate([f[k]["feats"][::50][f[k]["present"][::50]].astype(np.float32) for k in tr_keys])
        mean, std = smp.mean(0), smp.std(0) + 1e-6
        mean[-2:], std[-2:] = 0.0, 1.0
        train = {k: load_half(f[k], a.train_logits, mean, std, use_taad, a.focus, a.chain, a.chain_drop, rng) for k in tr_keys}
        dev = {k: load_half(f[k], a.train_logits, mean, std, use_taad, a.focus, a.chain) for k in dev_keys}
    with h5py.File(a.val_h5, "r") as f:
        vkeys = list(f.keys())[:a.eval_halves] if a.eval_halves else list(f.keys())
        val = {k: load_half(f[k], a.val_logits, mean, std, use_taad, a.focus, a.chain) for k in vkeys}
    pos = [(k, t) for k, h in train.items() for t in np.nonzero(((h["labels"] > 0) & h["present"] & ~h["visible"]).any(1))[0]]
    keys = list(train)
    fp, fg = next(iter(train.values()))["x"].shape[-1], next(iter(train.values()))["ctx"].shape[-1]
    print(f"[INFO] train {len(train)} dev {len(dev)} val {len(val)} | TAAD context {use_taad} | {len(pos)} hidden-event frames | "
          f"loaded in {time.time() - t0:.0f}s | device {device}", flush=True)

    counts = np.zeros(C)
    for k, h in train.items():
        lab = h["labels"][h["present"] & ~h["visible"]]
        counts += np.bincount(lab, minlength=C)
    cw = np.ones(C)
    cw[1:] = np.clip((counts[1:].sum() / np.maximum(counts[1:], 1) / (C - 1)) ** 0.5, 0.5, 5.0)
    cw = torch.tensor(cw, dtype=torch.float32, device=device)

    model = HPIv2(fp, fg, a.d).to(device)
    print(f"[INFO] parameters {sum(p.numel() for p in model.parameters()):,} | player features {fp} | context {fg}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.max_steps, pct_start=0.05)
    model.train()
    for step in range(1, a.max_steps + 1):
        x, c, pr, y, m = (t.to(device) for t in sample_batch(train, keys, pos, a.batch_size, a.window, a.sigma, a.pos_frac, rng))
        logp = F.log_softmax(model(x, c, pr), -1)
        lf = -(y * logp * cw).sum(-1)
        loss = (lf * m).sum() / m.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 1:
            print(f"step {step:6d}/{a.max_steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "args": vars(a)}, os.path.join(out_dir, "model.pt"))

    model.eval()
    hpi_dev = {k: events_from_probs(predict_half(model, h, device, a.window), h, a.nms_window) for k, h in dev.items()}
    hpi_val = {k: events_from_probs(predict_half(model, h, device, a.window), h, a.nms_window) for k, h in val.items()}

    best_thr, best = 0.5, -1
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        p = os.path.join(out_dir, "dev_tmp.json")
        merged_json({}, hpi_dev, list(dev), 0.0, thr, p)
        _, _, _, tpn, gtn = evaluate(p, a.gt_train, list(dev))
        n_pred = sum(1 for k in dev for e in hpi_dev[k] if e[4] >= thr)
        r, pr_ = tpn / max(gtn, 1), tpn / max(n_pred, 1)
        f1 = 2 * pr_ * r / max(pr_ + r, 1e-9)
        print(f"[dev] thr {thr:.2f}: hidden recall {r:.3f} precision {pr_:.3f} hidden-F1 {f1:.3f} ({n_pred} preds)")
        if f1 > best:
            best_thr, best = thr, f1
    print(f"[INFO] chosen HPI threshold (held-out train halves): {best_thr}")

    with open(os.path.join(out_dir, "hpi_val_events.json"), "w") as f:
        json.dump({"keys": list(val), "events": hpi_val, "dev_thr": best_thr}, f)
    taad = json.load(open(a.taad_json))["events"]
    vk = list(val)
    print("\n=== VALIDATION ===")
    print(f"{'model':<26} {'F1':>6} {'P':>6} {'R':>6} {'hidden found':>16} {'hidden prec':>12}")
    results = {}
    for name, hpi, thr in [("TAAD+audio alone", None, 0), (f"+ HPIv2 @dev-thr {best_thr}", hpi_val, best_thr)] + \
            [(f"+ HPIv2 @thr {t}", hpi_val, t) for t in (0.7, 0.8, 0.9, 0.95) if t != best_thr]:
        p = os.path.join(out_dir, "val_tmp.json")
        merged_json(taad, hpi, vk, a.taad_thr, thr, p)
        f1, pr_, rc, tpn, gtn = evaluate(p, a.gt_val, vk if a.eval_halves else None)
        n_h = sum(1 for k in vk for e in hpi[k] if e[4] >= thr) if hpi else 0
        hp = (tpn - 14 * (len(vk) == 6)) / max(n_h, 1) if hpi else float("nan")
        print(f"{name:<26} {f1:6.3f} {pr_:6.3f} {rc:6.3f} {tpn:6d}/{gtn} ({100 * tpn / max(gtn, 1):4.1f}%) {hp:12.3f}")
        results[name] = dict(f1=f1, precision=pr_, recall=rc, hidden_tp=tpn, hidden_gt=gtn, n_hpi=n_h)
    json.dump({"results": results, "dev_thr": best_thr, "args": vars(a)}, open(os.path.join(out_dir, "results.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
