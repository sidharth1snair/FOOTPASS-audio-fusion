import argparse
import json
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.optim import AdamW

from models.audio_fusion import AudioOnlyEventDetector

CLASS_NAMES = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']


def parse_args():

    p = argparse.ArgumentParser(description="Train a standalone audio-only event detector with a learned per-class lag")

    p.add_argument("--audio_features_file", type=str, default=os.path.join("data", "audio_features.h5"))
    p.add_argument("--gt_file", type=str, default=os.path.join("playbyplay_GT", "playbyplay_train.json"))
    p.add_argument("--val_games", type=str, nargs="+", default=["game_9", "game_10"], help="Games held out for validation")

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_tau", type=float, default=2.0, help="Separate, higher LR for the tau parameters (grid_sample gradients w.r.t. sampling position are small-scale)")
    p.add_argument("--label_dilation", type=int, default=3, help="Frames of dilation applied to training labels only")
    p.add_argument("--max_abs_tau", type=float, default=100.0, help="Max learned offset, in frames (100 = 4s @25fps)")

    p.add_argument("--conf_thresh", type=float, default=0.15)
    p.add_argument("--nms_window", type=int, default=15)
    p.add_argument("--delta", type=int, default=12)

    p.add_argument("--checkpoint_path", type=str, default=os.path.join("runs", "audio_baseline_best.pt"))
    p.add_argument("--fixed_tau", action="store_true", help="Ablation: keep tau at 0 (no learned audio delay)")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def build_frame_labels(gt, game, n_frames):
    """Build a (n_frames,) int array of class ids (0=background) from GT events."""

    sharp = np.zeros(n_frames, dtype=np.int64)
    half_keys = [k for k in gt['keys'] if k.startswith(game + "_")]

    for hk in half_keys:
        for e in gt['events'][hk]:
            frame, cls = e[0], e[3]
            if 0 <= frame < n_frames:
                sharp[frame] = cls

    return sharp


def dilate_labels(sharp, dilation):

    if dilation <= 0:
        return sharp.copy()

    dilated = sharp.copy()
    T = len(sharp)
    for idx in np.where(sharp != 0)[0]:
        cls = sharp[idx]
        start = max(0, idx - dilation)
        end = min(T, idx + dilation + 1)
        dilated[start:end] = cls

    return dilated


def evaluate_sequence(probs, sharp_labels, conf_thresh, nms_window, delta, n_classes=9):
    """
    probs: (C, T) class probabilities for a single game (softmax already applied)
    sharp_labels: (T,) int ground-truth class ids
    Returns per-class TP, FP, FN, GT counts (arrays of length n_classes).
    """

    T = probs.shape[1]

    cls_map = probs.argmax(axis=0)          # (T,)
    score_map = probs.max(axis=0)           # (T,)

    below = score_map < conf_thresh
    cls_map = cls_map.copy()
    cls_map[below] = 0

    # Per-class NMS: greedy sweep in score order, marking a +/-nms_window
    # "occupied" zone around each kept peak. O(n_dets * nms_window) instead
    # of the O(n_dets^2) pairwise comparison, which can blow up when an
    # undertrained model predicts non-background over huge stretches.
    for c in range(1, n_classes):
        t_idxs = np.where(cls_map == c)[0]
        if t_idxs.size == 0:
            continue
        scores = score_map[t_idxs]
        order = np.argsort(-scores)
        t_sorted = t_idxs[order]

        occupied = np.zeros(T, dtype=bool)
        suppressed_mask = np.zeros(t_sorted.shape[0], dtype=bool)

        for i, t0 in enumerate(t_sorted):
            if occupied[t0]:
                suppressed_mask[i] = True
                continue
            lo, hi = max(0, t0 - nms_window), min(T, t0 + nms_window + 1)
            occupied[lo:hi] = True

        suppressed_times = t_sorted[suppressed_mask]
        cls_map[suppressed_times] = 0

    # Collect GT events per class
    gt_events = {c: list(np.where(sharp_labels == c)[0]) for c in range(1, n_classes)}
    GT_per_class = np.array([0] + [len(gt_events[c]) for c in range(1, n_classes)])

    # Collect detections, sorted by score desc
    det_t = np.where(cls_map != 0)[0]
    dets = sorted([(score_map[t], cls_map[t], t) for t in det_t], key=lambda x: -x[0])

    matched = {c: np.zeros(len(gt_events[c]), dtype=bool) for c in range(1, n_classes)}
    TP_per_class = np.zeros(n_classes, dtype=int)
    FP_per_class = np.zeros(n_classes, dtype=int)

    for score, c, t_det in dets:
        times = gt_events[c]
        used = matched[c]
        best_i, best_dt = -1, delta + 1
        for i, t_gt in enumerate(times):
            if used[i]:
                continue
            dt = abs(t_gt - t_det)
            if dt <= delta and dt < best_dt:
                best_dt, best_i = dt, i
        if best_i >= 0:
            used[best_i] = True
            TP_per_class[c] += 1
        else:
            FP_per_class[c] += 1

    FN_per_class = GT_per_class - TP_per_class

    return TP_per_class, FP_per_class, FN_per_class, GT_per_class


if __name__ == '__main__':

    args = parse_args()

    os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    gt = json.load(open(args.gt_file))

    with h5py.File(args.audio_features_file, 'r') as h5f:
        all_games = sorted(h5f.keys())
        data = {}
        for g in all_games:
            crowd = h5f[g]['crowd_energy'][:]
            comm = h5f[g]['commentary_energy'][:]
            n_frames = len(crowd)
            sharp = build_frame_labels(gt, g, n_frames)
            data[g] = {
                'x': np.stack([crowd, comm], axis=0).astype(np.float32),  # (2,T)
                'sharp': sharp,
                'dilated': dilate_labels(sharp, args.label_dilation),
            }

    train_games = [g for g in all_games if g not in args.val_games]
    val_games = [g for g in all_games if g in args.val_games]
    print(f"[INFO] Train games: {train_games}")
    print(f"[INFO] Val games:   {val_games}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = AudioOnlyEventDetector(max_abs_tau=args.max_abs_tau).to(device)

    other_params = [p for n, p in model.named_parameters() if n != 'head.tau']
    groups = [{'params': other_params, 'lr': args.lr, 'weight_decay': 1e-4}]
    if args.fixed_tau:
        model.head.tau.requires_grad_(False)
        print("[INFO] Ablation: tau fixed at 0")
    else:
        groups.append({'params': [model.head.tau], 'lr': args.lr_tau, 'weight_decay': 0.0})
    opt = AdamW(groups)

    # Median-frequency balancing across the 8 action classes, so rare classes
    # (e.g. tackle: ~30 events) pull as much gradient as common ones
    # (e.g. pass: ~9000 events) instead of being drowned out under one softmax.
    class_counts = np.zeros(9, dtype=np.float64)
    for g in train_games:
        class_counts += np.bincount(data[g]['sharp'], minlength=9)
    fg_counts = class_counts[1:]
    median_fg = np.median(fg_counts[fg_counts > 0])
    fg_weights = np.clip(median_fg / np.maximum(fg_counts, 1), 0.4, 15.0)

    class_weights = torch.tensor([0.05] + fg_weights.tolist(), dtype=torch.float32, device=device)
    print(f"[INFO] Class counts (train, sharp labels): {class_counts.astype(int).tolist()}")
    print(f"[INFO] Class weights (median-frequency balanced): {[round(w, 2) for w in class_weights.cpu().tolist()]}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):

        model.train()
        np.random.shuffle(train_games)
        total_loss = 0.0

        for g in train_games:
            x = torch.from_numpy(data[g]['x']).unsqueeze(0).to(device)              # (1,2,T)
            y = torch.from_numpy(data[g]['dilated']).unsqueeze(0).long().to(device)  # (1,T)

            logits = model(x)  # (1,9,T)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_games)

        # ---- Validation ----
        model.eval()
        TP = np.zeros(9, dtype=int)
        FP = np.zeros(9, dtype=int)
        FN = np.zeros(9, dtype=int)
        GT = np.zeros(9, dtype=int)

        with torch.no_grad():
            for g in val_games:
                x = torch.from_numpy(data[g]['x']).unsqueeze(0).to(device)
                logits = model(x)  # (1,9,T)
                probs = logits.softmax(1)[0].cpu().numpy()  # (9,T)

                tp, fp, fn, gtc = evaluate_sequence(probs, data[g]['sharp'], args.conf_thresh, args.nms_window, args.delta)
                TP += tp; FP += fp; FN += fn; GT += gtc

        precision = np.divide(TP, TP + FP, out=np.zeros(9), where=(TP + FP) > 0)
        recall = np.divide(TP, TP + FN, out=np.zeros(9), where=(TP + FN) > 0)
        overall_TP, overall_FP, overall_GT = TP[1:].sum(), FP[1:].sum(), GT[1:].sum()
        overall_PR = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) > 0 else 0.0
        overall_REC = overall_TP / overall_GT if overall_GT > 0 else 0.0
        overall_F1 = 2 * overall_PR * overall_REC / (overall_PR + overall_REC) if (overall_PR + overall_REC) > 0 else 0.0

        tau_frames = model.head.tau_frames()
        tau_seconds = model.head.tau_seconds()

        print(f"\nEpoch {epoch:>3} | train loss {avg_loss:.4f} | val F1 {overall_F1:.4f} (P={overall_PR:.3f} R={overall_REC:.3f})")
        print("  " + " ".join(f"{CLASS_NAMES[c]}:P{precision[c]:.2f}/R{recall[c]:.2f}" for c in range(1, 9)))
        print("  learned tau (frames): " + " ".join(f"{CLASS_NAMES[c+1]}={tau_frames[c]:+.1f}" for c in range(8)))
        print("  learned tau (seconds): " + " ".join(f"{CLASS_NAMES[c+1]}={tau_seconds[c]:+.2f}s" for c in range(8)))

        if overall_F1 > best_val_f1:
            best_val_f1 = overall_F1
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_f1': overall_F1}, args.checkpoint_path)

    print(f"\n[INFO] Best val F1: {best_val_f1:.4f} -- checkpoint saved to {args.checkpoint_path}")
