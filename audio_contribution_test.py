"""
Does the trained audio branch (3-layer 1D CNN + learned-delay head) actually contribute?

The fusion is additive (final logits = video logits + audio logits), so from the saved
validation logits of the audio model we can recompute its audio part exactly with the
checkpoint's audio weights, and evaluate:
  full      : saved logits (video + real audio)                       -> must reproduce the reported F1
  no_audio  : saved logits - audio part (same trained video part, audio removed)
  shifted   : video part + audio part computed from audio shifted by --shift_sec (wrong moment)
  other_game: video part + audio part computed from another game's audio (wrong match)

Example:
  python audio_contribution_test.py --checkpoint runs/TAAD_audiofusion_kaggle/checkpoints/curr_model_40.pt
"""
import argparse
import json
import os
import shutil
import tempfile

import h5py
import numpy as np
import torch

from models.audio_fusion import AudioEncoder, EventConditionedAudioHead, TAAD_AUDIO_MARGIN
from run_TAAD_on_matches import get_audio_window
from utils.metric_utils import post_processing_TAAD_preds_and_export, evaluate_events_from_json

CLASS_NAMES = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']
T = 50  # clip length used at inference


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join("runs", "TAAD_audiofusion_kaggle", "checkpoints", "curr_model_40.pt"))
    p.add_argument("--logits_dir", default=os.path.join("kaggle_preds", "val", "TAAD_predictions"))
    p.add_argument("--tactical", default=os.path.join("data", "val_tactical_data.h5"))
    p.add_argument("--audio_h5", default=os.path.join("data", "audio_features.h5"))
    p.add_argument("--gt", default=os.path.join("playbyplay_GT", "playbyplay_val.json"))
    p.add_argument("--shift_sec", type=float, default=60.0)
    p.add_argument("--nms_window", type=int, default=25)
    p.add_argument("--thresholds", default="0.15,0.70")
    return p.parse_args()


def load_audio_branch(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    enc = AudioEncoder(in_channels=2, hidden=32, embed_dim=32)
    head = EventConditionedAudioHead(embed_dim=32, num_action_classes=8, max_abs_tau=TAAD_AUDIO_MARGIN)
    enc.load_state_dict({k[len("audio_encoder."):]: v for k, v in sd.items() if k.startswith("audio_encoder.")})
    head.load_state_dict({k[len("audio_head."):]: v for k, v in sd.items() if k.startswith("audio_head.")})
    return enc.eval(), head.eval()


@torch.no_grad()
def audio_part(enc, head, audio_h5, audio_game, start, length, frame_shift=0):
    """Audio logits (9, length) for frames start..start+length, computed clip by clip exactly as at inference
    (clips of 50 frames starting at `start`). frame_shift reads the audio from a different time."""
    starts = list(range(start, start + length, T))
    wins, masks = [], []
    for sf in starts:
        e, m = get_audio_window(audio_h5, audio_game, sf + frame_shift, sf + frame_shift + T, TAAD_AUDIO_MARGIN)
        wins.append(e)
        masks.append(m)
    out = []
    for i in range(0, len(wins), 512):
        x = torch.from_numpy(np.stack(wins[i:i + 512]))
        m = torch.tensor(masks[i:i + 512]).view(-1, 1, 1)
        logit = head(enc(x), offset=TAAD_AUDIO_MARGIN, output_length=T) * m  # (b, 9, 50)
        out.append(logit.permute(1, 0, 2).reshape(9, -1).numpy())
    return np.concatenate(out, axis=1)[:, :length]


def two_pass_audio(enc, head, audio_h5, audio_game, minf, L_total, frame_shift=0):
    """Mirror run_TAAD_on_matches: pass 1 on the clip grid from minf, pass 2 shifted by T//2, averaged."""
    a1 = audio_part(enc, head, audio_h5, audio_game, minf, L_total, frame_shift)
    a = a1.copy()
    s2 = minf + T // 2
    a2 = audio_part(enc, head, audio_h5, audio_game, s2, L_total - T // 2, frame_shift)
    n = a2.shape[1] - 25
    a[:, T // 2:T // 2 + n] = (a1[:, T // 2:T // 2 + n] + a2[:, :n]) / 2
    return a


def evaluate_variant(name, logits_by_half, tactical, gt, nms, thresholds, workdir):
    root = os.path.join(workdir, name)
    os.makedirs(os.path.join(root, "TAAD_predictions"), exist_ok=True)
    for k, arr in logits_by_half.items():
        np.save(os.path.join(root, "TAAD_predictions", f"avg_logits_{k}.npy"), arr.astype(np.float16))
    out_json = os.path.join(root, "pred.json")
    post_processing_TAAD_preds_and_export(nms, root, out_json, tactical)
    res = {}
    for thr in thresholds:
        m = evaluate_events_from_json(gt_json_path=gt, pred_json_path=out_json, class_names=CLASS_NAMES,
                                      delta=12, conf_thresh=thr, print_per_game=False)
        res[thr] = m
    return res


def main():
    a = parse_args()
    thresholds = [float(t) for t in a.thresholds.split(",")]
    enc, head = load_audio_branch(a.checkpoint)
    print("learned tau (frames):", np.round(head.tau_frames(), 1))

    with h5py.File(a.tactical, "r") as f:
        halves = list(f.keys())
        minf = {}
        for k in halves:
            d = f[k][:]
            d = d[~np.isnan(d[:, 9])]
            minf[k] = int(d[:, 0].min())

    games = sorted({k.rsplit("_", 1)[0] for k in halves})
    other = {g: games[(i + 1) % len(games)] for i, g in enumerate(games)}  # 18->24, 24->47, 47->18
    shift = int(round(a.shift_sec * 25))

    variants = {"full": {}, "no_audio": {}, "shifted": {}, "other_game": {}}
    for k in halves:
        full = np.load(os.path.join(a.logits_dir, f"avg_logits_{k}.npy")).astype(np.float32)  # (9, 26, minf+L)
        g = k.rsplit("_", 1)[0]
        L = full.shape[2] - minf[k]
        A = two_pass_audio(enc, head, a.audio_h5, g.split("_")[1], minf[k], L)
        video = full.copy()
        video[:, :, minf[k]:] -= A[:, None, :]
        A_shift = two_pass_audio(enc, head, a.audio_h5, g.split("_")[1], minf[k], L, frame_shift=shift)
        A_other = two_pass_audio(enc, head, a.audio_h5, other[g].split("_")[1], minf[k], L)
        variants["full"][k] = full
        variants["no_audio"][k] = video
        sv = video.copy(); sv[:, :, minf[k]:] += A_shift[:, None, :]; variants["shifted"][k] = sv
        ov = video.copy(); ov[:, :, minf[k]:] += A_other[:, None, :]; variants["other_game"][k] = ov
        print(f"{k}: mean |audio part| = {np.abs(A[1:]).mean():.3f}  (video logit std {video[:, :, minf[k]:].std():.3f})")

    workdir = tempfile.mkdtemp(prefix="audio_test_")
    try:
        rows = {}
        for name, lb in variants.items():
            rows[name] = evaluate_variant(name, lb, a.tactical, a.gt, a.nms_window, thresholds, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n=== Validation micro-F1 (6 halves) ===")
    print(f"{'variant':<12}" + "".join(f"  F1@{t:<5}" for t in thresholds))
    for name, r in rows.items():
        print(f"{name:<12}" + "".join(f"  {r[t]['overall']['f1']:<8.3f}" for t in thresholds))


if __name__ == "__main__":
    main()
