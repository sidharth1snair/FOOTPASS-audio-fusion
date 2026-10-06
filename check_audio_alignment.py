import argparse
import json
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():

    p = argparse.ArgumentParser(description="Sanity-check extracted audio energy against known GT event frames")

    p.add_argument("--audio_features_file", type=str, default=os.path.join("data", "audio_features.h5"))
    p.add_argument("--gt_file", type=str, default=os.path.join("playbyplay_GT", "playbyplay_train.json"))
    p.add_argument("--game", type=str, default="game_1", help="Game key, e.g. game_1")
    p.add_argument("--window", type=int, default=75, help="Frames shown before/after each event (75 frames = 3s @25fps)")
    p.add_argument("--n_events", type=int, default=6, help="Number of events to plot")
    p.add_argument("--out_dir", type=str, default="audio_sanity_check")

    return p.parse_args()


class_names = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']


if __name__ == '__main__':

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.audio_features_file, 'r') as h5f:
        crowd_energy = h5f[args.game]['crowd_energy'][:]
        commentary_energy = h5f[args.game]['commentary_energy'][:]

    gt = json.load(open(args.gt_file))

    half_keys = [k for k in gt['keys'] if k.startswith(args.game + "_")]
    events = []
    for hk in half_keys:
        events.extend(gt['events'][hk])

    events = sorted(events, key=lambda e: e[0])

    # Prefer rarer, more audio-salient classes (shot/header/cross) over background-heavy drive/pass
    priority = ['shot', 'header', 'cross', 'tackle', 'block', 'throw-in', 'pass', 'drive']
    events_by_class = {c: [e for e in events if class_names[e[3]] == c] for c in priority}

    selected = []
    for c in priority:
        for e in events_by_class[c]:
            if e[0] - args.window >= 0 and e[0] + args.window < len(crowd_energy):
                selected.append(e)
                break
        if len(selected) >= args.n_events:
            break

    print(f"[INFO] {args.game}: {len(events)} total events, checking {len(selected)} events: "
          + ", ".join(f"{class_names[e[3]]}@frame{e[0]}" for e in selected))

    fig, axes = plt.subplots(len(selected), 1, figsize=(8, 3 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax, event in zip(axes, selected):
        frame, team, jersey, cls = event[0], event[1], event[2], event[3]
        lo, hi = frame - args.window, frame + args.window
        x = np.arange(lo, hi) - frame  # frames relative to event

        ax.plot(x, crowd_energy[lo:hi], label='crowd energy')
        ax.plot(x, commentary_energy[lo:hi], label='commentary energy')
        ax.axvline(0, color='red', linestyle='--', label='event frame')
        ax.set_title(f"{class_names[cls]} (team {team}, jersey {jersey}, frame {frame})")
        ax.set_xlabel('frames relative to event')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"{args.game}_audio_alignment.png")
    plt.savefig(out_path, dpi=120)
    print(f"[INFO] Saved sanity-check plot to {out_path}")
