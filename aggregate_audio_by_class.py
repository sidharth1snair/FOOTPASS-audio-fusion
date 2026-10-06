import argparse
import json
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():

    p = argparse.ArgumentParser(description="Average crowd/commentary energy around events, per class, across many events")

    p.add_argument("--audio_features_file", type=str, default=os.path.join("data", "audio_features.h5"))
    p.add_argument("--gt_file", type=str, default=os.path.join("playbyplay_GT", "playbyplay_train.json"))
    p.add_argument("--window", type=int, default=75, help="Frames before/after each event (75 = 3s @25fps)")
    p.add_argument("--boundary_margin", type=int, default=250, help="Exclude events within this many frames of a half's start/end (kickoff/whistle noise)")
    p.add_argument("--min_events_per_class", type=int, default=5, help="Skip classes with fewer usable events than this")
    p.add_argument("--out_dir", type=str, default="audio_sanity_check")

    return p.parse_args()


class_names = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']


def get_valid_windows(gt, game, window, boundary_margin, n_frames):
    """Return dict: class_name -> list of (lo, hi) frame windows, one per usable event,
    excluding events too close to a half boundary or to the start/end of the recording."""

    half_keys = sorted(k for k in gt['keys'] if k.startswith(game + "_"))

    windows_by_class = {c: [] for c in class_names}

    for hk in half_keys:
        events = gt['events'][hk]
        if not events:
            continue

        half_start = min(e[0] for e in events)
        half_end = max(e[0] for e in events)

        for e in events:
            frame, cls = e[0], e[3]

            if frame - half_start < boundary_margin:
                continue
            if half_end - frame < boundary_margin:
                continue

            lo, hi = frame - window, frame + window
            if lo < 0 or hi >= n_frames:
                continue

            windows_by_class[class_names[cls]].append((lo, hi, frame))

    return windows_by_class


if __name__ == '__main__':

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gt = json.load(open(args.gt_file))

    with h5py.File(args.audio_features_file, 'r') as h5f:
        games = sorted(h5f.keys())
        print(f"[INFO] Aggregating over {len(games)} games: {games}")

        crowd_by_class = {c: [] for c in class_names}
        commentary_by_class = {c: [] for c in class_names}

        for game in games:
            if game + "_H1" not in gt['keys'] and game + "_H2" not in gt['keys']:
                print(f"[SKIP] {game}: no GT events found")
                continue

            crowd_energy = h5f[game]['crowd_energy'][:]
            commentary_energy = h5f[game]['commentary_energy'][:]
            n_frames = len(crowd_energy)

            windows_by_class = get_valid_windows(gt, game, args.window, args.boundary_margin, n_frames)

            for c, windows in windows_by_class.items():
                for lo, hi, frame in windows:
                    crowd_by_class[c].append(crowd_energy[lo:hi])
                    commentary_by_class[c].append(commentary_energy[lo:hi])

    classes_to_plot = [c for c in class_names if c != 'background' and len(crowd_by_class[c]) >= args.min_events_per_class]

    print("[INFO] Usable event counts per class:")
    for c in class_names:
        if c == 'background':
            continue
        print(f"    {c:10s}: {len(crowd_by_class[c])} events"
              + ("" if len(crowd_by_class[c]) >= args.min_events_per_class else "  (skipped, too few)"))

    if not classes_to_plot:
        print("[WARN] No class had enough usable events to plot.")
        exit()

    x = np.arange(-args.window, args.window)

    fig, axes = plt.subplots(len(classes_to_plot), 1, figsize=(8, 3.2 * len(classes_to_plot)))
    if len(classes_to_plot) == 1:
        axes = [axes]

    for ax, c in zip(axes, classes_to_plot):
        crowd_stack = np.stack(crowd_by_class[c])       # (n_events, 2*window)
        comm_stack = np.stack(commentary_by_class[c])

        n = crowd_stack.shape[0]

        crowd_mean, crowd_sem = crowd_stack.mean(axis=0), crowd_stack.std(axis=0) / np.sqrt(n)
        comm_mean, comm_sem = comm_stack.mean(axis=0), comm_stack.std(axis=0) / np.sqrt(n)

        ax.plot(x, crowd_mean, label='crowd energy (mean)', color='tab:blue')
        ax.fill_between(x, crowd_mean - crowd_sem, crowd_mean + crowd_sem, color='tab:blue', alpha=0.25)

        ax.plot(x, comm_mean, label='commentary energy (mean)', color='tab:orange')
        ax.fill_between(x, comm_mean - comm_sem, comm_mean + comm_sem, color='tab:orange', alpha=0.25)

        ax.axvline(0, color='red', linestyle='--', label='event frame')
        ax.set_title(f"{c}  (n={n} events, averaged across all games)")
        ax.set_xlabel('frames relative to event')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "aggregate_audio_by_class.png")
    plt.savefig(out_path, dpi=120)
    print(f"[INFO] Saved aggregate plot to {out_path}")
