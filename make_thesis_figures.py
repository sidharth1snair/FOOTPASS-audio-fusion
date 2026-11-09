"""
Figures for Chapter 4 of the thesis (written to thesis/figures/).

  fig_f1_progression.png   - test F1 from TAAD to the final system (3-seed mean +- sd)
  fig_timeline.png         - 40 s of one validation half: ground truth vs three systems
  fig_hidden_example.png   - a hidden-player event: broadcast frame + pitch map from tracking
  fig_highlight_timeline.png - audio excitement of a match, goals and the clips chosen for the reel
  fig_visible_detection.png  - copy of an existing TAAD demo frame

Usage:
  python make_thesis_figures.py
"""
import json
import os
import shutil
import subprocess
import sys

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Arc

from hpi_exclusivity import apply_rule
from verify_hidden_v2 import match_counts

OUT = os.path.join("thesis", "figures")
os.makedirs(OUT, exist_ok=True)
FPS = 25
CLASS = ['bg', 'Drive', 'Pass', 'Cross', 'Throw-in', 'Shot', 'Header', 'Tackle', 'Block']
CCOL = ['k', '#8c8c8c', '#4C78A8', '#F58518', '#54A24B', '#E45756', '#B279A2', '#9D755D', '#EECA3B']
plt.rcParams.update({"font.family": "serif", "font.size": 10})

gt_all = json.load(open(os.path.join("playbyplay_GT", "playbyplay_val.json")))["events"]
taad_all = json.load(open("playbyplay_PRED_model40_audio.json"))["events"]
hpi_all = json.load(open(os.path.join("runs_hidden", "v7_33g_seed0", "hpi_val_events.json")))["events"]
ver = json.load(open(os.path.join("runs_hidden", "ver_seed0", "verified_taad_val.json")))
c = json.load(open(os.path.join("runs_hidden", "v7_33g_seed0", "exclusivity.json")))["chosen"]


def systems(k):
    """Predictions of the three systems for one half, each a list of (event, source)."""
    taad = [e for e in taad_all.get(k, []) if e[4] >= 0.70]
    tk, hk = apply_rule(taad, [e for e in hpi_all.get(k, []) if e[4] >= c["thr"]], c["rule"], c["w"])
    tv, hv = apply_rule([e for e in ver["events"].get(k, []) if e[4] >= ver["thr"]],
                        [e for e in hpi_all.get(k, []) if e[4] >= c["thr"]], c["rule"], c["w"])
    return {"TAAD": [(e, "T") for e in taad],
            "TAAD + HPI + rule": [(e, "T") for e in tk] + [(e, "H") for e in hk],
            "Final (+ Verifier)": [(e, "T") for e in tv] + [(e, "H") for e in hv]}


def correct(preds, gt):
    pred = [e[:4] + [2.0 if s == "T" else e[4]] for e, s in preds]
    used, ok = match_counts(pred, gt)
    return used, ok


# ---------------------------------------------------------------- 1. F1 progression (test)
def fig_f1():
    names = ["TAAD", "+ rule", "+ HPI", "+ calibration", "+ Event\nVerifier"]
    micro = [0.471, 0.490, 0.543, 0.587, 0.699]
    micro_sd = [0, 0, 0.001, 0.003, 0.006]
    macro = [0.338, 0.347, 0.400, 0.462, 0.549]
    macro_sd = [0, 0, 0.003, 0.010, 0.018]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b1 = ax.bar(x - 0.19, micro, 0.38, yerr=micro_sd, capsize=3, color="#4C78A8", label="Micro F1")
    b2 = ax.bar(x + 0.19, macro, 0.38, yerr=macro_sd, capsize=3, color="#F58518", label="Macro F1")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012, f"{r.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("F1 on the internal test set")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_f1_progression.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 2. timeline
def pick_window(k, length):
    gt = gt_all[k]
    s = systems(k)
    _, ok_t = correct(s["TAAD"], gt)
    used_f, ok_f = correct(s["Final (+ Verifier)"], gt)
    score = np.zeros(max(int(e[0]) for e in gt) + 1)
    for (e, _), o in zip(s["TAAD"], ok_t):
        if not o:
            score[int(e[0])] += 1          # TAAD false alarm
    for (e, src), o in zip(s["Final (+ Verifier)"], ok_f):
        if o and src == "H":
            score[int(e[0])] += 2          # hidden event found
        if not o:
            score[int(e[0])] -= 1          # final false alarm
    cs = np.concatenate([[0], np.cumsum(score)])
    best = max(range(0, len(score) - length, 25), key=lambda i: cs[i + length] - cs[i])
    return best


def fig_timeline(k="game_47_H1", seconds=30):
    gt = gt_all[k]
    s = systems(k)
    length = seconds * FPS
    t0 = pick_window(k, length)
    t1 = t0 + length
    rows = ["Ground truth"] + list(s)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    y = {r: len(rows) - 1 - i for i, r in enumerate(rows)}
    for g in gt:
        f = int(g[0])
        if t0 <= f < t1:
            hidden = int(g[4]) == 0
            ax.scatter((f - t0) / FPS, y["Ground truth"], s=70, marker="o",
                       facecolors="none" if hidden else CCOL[int(g[3])], edgecolors=CCOL[int(g[3])], lw=1.8, zorder=3)
            ax.text((f - t0) / FPS, y["Ground truth"] + 0.22, f"#{int(g[2])}", ha="center", fontsize=6.5)
    for name, preds in s.items():
        _, ok = correct(preds, gt)
        for (e, src), o in zip(preds, ok):
            f = int(e[0])
            if not t0 <= f < t1:
                continue
            xx = (f - t0) / FPS
            if o:
                ax.scatter(xx, y[name], s=70, marker="s" if src == "H" else "o", color=CCOL[int(e[3])], zorder=3)
            else:
                ax.scatter(xx, y[name], s=55, marker="x", color="#d62728", lw=1.8, zorder=3)
            if o:
                ax.text(xx, y[name] + 0.22, f"#{int(e[2])}", ha="center", fontsize=6.5)
    ax.set_yticks([y[r] for r in rows], rows)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-1, seconds + 1)
    ax.set_xlabel(f"Time (s) from {t0 / FPS / 60:.1f} min of {k.replace('_', ' ')}")
    for r in rows:
        ax.axhline(y[r], color="#eeeeee", lw=6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    hs = [plt.Line2D([], [], marker="o", ls="", color=CCOL[i], label=CLASS[i]) for i in range(1, 9)
          if any(t0 <= int(g[0]) < t1 and int(g[3]) == i for g in gt)]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="hidden (truth)"),
           plt.Line2D([], [], marker="s", ls="", color="k", label="found by HPI"),
           plt.Line2D([], [], marker="x", ls="", color="#d62728", label="wrong prediction")]
    ax.legend(handles=hs, ncol=min(len(hs), 5), fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.45, -0.28))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_timeline.png"), dpi=200)
    plt.close(fig)
    print(f"[timeline] {k} frames {t0}-{t1}")


# ---------------------------------------------------------------- 3. hidden-player example
def draw_pitch(ax):
    L, W = 105, 68
    ax.add_patch(Rectangle((0, 0), L, W, fc="#5a9e4b", ec="white", lw=1.2))
    ax.plot([L / 2, L / 2], [0, W], color="white", lw=1)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, fill=False, ec="white", lw=1))
    for x0, d in ((0, 1), (L, -1)):
        ax.add_patch(Rectangle((x0 if d == 1 else x0 - 16.5, W / 2 - 20.16), 16.5, 40.32, fill=False, ec="white", lw=1))
        ax.add_patch(Rectangle((x0 if d == 1 else x0 - 5.5, W / 2 - 9.16), 5.5, 18.32, fill=False, ec="white", lw=1))
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(W + 3, -3)
    ax.set_aspect("equal")
    ax.axis("off")


def hidden_candidates():
    out = []
    for k in gt_all:
        gt = gt_all[k]
        s = systems(k)["Final (+ Verifier)"]
        pred = [e[:4] + [2.0 if src == "T" else e[4]] for e, src in s]
        used, ok = match_counts(pred, gt)
        for (e, src), o in zip(s, ok):
            if o and src == "H":
                g = [g for g in gt if int(g[0]) - 12 <= e[0] <= int(g[0]) + 12 and int(g[2]) == int(e[2])
                     and int(g[3]) == int(e[3]) and int(g[1]) == int(e[1])]
                if g and int(g[0][4]) == 0:
                    out.append((e[4], k, g[0], e))
    return sorted(out, key=lambda z: -z[0])


def frame_at(game, f):
    cap = cv2.VideoCapture(os.path.join("videos", f"{game}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None


def fig_hidden(pick=None):
    cands = hidden_candidates()
    if pick is None:
        for i, (sc, k, g, e) in enumerate(cands[:15]):
            print(f"  [{i}] {k} frame {int(g[0])} {CLASS[int(g[3])]} team {int(g[1])} #{int(g[2])} HPI {sc:.2f}")
        pick = 0
    sc, k, g, e = cands[pick]
    f, team, shirt, cls = int(g[0]), int(g[1]), int(g[2]), int(g[3])
    game = k.rsplit("_", 1)[0]
    img = frame_at(game, f)
    with h5py.File(os.path.join("data", "val_tactical_data.h5"), "r") as h:
        d = h[k][:]
    rows = d[d[:, 0] == f]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    if img is not None:
        a1.imshow(img)
    a1.set_title(f"Broadcast frame ({CLASS[cls].lower()} by #{shirt} not on screen)", fontsize=9)
    a1.axis("off")
    draw_pitch(a2)
    for r in rows:
        x, y = r[5] * 105, r[6] * 68
        vis = not np.isnan(r[9])
        col = "#1f77b4" if int(r[2]) == 0 else "#d62728"
        actor = int(r[2]) == team and int(r[3]) == shirt
        a2.scatter(x, y, s=150 if actor else 45, marker="*" if actor else "o",
                   facecolors=("gold" if actor else (col if vis else "none")),
                   edgecolors="k" if actor else col, lw=1.3, zorder=4 if actor else 3)
        if actor:
            a2.annotate(f"HPI: {CLASS[cls]} by #{shirt}\nscore {sc:.2f}, correct", (x, y), xytext=(10, -18),
                        textcoords="offset points", fontsize=8, color="white", weight="bold",
                        bbox=dict(boxstyle="round", fc="black", alpha=0.7))
    a2.set_title("Tracking data at the same frame", fontsize=9)
    hs = [plt.Line2D([], [], marker="o", ls="", color="#1f77b4", label="visible"),
          plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#1f77b4", label="off screen"),
          plt.Line2D([], [], marker="*", ls="", mfc="gold", mec="k", ms=11, label="acting player")]
    a2.legend(handles=hs, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_hidden_example.png"), dpi=200)
    plt.close(fig)
    print(f"[hidden] {k} frame {f} {CLASS[cls]} #{shirt} HPI {sc:.2f}")


# ---------------------------------------------------------------- 4. highlight timeline
def fig_highlight(game=18):
    sys.path.insert(0, ".")
    from generate_highlights import load_excitement_signal
    tmp = os.path.join("runs_hidden", "fig_reel")
    os.makedirs(tmp, exist_ok=True)
    subprocess.run([sys.executable, "generate_highlights.py", "--events_json", "playbyplay_PRED_model40_audio.json",
                    "--game", str(game), "--checkpoint",
                    os.path.join("runs", "TAAD_audiofusion_kaggle", "checkpoints", "curr_model_40.pt"),
                    "--no_video", "--no_scoreboard", "--out_dir", tmp], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    clips = json.load(open(os.path.join(tmp, f"highlights_game_{game}_pred.json")))
    goals = json.load(open(os.path.join("highlights", f"scoreboard_game_{game}.json")))["goals"]
    x = load_excitement_signal(os.path.join("data", "audio_features.h5"), game)
    t = np.arange(len(x)) / FPS / 60
    k = 25 * 10
    xs = np.convolve(x, np.ones(k) / k, mode="same")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(t, xs, color="#4C78A8", lw=0.7, label="audio excitement (10 s mean)")
    for i, cl in enumerate(clips):
        ax.axvspan(cl["start_s"] / 60, cl["end_s"] / 60, color="#F58518", alpha=0.6, lw=0,
                   label="clip in the reel" if i == 0 else None)
    for i, gl in enumerate(goals):
        if not gl.get("overturned"):
            ax.axvline(gl["time"] / 60, color="#d62728", lw=1.4, ls="--", label="goal" if i == 0 else None)
    ax.set_xlabel("Match time in the video (min)")
    ax.set_ylabel("Excitement")
    ax.set_xlim(0, t[-1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_highlight_timeline.png"), dpi=200)
    plt.close(fig)
    got = sum(any(cl["start_s"] <= gl["time"] <= cl["end_s"] + 30 for cl in clips) for gl in goals)
    print(f"[highlight] game {game}: {len(clips)} clips, {sum(c_['end_s'] - c_['start_s'] for c_ in clips):.0f} s, "
          f"{len(goals)} goals")


def fig_shot(pick=1):
    """A correct TAAD shot with the broadcast audio around it: frame with boxes + crowd/commentary curves."""
    with h5py.File(os.path.join("data", "audio_features.h5"), "r") as af:
        audio = {g: (af[g]["crowd_energy"][:], af[g]["commentary_energy"][:]) for g in af}
    cands = []
    for k in gt_all:
        game = k.rsplit("_", 1)[0]
        if game not in audio:
            continue
        crowd, comm = audio[game]
        base = np.median(crowd)
        preds = [e for e in taad_all.get(k, []) if e[4] >= 0.70]
        used, ok = match_counts(preds, gt_all[k])
        for e, o in zip(preds, ok):
            f = int(e[0])
            if o and int(e[3]) == 5 and f + 125 < len(crowd):
                rise = crowd[f: f + 125].max() / max(crowd[max(f - 125, 0): f].mean(), 1e-9)
                cands.append((rise, k, e))
    cands.sort(key=lambda z: -z[0])
    for i, (r, k, e) in enumerate(cands[:6]):
        print(f"  [{i}] {k} frame {int(e[0])} #{int(e[2])} conf {e[4]:.2f} crowd rise x{r:.2f}")
    r, k, e = cands[pick]
    f, team, shirt = int(e[0]), int(e[1]), int(e[2])
    game = k.rsplit("_", 1)[0]
    img = frame_at(game, f)
    sx, sy = 1920 / img.shape[1], 1080 / img.shape[0]
    with h5py.File(os.path.join("data", "val_tactical_data.h5"), "r") as h:
        d = h[k][:]
    rows = d[(d[:, 0] == f) & ~np.isnan(d[:, 9])]
    crowd, comm = audio[game]
    norm = lambda x: np.clip((x - np.percentile(x, 5)) / (np.percentile(x, 99.5) - np.percentile(x, 5)), 0, 1)
    sm = lambda x: np.convolve(x, np.ones(13) / 13, mode="same")
    crowd, comm = norm(sm(crowd)), norm(sm(comm))
    fig = plt.figure(figsize=(6.4, 5.4))
    a1 = fig.add_axes([0.02, 0.36, 0.96, 0.62])
    a1.imshow(img)
    for rr in rows:
        x, y, w, hh = rr[9] / sx, rr[10] / sy, rr[11] / sx, rr[12] / sy
        actor = int(rr[2]) == team and int(rr[3]) == shirt
        a1.add_patch(Rectangle((x, y), w, hh, fill=False, lw=2.2 if actor else 0.8,
                               ec="#2ca02c" if actor else ("#1f77b4" if int(rr[2]) == 0 else "#d62728")))
        if actor:
            a1.text(x, y - 4, f"SHOT #{shirt}  {e[4]:.2f}  correct", color="white", fontsize=8, weight="bold",
                    bbox=dict(fc="#2ca02c", ec="none", pad=1.5))
    a1.axis("off")
    a2 = fig.add_axes([0.1, 0.08, 0.86, 0.22])
    t = np.arange(f - 150, f + 250)
    a2.plot((t - f) / FPS, crowd[t], color="#F58518", lw=1, label="crowd (20-500 Hz)")
    a2.plot((t - f) / FPS, comm[t], color="#4C78A8", lw=1, label="commentary (300-3400 Hz)")
    a2.axvline(0, color="#2ca02c", lw=1.5, ls="--")
    a2.text(-0.15, 1.0, "shot", color="#2ca02c", fontsize=8, ha="right")
    a2.set_xlabel("Time from the shot (s)")
    a2.set_ylabel("Energy")
    a2.set_ylim(0, 1.12)
    a2.spines[["top", "right"]].set_visible(False)
    a2.legend(fontsize=7.5, frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 0.98))
    fig.savefig(os.path.join(OUT, "fig_visible_detection.png"), dpi=200)
    plt.close(fig)
    print(f"[shot] {k} frame {f} #{shirt}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["f1", "timeline", "hidden", "shot"]
    if "shot" in which:
        fig_shot(int(os.environ.get("PICK", 1)))
    if "f1" in which:
        fig_f1()
    if "timeline" in which:
        fig_timeline()
    if "hidden" in which:
        fig_hidden(int(os.environ.get("PICK", 0)) if "PICK" in os.environ else None)
    if "highlight" in which:
        fig_highlight()
    if "visible" in which:
        shutil.copy(os.path.join("demo_outputs", "demo_game_18_H1_2286s_frame57328.png"),
                    os.path.join(OUT, "fig_visible_detection.png"))
    print("saved to", OUT)
