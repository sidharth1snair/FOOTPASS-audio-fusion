"""
Demo video for the project website, made from tracking data only (no broadcast footage,
which the SoccerNet NDA does not allow to redistribute).

Top: top-down pitch with every player (filled = visible on the broadcast, hollow = off screen).
     Predictions of the final system (TAAD + Event Verifier + HPI + rule) appear on the acting player:
     green = correct, red = wrong; a gold star marks a hidden-player event found by HPI.
Bottom: event timeline revealed as the clip plays - ground truth, TAAD alone, final system.

Usage:
  python make_demo_video.py [--game game_47_H1 --start 9350 --seconds 30]
"""
import argparse
import os

import av
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_thesis_figures import CCOL, CLASS, correct, draw_pitch, gt_all, systems

FPS = 25


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="game_47_H1")
    p.add_argument("--start", type=int, default=9350)
    p.add_argument("--seconds", type=int, default=30)
    p.add_argument("--out", default=os.path.join("website", "assets", "demo.mp4"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0, t1 = a.start, a.start + a.seconds * FPS
    gt = gt_all[a.game]
    s = systems(a.game)
    rows = {"Ground truth": [(g, None, True) for g in gt]}
    for name, key in (("TAAD alone", "TAAD"), ("Final system", "Final (+ Verifier)")):
        _, ok = correct(s[key], gt)
        rows[name] = [(e, src, o) for (e, src), o in zip(s[key], ok)]
    with h5py.File(os.path.join("data", "val_tactical_data.h5"), "r") as h:
        d = h[a.game][:]
    d = d[(d[:, 0] >= t0) & (d[:, 0] < t1)]
    by_frame = {f: d[d[:, 0] == f] for f in np.unique(d[:, 0]).astype(int)}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0f1720")
    ap = fig.add_axes([0.08, 0.30, 0.84, 0.66])
    at = fig.add_axes([0.14, 0.06, 0.80, 0.19], facecolor="#0f1720")
    container = av.open(a.out, "w")
    st = container.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = 1280, 720, "yuv420p"
    st.options = {"crf": "23"}
    hold = int(1.2 * FPS)
    final = rows["Final system"]
    hidden_found = {int(e[0]) for e, src, o in final if o and src == "H"}

    for f in range(t0, t1):
        ap.clear(); at.clear()
        draw_pitch(ap)
        ap.set_facecolor("#0f1720")
        fr = by_frame.get(f, by_frame[min(by_frame, key=lambda k: abs(k - f))])
        for r in fr:
            x, y = r[5] * 105, r[6] * 68
            vis = not np.isnan(r[9])
            col = "#4aa3ff" if int(r[2]) == 0 else "#ff5a5a"
            ap.scatter(x, y, s=110, facecolors=col if vis else "none", edgecolors=col, lw=2, zorder=3)
            ap.text(x, y + 0.15, f"{int(r[3])}", ha="center", va="center", fontsize=6.5,
                    color="white" if vis else col, zorder=4, weight="bold")
        for e, src, o in final:
            if e[0] <= f < e[0] + hold:
                hit = [r for r in fr if int(r[2]) == int(e[1]) and int(r[3]) == int(e[2])]
                if not hit:
                    continue
                x, y = hit[0][5] * 105, hit[0][6] * 68
                hid = src == "H"
                colr = "#ffd23f" if hid else ("#2ecc71" if o else "#e74c3c")
                if hid:
                    ap.scatter(x, y, s=600, marker="*", color=colr, edgecolors="k", zorder=5)
                else:
                    ap.scatter(x, y, s=420, facecolors="none", edgecolors=colr, lw=3, zorder=5)
                label = (f"HPI: {CLASS[int(e[3])]} by #{int(e[2])} (player off screen)" if hid
                         else f"{CLASS[int(e[3])]} #{int(e[2])} {'correct' if o else 'wrong'}")
                ap.annotate(label, (x, y), xytext=(14, 14), textcoords="offset points", fontsize=11,
                            color="black", weight="bold", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.3", fc=colr, ec="none"))
        vis_n = int(sum(not np.isnan(r[9]) for r in fr))
        ap.set_title(f"{a.game.replace('_', ' ')}   {f / FPS / 60:5.2f} min   |   players on screen: {vis_n} of {len(fr)}",
                     color="white", fontsize=13, pad=6)
        # timeline
        ys = {"Ground truth": 2, "TAAD alone": 1, "Final system": 0}
        tnow = (f - t0) / FPS
        for name, evs in rows.items():
            at.axhline(ys[name], color="#24313d", lw=8, zorder=0)
            for e, src, o in evs:
                tt = (int(e[0]) - t0) / FPS
                if not 0 <= tt <= tnow:
                    continue
                if name == "Ground truth":
                    hidden = int(e[4]) == 0
                    at.scatter(tt, ys[name], s=45, facecolors="none" if hidden else CCOL[int(e[3])],
                               edgecolors=CCOL[int(e[3])], lw=1.6, zorder=3)
                elif o:
                    at.scatter(tt, ys[name], s=55 if src == "H" else 45, marker="*" if src == "H" else "o",
                               color="#ffd23f" if src == "H" else CCOL[int(e[3])], zorder=3)
                else:
                    at.scatter(tt, ys[name], s=40, marker="x", color="#e74c3c", lw=1.8, zorder=3)
        at.axvline(tnow, color="white", lw=1)
        at.set_xlim(-0.5, a.seconds + 0.5)
        at.set_ylim(-0.6, 2.6)
        at.set_yticks([2, 1, 0], list(ys))
        at.tick_params(colors="white", labelsize=10)
        at.set_xlabel("seconds", color="white", fontsize=9)
        for sp in at.spines.values():
            sp.set_visible(False)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for pkt in st.encode(vf):
            container.mux(pkt)
    for pkt in st.encode():
        container.mux(pkt)
    container.close()
    print("wrote", a.out, "| hidden events found by HPI in clip:", len([h for h in hidden_found if t0 <= h < t1]))


if __name__ == "__main__":
    main()
