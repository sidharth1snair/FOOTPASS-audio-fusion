"""
Render a short demo video (and PNG snapshots) of the TAAD + audio model's detections.

For a window of one halftime it draws:
  - every visible player's bounding box (thin, coloured by team side),
  - each predicted event on the predicted player for ~1 s: action, shirt number, confidence,
    green = correct (right action, player and time within +/- delta frames), red = false alarm,
  - ground-truth events the model missed (grey "missed"),
  - the crowd and commentary audio energy around the current frame.

Example:
  python annotate_events.py --game game_18_H1 --pred playbyplay_PRED_model40_audio.json
"""
import argparse
import os

import av
import cv2
import h5py
import numpy as np
import json

CLASS_NAMES = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']
FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14)
SX, SY = 1920 / 640, 1080 / 352  # tactical boxes are full-HD, val videos are 640x352


def parse_args():
    p = argparse.ArgumentParser(description="Annotated demo video of predicted events")
    p.add_argument("--game", default="game_18_H1", help="Halftime key, e.g. game_18_H1")
    p.add_argument("--pred", default="playbyplay_PRED_model40_audio.json")
    p.add_argument("--gt", default=os.path.join("playbyplay_GT", "playbyplay_val.json"))
    p.add_argument("--tactical", default=os.path.join("data", "val_tactical_data.h5"))
    p.add_argument("--audio_h5", default=os.path.join("data", "audio_features.h5"))
    p.add_argument("--video_dir", default="videos")
    p.add_argument("--thr", type=float, default=0.70, help="Confidence threshold for showing a prediction")
    p.add_argument("--delta", type=int, default=12, help="Matching tolerance in frames (same as evaluation.py)")
    p.add_argument("--start_sec", type=float, default=None, help="Window start in seconds from the video start (auto if omitted)")
    p.add_argument("--seconds", type=float, default=30.0, help="Window length")
    p.add_argument("--focus_class", default=None, help="Centre the window on the most confident correct detection of this class, e.g. shot")
    p.add_argument("--scale", type=int, default=2, help="Upscale factor for readability")
    p.add_argument("--out_dir", default="demo_outputs")
    return p.parse_args()


def match_events(preds, gts, delta):
    """Greedy matching as in evaluation.py: same (team, shirt, class), closest frame within delta, highest score first."""
    used = [False] * len(gts)
    pred_ok = []
    for pe in sorted(range(len(preds)), key=lambda i: -preds[i][4]):
        f, team, shirt, c, _ = preds[pe]
        best, best_dt = None, delta + 1
        for gi, g in enumerate(gts):
            if used[gi] or (int(g[1]), int(g[2]), int(g[3])) != (int(team), int(shirt), int(c)):
                continue
            dt = abs(int(g[0]) - int(f))
            if dt <= delta and dt < best_dt:
                best, best_dt = gi, dt
        if best is not None:
            used[best] = True
        pred_ok.append((pe, best is not None))
    ok = dict(pred_ok)
    return [ok[i] for i in range(len(preds))], used


def pick_window(preds, ok, frames_len, lo, hi):
    """Window with the most correct detections, rare classes (cross/shot/header/tackle/block) weighted x5."""
    score = np.zeros(hi + 1)
    for p, good in zip(preds, ok):
        if good:
            score[int(p[0])] += 5 if int(p[3]) >= 3 else 1
    csum = np.concatenate([[0], np.cumsum(score)])
    best_s, best_v = lo, -1
    for s in range(lo, max(lo + 1, hi - frames_len), 25):
        v = csum[min(s + frames_len, hi + 1)] - csum[s]
        if v > best_v:
            best_s, best_v = s, v
    return best_s


def team_colour(ltr):
    return (255, 160, 60) if int(ltr) == 0 else (60, 60, 255)  # BGR: blue-ish / red-ish


def draw_label(img, text, x, y, bg, scale):
    fs = 0.45 * scale
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), bg, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    game_id = a.game.rsplit("_", 1)[0]

    preds_all = json.load(open(a.pred))["events"][a.game]
    gts = json.load(open(a.gt))["events"][a.game]
    preds = [p for p in preds_all if p[4] >= a.thr]
    ok, gt_found = match_events(preds, gts, a.delta)
    tp = sum(ok)
    print(f"[INFO] {a.game}: {len(preds)} predictions >= {a.thr}, {tp} correct, {len(gts)} GT events, {len(gts) - sum(gt_found)} missed")

    with h5py.File(a.tactical, "r") as f:
        data = f[a.game][:]
    data = data[~np.isnan(data[:, ROI_X])]
    lo, hi = int(data[:, FRAME].min()), int(data[:, FRAME].max())

    container = av.open(os.path.join(a.video_dir, f"{game_id}.mp4"))
    vs = container.streams.video[0]
    fps = float(vs.average_rate)
    n_frames = int(round(a.seconds * fps))
    if a.start_sec is not None:
        start = int(round(a.start_sec * fps))
    elif a.focus_class:
        c = CLASS_NAMES.index(a.focus_class)
        hits = [p for p, good in zip(preds, ok) if good and int(p[3]) == c]
        assert hits, f"no correct '{a.focus_class}' detection in {a.game} at thr {a.thr}"
        best = max(hits, key=lambda p: p[4])
        start = max(int(best[0]) - int(0.6 * n_frames), 0)
    else:
        start = pick_window(preds, ok, n_frames, lo, hi)
    end = start + n_frames
    print(f"[INFO] window: frames {start}-{end} ({start / fps / 60:.2f}-{end / fps / 60:.2f} min of the video)")

    with h5py.File(a.audio_h5, "r") as f:
        crowd = f[game_id]["crowd_energy"][:]
        comm = f[game_id]["commentary_energy"][:]
    def norm(x):
        lo_, hi_ = np.percentile(x, 5), np.percentile(x, 99.5)
        return np.clip((x - lo_) / max(hi_ - lo_, 1e-9), 0, 1)
    crowd, comm = norm(crowd), norm(comm)

    boxes_by_frame = {}
    sel = data[(data[:, FRAME] >= start) & (data[:, FRAME] < end)]
    for r in sel:
        boxes_by_frame.setdefault(int(r[FRAME]), []).append(r)
    show_frames = int(fps)  # keep a label on screen for ~1 s
    win_preds = [(p, g) for p, g in zip(preds, ok) if start - show_frames <= p[0] < end]
    missed = [g for g, used in zip(gts, gt_found) if not used and start - show_frames <= g[0] < end]

    s = a.scale
    W, H = vs.width * s, vs.height * s
    PANEL = 70 * s
    out_name = f"demo_{a.game}_{int(start / fps)}s"
    out_path = os.path.join(a.out_dir, out_name + ".mp4")
    out = av.open(out_path, "w")
    ostream = out.add_stream("libx264", rate=int(round(fps)))
    ostream.width, ostream.height, ostream.pix_fmt = W, H + PANEL, "yuv420p"
    ostream.options = {"crf": "20"}

    container.seek(int(max(start - 2 * fps, 0) / fps / vs.time_base), stream=vs)
    snapshots, written = [], 0
    for frame in container.decode(vs):
        fidx = int(round(frame.pts * vs.time_base * fps))
        if fidx < start:
            continue
        if fidx >= end:
            break
        img = cv2.resize(frame.to_ndarray(format="bgr24"), (W, H), interpolation=cv2.INTER_CUBIC)

        player_box = {}
        for r in boxes_by_frame.get(fidx, []):
            x, y = int(r[ROI_X] / SX * s), int(r[ROI_Y] / SY * s)
            w, h = int(r[ROI_WIDTH] / SX * s), int(r[ROI_HEIGHT] / SY * s)
            player_box[(int(r[LEFT_TO_RIGHT]), int(r[SHIRT_NUMBER]))] = (x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), team_colour(r[LEFT_TO_RIGHT]), 1)

        active = False
        for p, good in win_preds:
            if p[0] <= fidx < p[0] + show_frames:
                active = True
                key = (int(p[1]), int(p[2]))
                colour = (40, 170, 40) if good else (40, 40, 220)
                text = f"{CLASS_NAMES[int(p[3])].upper()} #{int(p[2])} {p[4]:.2f} {'OK' if good else 'FALSE'}"
                if key in player_box:
                    x, y, w, h = player_box[key]
                    cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), colour, 2)
                    draw_label(img, text, x, y - 4, colour, s)
                else:  # player not visible in this frame (no bounding box)
                    draw_label(img, text + " (off-screen)", 8, 28 * s, colour, s)
        for g in missed:
            if g[0] <= fidx < g[0] + show_frames:
                draw_label(img, f"missed: {CLASS_NAMES[int(g[3])]} #{int(g[2])}", 8, H - 8, (90, 90, 90), s)

        # audio panel: last 10 s and next 2 s of crowd / commentary energy
        panel = np.full((PANEL, W, 3), 25, np.uint8)
        a0, a1 = fidx - int(10 * fps), fidx + int(2 * fps)
        xs = np.linspace(0, W - 1, a1 - a0).astype(int)
        for sig, col, row in [(crowd, (60, 200, 255), 0), (comm, (255, 200, 80), 1)]:
            seg = np.array([sig[i] if 0 <= i < len(sig) else 0 for i in range(a0, a1)])
            top, hgt = 6 * s + row * PANEL // 2, PANEL // 2 - 10 * s
            pts = np.stack([xs, top + hgt - (seg * hgt).astype(int)], 1)
            cv2.polylines(panel, [pts.reshape(-1, 1, 2)], False, col, 1 * s, cv2.LINE_AA)
            cv2.putText(panel, ["crowd", "commentary"][row], (6, top + 12 * s // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, col, 1, cv2.LINE_AA)
        now_x = int(W * 10 / 12)
        cv2.line(panel, (now_x, 0), (now_x, PANEL), (255, 255, 255), 1)
        canvas = np.vstack([img, panel])
        cv2.putText(canvas, f"{a.game}  {fidx / fps / 60:05.2f} min  |  TAAD + audio (learned delay)", (W - 330 * s, H + PANEL - 6 * s),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, (255, 255, 255), 1, cv2.LINE_AA)

        vf = av.VideoFrame.from_ndarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), format="rgb24")
        vf.pts = written
        for pkt in ostream.encode(vf):
            out.mux(pkt)
        written += 1
        if active and (not snapshots or fidx - snapshots[-1] > 3 * fps) and len(snapshots) < 4:
            cv2.imwrite(os.path.join(a.out_dir, f"{out_name}_frame{fidx}.png"), canvas)
            snapshots.append(fidx)

    for pkt in ostream.encode():
        out.mux(pkt)
    out.close()
    container.close()
    n_ok = sum(1 for p, g in win_preds if g and start <= p[0] < end)
    print(f"[INFO] in window: {len([1 for p, _ in win_preds if start <= p[0] < end])} detections ({n_ok} correct), {len([g for g in missed if start <= g[0] < end])} missed")
    print(f"[INFO] wrote {out_path} ({written} frames) and {len(snapshots)} snapshots")


if __name__ == "__main__":
    main()
