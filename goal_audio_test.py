"""
Does audio mark the best moments? Goals are compared with ordinary shots / headers.

Goals   : on-screen score changes (generate_highlights.scan_scoreboard), kept only for games where the score
          is on screen at least --min_visible of the time, and not overturned.
Goal event: the last ground-truth shot / header in the 60 s before the score change.
Audio   : peak excitement (commentary + crowd energy, robust z-score per match) in the 10 s after the event.
Result  : percentile of each goal among the same match's shots / headers (50% = no effect), 95% CI by resampling games.
Control : same frames in ANOTHER match's audio. A real effect must disappear.
The 6 games frozen as the HPI test set (hidden_data/split_v2.json) are left out.

Usage:
  python goal_audio_test.py
"""
import argparse
import glob
import json
import os
import re

import h5py
import numpy as np

from generate_highlights import excitement_from_energy, parse_boxes, scan_scoreboard

FPS, SHOT, HEADER = 25, 5, 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", default="videos")
    p.add_argument("--audio_h5", default=os.path.join("data", "audio_features.h5"))
    p.add_argument("--min_visible", type=float, default=0.6)
    p.add_argument("--after_s", type=float, default=10.0)
    p.add_argument("--lookback_s", type=float, default=60.0)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", default=os.path.join("runs_hidden", "goal_audio_test.json"))
    return p.parse_args()


def main():
    a = parse_args()
    rng = np.random.default_rng(0)
    test = set(json.load(open(os.path.join("hidden_data", "split_v2.json")))["hpi_test_games"])
    gt = {**json.load(open(os.path.join("playbyplay_GT", "playbyplay_train.json")))["events"],
          **json.load(open(os.path.join("playbyplay_GT", "playbyplay_val.json")))["events"]}
    sb_box, name_boxes = parse_boxes("318,66,90,38")[0], parse_boxes("140,66,160,38;455,66,130,38")
    af = h5py.File(a.audio_h5, "r")
    vids = {int(re.findall(r"game_(\d+)", v)[0]): v for v in glob.glob(os.path.join(a.video_dir, "game_*.mp4"))}
    games = sorted(g for g in vids if g not in test and f"game_{g}" in af and f"game_{g}_H1" in gt)
    print(len(games), "games with video, audio and labels (HPI test games left out)")

    exc, data, skipped = {}, {}, []
    for g in games:
        r = scan_scoreboard(vids[g], sb_box, name_boxes, os.path.join("highlights", f"scoreboard_game_{g}.json"),
                            change_thresh=6.0, visible_thresh=25.0, persist=10)
        if r["visible_fraction"] < a.min_visible:
            skipped.append(g)
            continue
        ev = sorted(e[0] for k in (f"game_{g}_H1", f"game_{g}_H2") for e in gt[k] if int(e[3]) in (SHOT, HEADER))
        ev = np.array(ev)
        is_goal = np.zeros(len(ev), bool)
        lost = 0
        for x in r["goals"]:
            if x["overturned"]:
                continue
            T = x["time"] * FPS
            c = np.nonzero((ev >= T - a.lookback_s * FPS) & (ev <= T))[0]
            if c.size:
                is_goal[c[-1]] = True
            else:
                lost += 1
        exc[g] = excitement_from_energy(af[f"game_{g}"]["commentary_energy"][:], af[f"game_{g}"]["crowd_energy"][:])
        data[g] = dict(ev=ev, goal=is_goal, lost=lost)
    # games with no detected goal are detector failures (other score-box layout), not goalless matches, and their
    # audio level differs, so goals are only compared with shots / headers of the SAME match
    kept = sorted(g for g in data if data[g]["goal"].sum() > 0)
    print(f"kept {len(kept)} games, skipped {len(skipped)} (score rarely on screen)")
    print("goals per kept game:", {g: int(data[g]["goal"].sum()) for g in kept})
    print("score changes with no shot / header in the 60 s before:", sum(data[g]["lost"] for g in kept))

    w = int(a.after_s * FPS)
    peak = lambda e, f: float(e[int(f):int(f) + w].max()) if int(f) + 1 < len(e) else 0.0
    for i, g in enumerate(kept):
        o = exc[kept[(i + 1) % len(kept)]]
        data[g]["real"] = np.array([peak(exc[g], f) for f in data[g]["ev"]])
        data[g]["wrong"] = np.array([peak(o, f) for f in data[g]["ev"]])

    def goal_percentiles(key):
        """Per goal: share of the same match's other shots / headers that are quieter (0.5 = no effect)."""
        out = {}
        for g in kept:
            s, y = data[g][key], data[g]["goal"]
            out[g] = [float(np.mean(s[~y] < v)) for v in s[y]]
        return out

    res = {}
    for key, name in [("real", "correct audio"), ("wrong", "WRONG-match audio")]:
        pc = goal_percentiles(key)
        allp = np.concatenate([pc[g] for g in kept])
        b = [np.concatenate([pc[g] for g in rng.choice(kept, len(kept))]).mean() for _ in range(a.n_boot)]
        ci = np.percentile(b, [2.5, 97.5])
        res[key] = dict(mean_percentile=float(allp.mean()), ci=ci.tolist(), goals=int(allp.size), top10=float(np.mean(allp >= 0.9)),
                        top25=float(np.mean(allp >= 0.75)))
        print(f"{name:18s} goals {allp.size} | louder than {100 * allp.mean():.1f}% of same-match shots/headers (50% = no effect), "
              f"95% CI [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] | in loudest 10%: {100 * res[key]['top10']:.0f}% | in loudest 25%: {100 * res[key]['top25']:.0f}%")
    res["kept_games"], res["skipped_games"] = kept, skipped
    json.dump(res, open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
