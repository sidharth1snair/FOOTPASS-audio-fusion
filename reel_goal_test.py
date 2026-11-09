"""
Does loudness make the reel catch more GOALS? The scoreboard is used only to define the goals, never to build the reel.

Goal    : last ground-truth shot / header in the 60 s before an on-screen score change (not overturned).
Games   : matches with detected goals AND TAAD predictions. Game 18 is val; the others are train games,
          where TAAD predictions are optimistic.
Audio   : none | energy + tau | energy, tau = 0 | energy from the WRONG match | energy shifted 60 s.
Reel lengths and audio_weight = 2 are fixed before looking at the results.

Usage:
  python reel_goal_test.py
"""
import json
import os
import shutil

import numpy as np

from generate_highlights import load_excitement_signal, load_tau_seconds
from highlight_comparison import reel
from utils.metric_utils import post_processing_TAAD_preds_and_export

GAMES = [8, 10, 12, 13, 15, 17, 18, 23]
FPS = 25
CHECKPOINT = os.path.join("runs", "TAAD_audiofusion_kaggle", "checkpoints", "curr_model_40.pt")
EVENTS = os.path.join("runs_hidden", "taad_reel_games_events.json")


def taad_events():
    if not os.path.exists(EVENTS):
        tmp = os.path.join("runs_hidden", "_taad_reel_tmp")
        os.makedirs(os.path.join(tmp, "TAAD_predictions"), exist_ok=True)
        for g in GAMES:
            if g == 18:
                continue
            for h in ("H1", "H2"):
                shutil.copy(os.path.join("kaggle_preds", "TAAD_predictions", f"avg_logits_game_{g}_{h}.npy"),
                            os.path.join(tmp, "TAAD_predictions"))
        post_processing_TAAD_preds_and_export(nms_window=25, root_path=tmp, save_path=EVENTS,
                                              h5file_path=os.path.join("data", "train_tactical_data.h5"))
        shutil.rmtree(tmp)
    ev = json.load(open(EVENTS))["events"]
    ev.update({k: v for k, v in json.load(open("playbyplay_PRED_model40_audio.json"))["events"].items() if k.startswith("game_18_")})
    return ev


def main():
    rng = np.random.default_rng(0)
    gt = {**json.load(open(os.path.join("playbyplay_GT", "playbyplay_train.json")))["events"],
          **json.load(open(os.path.join("playbyplay_GT", "playbyplay_val.json")))["events"]}
    ev = taad_events()
    goals, rows = {}, {}
    for g in GAMES:
        keys = (f"game_{g}_H1", f"game_{g}_H2")
        shots = np.array(sorted(e[0] for k in keys for e in gt[k] if int(e[3]) in (5, 6)))
        out = []
        for x in json.load(open(os.path.join("highlights", f"scoreboard_game_{g}.json")))["goals"]:
            T = x["time"] * FPS
            c = shots[(shots >= T - 60 * FPS) & (shots <= T)]
            if not x["overturned"] and c.size:
                out.append(float(c[-1]) / FPS)
        goals[g] = sorted(set(out))
        rows[g] = sorted((int(e[0]), int(e[1]), int(e[2]), int(e[3]), float(e[4])) for k in keys for e in ev[k] if e[4] >= 0.70)
    print("goals per game:", {g: len(v) for g, v in goals.items()}, "| total", sum(len(v) for v in goals.values()))

    tau = load_tau_seconds(CHECKPOINT, FPS)
    zero = np.zeros_like(tau)
    energy = {g: load_excitement_signal(os.path.join("data", "audio_features.h5"), g) for g in GAMES}
    audio = {"no audio": lambda g: (None, tau, 0.0),
             "energy + tau": lambda g: (energy[g], tau, 2.0),
             "energy, tau = 0": lambda g: (energy[g], zero, 2.0),
             "energy, WRONG match": lambda g: (energy[GAMES[(GAMES.index(g) + 1) % len(GAMES)]], tau, 2.0),
             "energy, shifted 60 s": lambda g: (np.roll(energy[g], 60 * FPS), tau, 2.0)}
    res, flags = {}, {}
    print(f"\n{'reel':>6s} {'audio':22s} {'goals in reel':>15s} {'seconds':>8s}")
    for target in (120.0, 180.0, 300.0):
        for name, f in audio.items():
            cov, secs = [], 0.0
            for g in GAMES:
                e, t, w = f(g)
                clips = reel(rows[g], t, e, w, target)
                secs += sum(c["end"] - c["start"] for c in clips)
                cov += [any(c["start"] <= s <= c["end"] for c in clips) for s in goals[g]]
            cov = np.array(cov)
            flags[target, name] = cov
            res[f"{int(target)} s | {name}"] = dict(covered=int(cov.sum()), goals=len(cov), seconds=secs)
            print(f"{int(target):5d}s {name:22s} {cov.sum():5d}/{len(cov)} ({100 * cov.mean():4.1f}%) {secs:8.0f}")
    print("\nPaired bootstrap over goals (180 s reel, difference in goals covered, 95% CI):")
    res["bootstrap"] = {}
    for other in ("no audio", "energy, WRONG match", "energy, shifted 60 s", "energy, tau = 0"):
        d = flags[180.0, "energy + tau"].astype(float) - flags[180.0, other].astype(float)
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        ci = np.percentile(b, [2.5, 97.5])
        print(f"  energy + tau vs {other:22s} {100 * d.mean():+5.1f} pts  CI [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]")
        res["bootstrap"][other] = dict(diff=float(d.mean()), ci=ci.tolist())
    json.dump(res, open(os.path.join("runs_hidden", "reel_goal_test.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
