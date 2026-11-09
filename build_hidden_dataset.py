"""
Build the per-frame, per-player dataset for Hidden-Player event Inference (HPI).

For every halftime it stores, on a dense (frames x 26 role slots) grid:
  feats   (T, 26, F) float16  tracking features (always available, also during replays / off-screen)
  present (T, 26)    bool     player slot has a tracking row at this frame
  visible (T, 26)    bool     player has a bounding box (TAAD can see him)
  labels  (T, 26)    int8     action class at the event frame (0 = none), from the tactical CLS column
  shirt   (T, 26)    int16    shirt number of the player in that slot (-1 unknown), for exporting events
  audio   (T, 2)     float16  crowd / commentary energy (optional, from audio_features.h5)
  frame0  scalar              absolute frame number of row 0
TAAD(+audio) logits are NOT copied (they are large); the training script reads them from the
avg_logits_*.npy files or preds_*.h5 with the same absolute-frame indexing.

Slot index = left_to_right * 13 + (role_id - 1), identical to TAAD / DST.

Usage:
  python build_hidden_dataset.py --set val --out hidden_data/val.h5
  python build_hidden_dataset.py --set train --out hidden_data/train.h5 --halves_from_logits kaggle_preds/TAAD_predictions
  python build_hidden_dataset.py --set val --only game_18_H1 --out hidden_data/test_one.h5   (quick test)
"""
import argparse
import glob
import os

import h5py
import numpy as np

FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS = range(14)
M = 26
FEATURE_NAMES = ["x", "y", "vx", "vy", "speed", "accel", "turn", "d_team", "d_opp", "d_goal", "present", "visible"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=["train", "val"], required=True)
    p.add_argument("--data_root", default=".")
    p.add_argument("--audio_h5", default=os.path.join("data", "audio_features.h5"))
    p.add_argument("--halves_from_logits", default=None,
                   help="Only build halves that have avg_logits_<half>.npy in this folder (or keys of a preds_*.h5 file)")
    p.add_argument("--only", default=None, help="Comma-separated halftime keys (for testing)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def available_halves(path):
    if path is None:
        return None
    if path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            return set(f.keys())
    return {os.path.basename(p)[len("avg_logits_"):-4] for p in glob.glob(os.path.join(path, "avg_logits_*.npy"))}


def nearest_distances(pos, present, team_of_slot):
    """pos (T,26,2), present (T,26) -> nearest teammate / opponent distance per slot (T,26)."""
    T = pos.shape[0]
    d_team = np.full((T, M), 2.0, np.float32)
    d_opp = np.full((T, M), 2.0, np.float32)
    same = team_of_slot[:, None] == team_of_slot[None, :]
    eye = np.eye(M, dtype=bool)
    for s in range(0, T, 20000):  # chunk to bound memory (20000 x 26 x 26 floats)
        p = pos[s:s + 20000]
        pr = present[s:s + 20000]
        diff = p[:, :, None, :] - p[:, None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))  # (t,26,26)
        valid = pr[:, :, None] & pr[:, None, :] & ~eye[None]
        dist = np.where(valid, dist, np.inf)
        dt = np.where(same[None], dist, np.inf).min(-1)
        do = np.where(~same[None], dist, np.inf).min(-1)
        d_team[s:s + 20000] = np.where(np.isfinite(dt), dt, 2.0)
        d_opp[s:s + 20000] = np.where(np.isfinite(do), do, 2.0)
    return d_team, d_opp


def build_half(rows, audio=None):
    rows = rows[(rows[:, ROLE_ID] >= 1) & (rows[:, ROLE_ID] <= 13)]
    f0, f1 = int(rows[:, FRAME].min()), int(rows[:, FRAME].max())
    T = f1 - f0 + 1
    t = rows[:, FRAME].astype(np.int64) - f0
    slot = (rows[:, LEFT_TO_RIGHT].astype(np.int64) * 13 + rows[:, ROLE_ID].astype(np.int64) - 1)

    present = np.zeros((T, M), bool)
    visible = np.zeros((T, M), bool)
    labels = np.zeros((T, M), np.int8)
    shirt = np.full((T, M), -1, np.int16)
    pos = np.zeros((T, M, 2), np.float32)
    vel = np.zeros((T, M, 2), np.float32)
    present[t, slot] = True
    visible[t, slot] = ~np.isnan(rows[:, ROI_X])
    labels[t, slot] = rows[:, CLS].astype(np.int8)
    shirt[t, slot] = np.nan_to_num(rows[:, SHIRT_NUMBER], nan=-1).astype(np.int16)
    pos[t, slot, 0], pos[t, slot, 1] = rows[:, X_POS], rows[:, Y_POS]
    vel[t, slot, 0], vel[t, slot, 1] = rows[:, X_SPEED], rows[:, Y_SPEED]
    pos = np.nan_to_num(pos)
    vel = np.nan_to_num(vel)

    speed = np.sqrt((vel ** 2).sum(-1))
    accel = np.zeros_like(speed)
    accel[1:] = speed[1:] - speed[:-1]
    ang = np.arctan2(vel[..., 1], vel[..., 0])
    turn = np.zeros_like(speed)
    turn[1:] = np.abs(np.angle(np.exp(1j * (ang[1:] - ang[:-1]))))
    turn = np.where(speed > 1e-3, turn, 0.0)
    both = present.copy()
    both[1:] &= present[:-1]
    accel, turn = accel * both, turn * both  # no derivative across gaps

    team_of_slot = np.arange(M) // 13
    d_team, d_opp = nearest_distances(pos, present, team_of_slot)
    # team 0 plays left-to-right (attacks x = 1), team 1 attacks x = 0 (checked: team 0 mean x < team 1 mean x)
    goal_x = np.where(team_of_slot == 0, 1.0, 0.0)[None, :]
    d_goal = np.sqrt((pos[..., 0] - goal_x) ** 2 + (pos[..., 1] - 0.5) ** 2)

    feats = np.stack([pos[..., 0], pos[..., 1], vel[..., 0], vel[..., 1], speed, accel, turn,
                      d_team, d_opp, d_goal, present.astype(np.float32), visible.astype(np.float32)], -1)
    feats[~present] = 0.0
    feats[..., FEATURE_NAMES.index("present")] = present  # keep flags after zeroing
    feats[..., FEATURE_NAMES.index("visible")] = visible

    out = dict(feats=feats.astype(np.float16), present=present, visible=visible, labels=labels, shirt=shirt, frame0=f0)
    if audio is not None:
        crowd, comm = audio
        a = np.zeros((T, 2), np.float32)
        idx = np.arange(f0, f1 + 1)
        ok = idx < len(crowd)
        a[ok, 0], a[ok, 1] = crowd[idx[ok]], comm[idx[ok]]
        out["audio"] = a.astype(np.float16)
    return out


def main():
    a = parse_args()
    tac_path = os.path.join(a.data_root, "data", f"{a.set}_tactical_data.h5")
    keep = available_halves(a.halves_from_logits)
    only = set(a.only.split(",")) if a.only else None
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    audio_f = h5py.File(a.audio_h5, "r") if os.path.exists(a.audio_h5) else None
    totals = np.zeros(9, np.int64)
    hidden_totals = np.zeros(9, np.int64)
    with h5py.File(tac_path, "r") as tac, h5py.File(a.out, "w") as out:
        out.attrs["feature_names"] = FEATURE_NAMES
        for k in tac.keys():
            if (keep is not None and k not in keep) or (only is not None and k not in only):
                continue
            game = k.rsplit("_", 1)[0]
            audio = None
            if audio_f is not None and game in audio_f:
                audio = (audio_f[game]["crowd_energy"][:], audio_f[game]["commentary_energy"][:])
            h = build_half(tac[k][:], audio)
            g = out.create_group(k)
            for name, arr in h.items():
                if name == "frame0":
                    g.attrs["frame0"] = arr
                else:
                    g.create_dataset(name, data=arr, compression="lzf")
            lab, vis = h["labels"], h["visible"]
            ev = np.bincount(lab[lab > 0], minlength=9)
            hev = np.bincount(lab[(lab > 0) & ~vis], minlength=9)
            totals += ev
            hidden_totals += hev
            print(f"{k}: frames {lab.shape[0]}, events {ev.sum()}, hidden {hev.sum()} ({100 * hev.sum() / max(ev.sum(), 1):.1f}%), "
                  f"audio {'yes' if 'audio' in h else 'no'}", flush=True)
    names = ['bg', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']
    print("\nall events   :", {names[c]: int(totals[c]) for c in range(1, 9)})
    print("hidden events:", {names[c]: int(hidden_totals[c]) for c in range(1, 9)})
    print("saved", a.out)


if __name__ == "__main__":
    main()
