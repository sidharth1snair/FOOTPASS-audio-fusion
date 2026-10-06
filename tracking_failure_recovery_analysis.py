"""
Tracking-failure recovery analysis: for each GT event that model_20 (video-only)
missed, check whether model_40 (video+audio) recovered it -- split by whether the
GT event had a visible player bounding box (bbox=1) or not (bbox=0, i.e. a
tracking failure, per FOOTPASS's own ~20%-missed-due-to-tracking finding).
"""

import json

CLASS_NAMES = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']
C = len(CLASS_NAMES)
DELTA = 12
CONF_THRESH = 0.70  # best threshold for both models, per earlier sweep

GT_PATH = "playbyplay_GT/playbyplay_val.json"
PRED20_PATH = "playbyplay_PRED_model20/playbyplay_TAAD_val.json"
PRED40_PATH = "playbyplay_PRED_model40_audio.json"


def match_predictions(gt, pred, conf_thresh):
    """
    Returns: dict game_key -> dict (team,shirt,class,frame,bbox,replay) -> matched (bool)
    i.e. for every GT event, whether this prediction set matched it.
    """
    gt_keys = [str(k) for k in gt.get("keys", [])]
    pred_keys = [str(k) for k in pred.get("keys", [])]
    keys_to_eval = sorted(set(gt_keys).intersection(pred_keys))

    results = {}  # game -> list of (team, shirt, cls, frame, bbox, replay, matched)

    for k in keys_to_eval:
        gt_list = gt["events"].get(k, [])
        pred_list = pred["events"].get(k, [])

        gt_events = {}  # (team,shirt,cls) -> list of (frame, bbox, replay)
        for e in gt_list:
            if len(e) < 4:
                continue
            t_gt, team, shirt, cls = int(e[0]), int(e[1]), int(e[2]), int(e[3])
            bbox = int(e[4]) if len(e) > 4 else 0
            replay = int(e[5]) if len(e) > 5 else 0
            if cls <= 0 or cls >= C:
                continue
            gt_events.setdefault((team, shirt, cls), []).append((t_gt, bbox, replay))

        for key_grp in gt_events:
            gt_events[key_grp].sort(key=lambda x: x[0])

        dets = []
        for e in pred_list:
            if len(e) < 4:
                continue
            t_det, team, shirt, cls = int(e[0]), int(e[1]), int(e[2]), int(e[3])
            score = float(e[4]) if len(e) >= 5 else 1.0
            if score < conf_thresh or cls <= 0 or cls >= C:
                continue
            dets.append((score, team, shirt, cls, t_det))
        dets.sort(key=lambda x: -x[0])

        matched_gt = {kk: [False] * len(v) for kk, v in gt_events.items()}

        for score, team, shirt, cls, t_det in dets:
            key_grp = (team, shirt, cls)
            if key_grp not in gt_events:
                continue
            times = gt_events[key_grp]
            used = matched_gt[key_grp]
            best_i, best_dt = -1, DELTA + 1
            for i, (t_gt, bbox, replay) in enumerate(times):
                if used[i]:
                    continue
                dt = abs(t_gt - t_det)
                if dt <= DELTA and dt < best_dt:
                    best_dt, best_i = dt, i
            if best_i >= 0:
                used[best_i] = True

        events_out = []
        for key_grp, times in gt_events.items():
            team, shirt, cls = key_grp
            used = matched_gt[key_grp]
            for (t_gt, bbox, replay), is_matched in zip(times, used):
                events_out.append((team, shirt, cls, t_gt, bbox, replay, is_matched))

        results[k] = events_out

    return results


if __name__ == '__main__':

    gt = json.load(open(GT_PATH))
    pred20 = json.load(open(PRED20_PATH))
    pred40 = json.load(open(PRED40_PATH))

    m20 = match_predictions(gt, pred20, CONF_THRESH)
    m40 = match_predictions(gt, pred40, CONF_THRESH)

    # Align model_20's and model_40's per-event match status (same GT events, same order per game)
    tracking_failure_total = 0
    tracking_failure_missed_by_20 = 0
    tracking_failure_recovered_by_40 = 0
    tracking_failure_still_missed = 0

    visible_total = 0
    visible_missed_by_20 = 0
    visible_recovered_by_40 = 0
    visible_still_missed = 0

    per_class_recovery = {c: [0, 0] for c in range(1, C)}  # cls -> [tracking_failure_missed_by_20, recovered_by_40]

    for game in m20:
        events20 = m20[game]
        events40_lookup = {(t, s, c, f): matched for (t, s, c, f, b, r, matched) in m40.get(game, [])}

        for (team, shirt, cls, frame, bbox, replay, matched20) in events20:
            matched40 = events40_lookup.get((team, shirt, cls, frame), False)

            if bbox == 0:  # tracking failure: no visible bbox for this GT event
                tracking_failure_total += 1
                if not matched20:
                    tracking_failure_missed_by_20 += 1
                    per_class_recovery[cls][0] += 1
                    if matched40:
                        tracking_failure_recovered_by_40 += 1
                        per_class_recovery[cls][1] += 1
                    else:
                        tracking_failure_still_missed += 1
            else:  # bbox == 1, visible/live player
                visible_total += 1
                if not matched20:
                    visible_missed_by_20 += 1
                    if matched40:
                        visible_recovered_by_40 += 1
                    else:
                        visible_still_missed += 1

    print("=" * 70)
    print("TRACKING-FAILURE RECOVERY ANALYSIS (val set, conf_thresh=0.70, delta=12)")
    print("=" * 70)

    print(f"\n--- Events with NO visible bbox (tracking failure) ---")
    print(f"Total such GT events:              {tracking_failure_total}")
    print(f"Missed by model_20 (video only):   {tracking_failure_missed_by_20} "
          f"({100*tracking_failure_missed_by_20/max(tracking_failure_total,1):.1f}% of all no-bbox events)")
    print(f"  -> Recovered by model_40 (audio): {tracking_failure_recovered_by_40} "
          f"({100*tracking_failure_recovered_by_40/max(tracking_failure_missed_by_20,1):.1f}% of model_20's no-bbox misses)")
    print(f"  -> Still missed by model_40:      {tracking_failure_still_missed}")

    print(f"\n--- Events WITH visible bbox (control group, not a tracking failure) ---")
    print(f"Total such GT events:              {visible_total}")
    print(f"Missed by model_20 (video only):   {visible_missed_by_20} "
          f"({100*visible_missed_by_20/max(visible_total,1):.1f}% of all visible-bbox events)")
    print(f"  -> Recovered by model_40 (audio): {visible_recovered_by_40} "
          f"({100*visible_recovered_by_40/max(visible_missed_by_20,1):.1f}% of model_20's visible-bbox misses)")
    print(f"  -> Still missed by model_40:      {visible_still_missed}")

    print(f"\n--- Per-class breakdown (tracking-failure misses only) ---")
    print(f"{'class':10s} {'missed_by_20':>14s} {'recovered_by_40':>16s} {'recovery_rate':>14s}")
    for c in range(1, C):
        missed, recovered = per_class_recovery[c]
        rate = f"{100*recovered/missed:.1f}%" if missed > 0 else "n/a"
        print(f"{CLASS_NAMES[c]:10s} {missed:>14d} {recovered:>16d} {rate:>14s}")
