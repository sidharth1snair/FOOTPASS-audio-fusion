import argparse
import json
import os

import av
import cv2
import h5py
import numpy as np


CLASS_NAMES = ['background', 'drive', 'pass', 'cross', 'throw-in', 'shot', 'header', 'tackle', 'block']

# How highlight-worthy each action class is on its own (index = class id).
DEFAULT_CLASS_WEIGHTS = [0.0, 0.1, 0.1, 0.5, 0.2, 1.0, 0.6, 0.5, 0.5]


def parse_args():

    p = argparse.ArgumentParser(description="Build a highlight reel from play-by-play event predictions")

    p.add_argument("--events_json", type=str, default=None, help="Predictions (TAAD / DST) or ground-truth play-by-play JSON")
    p.add_argument("--is_gt", action="store_true", help="events_json is ground truth (rows have no confidence score)")
    p.add_argument("--game", type=int, default=None, help="Game number, e.g. 18 for game_18")
    p.add_argument("--audio_only", action="store_true",
                   help="Any broadcast video: pick moments from crowd/commentary peaks only (no events, no player data)")
    p.add_argument("--video", type=str, default=None, help="Path of the video for --audio_only (any match with sound)")
    p.add_argument("--peak_z", type=float, default=2.0, help="--audio_only: minimum excitement (robust z-score) of a peak")
    p.add_argument("--video_dir", type=str, default="videos", help="Folder with game_X.mp4 (use the full-HD versions to get audio in the reel)")
    p.add_argument("--audio_h5", type=str, default=os.path.join("data", "audio_features.h5"))
    p.add_argument("--checkpoint", type=str, default=None, help="Audio-fusion checkpoint to read the learned per-class tau from")
    p.add_argument("--out_dir", type=str, default="highlights")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--conf_thresh", type=float, default=0.70)
    p.add_argument("--target_seconds", type=float, default=180.0, help="Approximate reel length")
    p.add_argument("--merge_gap", type=float, default=10.0, help="Key events within this many seconds of a moment's first event join it")
    p.add_argument("--min_class_weight", type=float, default=0.5, help="Only classes at least this highlight-worthy start moments (default: cross, shot, header, tackle, block)")
    p.add_argument("--pre_roll", type=float, default=5.0)
    p.add_argument("--post_roll", type=float, default=3.0)
    p.add_argument("--max_reaction", type=float, default=4.0, help="Cap (s) on the audio-reaction window after an event")
    p.add_argument("--audio_weight", type=float, default=2.0, help="How strongly audio excitement boosts a moment's score")
    p.add_argument("--compare_gt", type=str, default=None, help="GT JSON: also build the GT reel and report moment recall")
    p.add_argument("--no_video", action="store_true", help="Only select moments and write the JSON, skip rendering")
    p.add_argument("--no_scoreboard", action="store_true", help="Don't detect goals from the on-screen score")
    p.add_argument("--score_box", type=str, default="318,66,90,38", help="x,y,w,h of the score digits (1920x1080 pixels)")
    p.add_argument("--name_boxes", type=str, default="140,66,160,38;455,66,130,38",
                   help="x,y,w,h;... of fixed scoreboard parts (team names) used to tell when the graphic is on screen")
    p.add_argument("--goal_lookback", type=float, default=60.0, help="Seconds before a score change to search for the goal shot/header")
    p.add_argument("--scoreboard_preview", type=float, default=None, help="Save a frame at this time (s) with the boxes drawn, then exit")

    args = p.parse_args()
    if args.audio_only:
        if args.video is None and args.game is None:
            p.error("--audio_only needs --video (or --game to use videos_dir/game_X.mp4)")
    elif args.events_json is None or args.game is None:
        p.error("--events_json and --game are required (or use --audio_only)")
    return args


def load_events(path, game, is_gt):

    with open(path, 'r') as f:
        data = json.load(f)

    rows = []
    for key in data['keys']:
        if not key.startswith(f"game_{game}_"):
            continue
        for e in data['events'][key]:
            frame, team, shirt, cls = int(e[0]), int(e[1]), int(e[2]), int(e[3])
            conf = 1.0 if is_gt else float(e[4])
            rows.append((frame, team, shirt, cls, conf))

    rows.sort(key=lambda r: r[0])
    return rows


def load_tau_seconds(checkpoint, fps):

    if checkpoint is None:
        return np.zeros(len(CLASS_NAMES))

    import torch
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    tau = ckpt['model_state_dict']['audio_head.tau'].numpy() / fps
    return np.concatenate([[0.0], tau])


def excitement_from_energy(comm, crowd):
    """Commentary + crowd energy as a robust z-score against the match's own baseline."""

    def robust_z(x):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        return (x - med) / (1.4826 * mad)

    return 0.7 * robust_z(comm) + 0.3 * robust_z(crowd)


def load_excitement_signal(audio_h5, game):
    """Excitement signal from the precomputed audio features (None if no audio for this game)."""

    if game is None or not os.path.exists(audio_h5):
        return None

    with h5py.File(audio_h5, 'r') as af:
        key = f"game_{game}"
        if key not in af:
            return None
        comm = af[key]['commentary_energy'][:]
        crowd = af[key]['crowd_energy'][:]

    return excitement_from_energy(comm, crowd)


def compute_excitement_from_video(video_path, fps, cache_path):
    """Extract crowd/commentary energy straight from a video file (cached), for videos outside FOOTPASS."""

    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return excitement_from_energy(d['commentary_energy'], d['crowd_energy'])

    from utils.audio_utils import compute_audio_features

    src = av.open(video_path)
    if not src.streams.audio:
        src.close()
        raise SystemExit(f"[ERROR] {video_path} has no audio track: audio-only mode needs sound")
    n_frames = int(src.duration / 1e6 * fps)
    src.close()

    feats = compute_audio_features(video_path, n_frames=n_frames, fps=fps)
    np.savez(cache_path, crowd_energy=feats['crowd_energy'], commentary_energy=feats['commentary_energy'])
    return excitement_from_energy(feats['commentary_energy'], feats['crowd_energy'])


def audio_only_moments(excitement, args):
    """One moment per strong, separated crowd/commentary peak. The reaction comes after the action, so the
    clip starts well before the peak."""

    from scipy.signal import find_peaks

    win = max(int(args.fps), 1)
    smooth = np.convolve(excitement, np.ones(win) / win, mode='same')
    peaks, props = find_peaks(smooth, height=args.peak_z, distance=int(args.merge_gap * args.fps))

    moments = []
    for frame, h in zip(peaks, props['peak_heights']):
        t = frame / args.fps
        ev = dict(frame=int(frame), team=-1, shirt=-1, cls=0, conf=1.0, excitement=float(h),
                  reaction_end=0.0, score=float(h), audio_only=True)
        moments.append({'events': [ev], 'best': ev, 'score': float(h),
                        'start': max(t - 10.0, 0.0), 'end': t + 4.0})
    return moments


def _scaled_box(box, width, height):
    x, y, w, h = box
    sx, sy = width / 1920, height / 1080
    return int(x * sx), int(y * sy), max(int(w * sx), 1), max(int(h * sy), 1)


def _crop_gray(img, box):
    x, y, w, h = box
    return cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)


def scan_scoreboard(video_path, score_box, name_boxes, cache_path, change_thresh, visible_thresh, persist):
    """
    Finds the times at which the on-screen score changes. Only keyframes are decoded (about one per second).
    The team-name blocks, which never change, tell us when the scoreboard graphic is on screen; a change in
    the score box counts only once it has persisted for `persist` on-screen samples, which ignores replays,
    close-ups and short overlays such as substitution banners.
    """

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    src = av.open(video_path)
    vin = src.streams.video[0]
    vin.codec_context.skip_frame = 'NONKEY'
    W, H = vin.codec_context.width, vin.codec_context.height
    sbox = _scaled_box(score_box, W, H)
    nboxes = [_scaled_box(b, W, H) for b in name_boxes]

    times, names, scores = [], [], []
    for frame in src.decode(vin):
        if frame.time is None:
            continue
        img = frame.to_ndarray(format='bgr24')
        times.append(frame.time)
        names.append([_crop_gray(img, b) for b in nboxes])
        scores.append(_crop_gray(img, sbox))
    src.close()

    refs = [np.median(np.stack([n[i] for n in names]), axis=0) for i in range(len(nboxes))]
    visible = [all(float(np.mean(np.abs(n[i] - refs[i]))) < visible_thresh for i in range(len(nboxes))) for n in names]
    vis = [(t, s) for t, s, v in zip(times, scores, visible) if v]

    def diff(a, b):
        return float(np.mean(np.abs(a - b)))

    goals = []
    if vis:
        states = [np.median(np.stack([s for _, s in vis[:persist]]), axis=0)]
        run = []
        for t, s in vis:
            if diff(s, states[-1]) <= change_thresh:
                run = []
                continue
            if run and diff(s, run[0][1]) > change_thresh:
                run = []
            run.append((t, s))
            if len(run) >= persist:
                new_state = np.median(np.stack([x for _, x in run]), axis=0)
                if goals and len(states) >= 2 and not goals[-1]['overturned'] \
                        and diff(new_state, states[-2]) <= change_thresh:
                    # score went back to what it was before the last goal: that goal was overturned (e.g. VAR)
                    goals[-1]['overturned'] = True
                    states.pop()
                else:
                    goals.append({'time': round(run[0][0], 1), 'overturned': False})
                    states.append(new_state)
                run = []

    result = {'goals': goals, 'visible_fraction': round(len(vis) / max(len(times), 1), 3)}
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)
    return result


def score_events(rows, args, tau_s, excitement, conf_thresh=None):

    conf_thresh = args.conf_thresh if conf_thresh is None else conf_thresh
    scored = []
    for frame, team, shirt, cls, conf in rows:
        if cls == 0 or conf < conf_thresh:
            continue

        reaction_end = min(max(tau_s[cls], 0.0) + 1.0, args.max_reaction)
        excite = 0.0
        if excitement is not None:
            lo = frame
            hi = min(frame + int(round(reaction_end * args.fps)) + 1, len(excitement))
            if hi > lo:
                excite = float(np.clip(np.max(excitement[lo:hi]) / 6.0, 0.0, 1.0))

        score = DEFAULT_CLASS_WEIGHTS[cls] * conf * (1.0 + args.audio_weight * excite)
        scored.append(dict(frame=frame, team=team, shirt=shirt, cls=cls, conf=conf,
                           excitement=excite, reaction_end=reaction_end, score=score))
    return scored


GOAL_CLASSES = (5, 6)  # shot, header


def attach_goals(events, all_events, goals, args):
    """Turn each on-screen score change into a goal event: the last shot/header before the change."""

    events = list(events)
    for g in goals:
        T = g['time']
        found = None
        for pool in (events, all_events):
            cands = [e for e in pool if e['cls'] in GOAL_CLASSES and not e.get('goal')
                     and T - args.goal_lookback <= e['frame'] / args.fps <= T]
            if cands:
                found = max(cands, key=lambda e: e['frame'])
                break
        if found is None:
            # no shot/header detected just before the score change: centre the clip on the change itself
            found = dict(frame=int((T - 6) * args.fps), team=-1, shirt=-1, cls=5, conf=0.0,
                         excitement=0.0, reaction_end=0.0)
        goal_ev = dict(found, goal=True, overturned=g['overturned'], score=50.0 if g['overturned'] else 100.0)
        events = [e for e in events if e is not found] + [goal_ev]
    events.sort(key=lambda e: e['frame'])
    return events


def group_moments(events, args):

    moments = []
    gap = args.merge_gap * args.fps
    key_events = [e for e in events if e.get('goal') or DEFAULT_CLASS_WEIGHTS[e['cls']] >= args.min_class_weight]
    for ev in key_events:
        # measured from the moment's first event so continuous play can't chain into one long clip
        if moments and ev['frame'] - moments[-1]['events'][0]['frame'] <= gap:
            moments[-1]['events'].append(ev)
        else:
            moments.append({'events': [ev]})

    for m in moments:
        best = max(m['events'], key=lambda e: e['score'])
        last = m['events'][-1]
        m['score'] = best['score']
        m['best'] = best
        m['start'] = max(m['events'][0]['frame'] / args.fps - args.pre_roll, 0.0)
        m['end'] = last['frame'] / args.fps + last['reaction_end'] + args.post_roll
        if best.get('goal'):
            # longer build-up and the celebration
            t = best['frame'] / args.fps
            m['start'] = max(min(m['start'], t - 8.0), 0.0)
            m['end'] = max(m['end'], t + 12.0)
    return moments


def select_moments(moments, args):

    chosen, total = [], 0.0
    for m in sorted(moments, key=lambda m: m['score'], reverse=True):
        if m['score'] <= 0 or total >= args.target_seconds:
            break
        chosen.append(m)
        total += m['end'] - m['start']

    chosen.sort(key=lambda m: m['start'])

    # pre/post roll can make neighbouring clips overlap: merge them into one clip
    merged = []
    for m in chosen:
        if merged and m['start'] <= merged[-1]['end']:
            prev = merged[-1]
            prev['end'] = max(prev['end'], m['end'])
            prev['events'] += m['events']
            if m['score'] > prev['score']:
                prev['score'], prev['best'] = m['score'], m['best']
        else:
            merged.append(dict(m))
    return merged


def caption(ev):

    if ev.get('audio_only'):
        return "Highlight"
    if ev.get('goal'):
        head = "Goal ruled out" if ev.get('overturned') else "GOAL"
    else:
        head = CLASS_NAMES[ev['cls']].capitalize()
    if ev['team'] < 0:
        return head
    team = "Team A" if ev['team'] == 0 else "Team B"
    return f"{head} - {team} #{ev['shirt']}"


def render_reel(clips, video_path, out_path, fps):

    src = av.open(video_path)
    vin = src.streams.video[0]
    ain = src.streams.audio[0] if src.streams.audio else None

    dst = av.open(out_path, 'w')
    out_fps = vin.average_rate or fps
    vout = dst.add_stream('libx264', rate=out_fps)
    vout.width, vout.height = vin.codec_context.width, vin.codec_context.height
    vout.pix_fmt = 'yuv420p'
    vout.options = {'crf': '23', 'preset': 'veryfast'}

    aout, resampler, a_pts = None, None, 0
    if ain is not None:
        aout = dst.add_stream('aac', rate=ain.codec_context.sample_rate)
        aout.layout = 'stereo'
        resampler = av.AudioResampler(format='fltp', layout='stereo', rate=ain.codec_context.sample_rate)

    v_pts = 0
    for clip in clips:
        start, end = clip['start'], clip['end']
        label = caption(clip['best'])
        label_until = clip['best']['frame'] / fps + clip['best']['reaction_end']

        src.seek(int(start / vin.time_base), stream=vin, backward=True, any_frame=False)
        streams = [vin] + ([ain] if ain is not None else [])
        clip_v_start, clip_a_start = v_pts, a_pts
        video_done = False

        for packet in src.demux(*streams):
            if packet.dts is None:
                continue
            for frame in packet.decode():
                t = frame.time
                if t is None or t < start:
                    continue
                if packet.stream is vin:
                    if t > end:
                        video_done = True
                        continue
                    img = frame.to_ndarray(format='bgr24')
                    if t <= label_until:
                        cv2.rectangle(img, (0, img.shape[0] - 42), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
                        cv2.putText(img, label, (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8 * img.shape[0] / 352, (255, 255, 255), 2, cv2.LINE_AA)
                    out = av.VideoFrame.from_ndarray(img, format='bgr24')
                    out.pts = v_pts
                    v_pts += 1
                    for pkt in vout.encode(out):
                        dst.mux(pkt)
                elif t <= end:
                    # keep audio in step with the video already written for this clip
                    if (a_pts - clip_a_start) / aout.rate > (v_pts - clip_v_start) / float(out_fps) + 0.5:
                        continue
                    for rf in resampler.resample(frame):
                        rf.pts = a_pts
                        a_pts += rf.samples
                        for pkt in aout.encode(rf):
                            dst.mux(pkt)
            if video_done:
                break

        if aout is not None:
            # pad/trim the clip's audio to exactly match its video length
            target = clip_a_start + int(round((v_pts - clip_v_start) / float(out_fps) * aout.rate))
            if a_pts < target:
                silence = av.AudioFrame(format='fltp', layout='stereo', samples=target - a_pts)
                for plane in silence.planes:
                    plane.update(bytes(plane.buffer_size))
                silence.sample_rate = aout.rate
                silence.pts = a_pts
                a_pts = target
                for pkt in aout.encode(silence):
                    dst.mux(pkt)

    for pkt in vout.encode():
        dst.mux(pkt)
    if aout is not None:
        for pkt in aout.encode():
            dst.mux(pkt)
    dst.close()
    src.close()

    return ain is not None


def clip_overlap(a, b):
    return max(0.0, min(a['end'], b['end']) - max(a['start'], b['start']))


def build_clips(events_json, is_gt, args, tau_s, excitement, goals):
    rows = load_events(events_json, args.game, is_gt)
    events = score_events(rows, args, tau_s, excitement)
    if goals:
        events = attach_goals(events, score_events(rows, args, tau_s, excitement, conf_thresh=0.15), goals, args)
    return select_moments(group_moments(events, args), args)


def parse_boxes(text):
    return [tuple(int(v) for v in b.split(',')) for b in text.split(';')]


def save_scoreboard_preview(video_path, t, score_box, name_boxes, out_path):
    src = av.open(video_path)
    vin = src.streams.video[0]
    src.seek(int(t / vin.time_base), stream=vin)
    for frame in src.decode(vin):
        if frame.time is not None and frame.time >= t:
            img = frame.to_ndarray(format='bgr24')
            break
    src.close()
    H, W = img.shape[:2]
    for box, color in [(score_box, (0, 0, 255))] + [(b, (0, 255, 0)) for b in name_boxes]:
        x, y, w, h = _scaled_box(box, W, H)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite(out_path, img)


def to_json(clips):
    return [dict(start_s=round(c['start'], 2), end_s=round(c['end'], 2), score=round(c['score'], 3),
                 label=caption(c['best']), best_frame=c['best']['frame'],
                 excitement=round(c['best']['excitement'], 3), n_events=len(c['events']))
            for c in clips]


def run_audio_only(args):

    video_path = args.video or os.path.join(args.video_dir, f"game_{args.game}.mp4")
    name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"[INFO] Audio-only mode on {video_path} (first run extracts the audio, a few minutes)")
    excitement = compute_excitement_from_video(video_path, args.fps, os.path.join(args.out_dir, f"audio_{name}.npz"))

    clips = select_moments(audio_only_moments(excitement, args), args)
    total = sum(c['end'] - c['start'] for c in clips)
    print(f"[INFO] Selected {len(clips)} clips, {total:.0f} s total")
    for c in clips:
        print(f"    {c['start'] / 60:5.1f}-{c['end'] / 60:5.1f} min  excitement {c['score']:.2f}")

    with open(os.path.join(args.out_dir, f"highlights_{name}_audio_only.json"), 'w') as f:
        json.dump(to_json(clips), f, indent=2)

    if args.compare_gt and args.game is not None:
        tau_s = load_tau_seconds(args.checkpoint, args.fps)
        gt_clips = build_clips(args.compare_gt, True, args, tau_s, excitement, [])
        hit = sum(1 for g in gt_clips if any(clip_overlap(g, c) > 0 for c in clips))
        print(f"[EVAL] GT reel has {len(gt_clips)} moments; audio-only reel covers {hit} "
              f"({100 * hit / max(len(gt_clips), 1):.1f}% moment recall)")

    if not args.no_video:
        out_path = os.path.join(args.out_dir, f"highlights_{name}_audio_only.mp4")
        render_reel(clips, video_path, out_path, args.fps)
        print(f"[INFO] Saved {out_path}")


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.audio_only:
        run_audio_only(args)
        raise SystemExit

    tau_s = load_tau_seconds(args.checkpoint, args.fps)
    excitement = load_excitement_signal(args.audio_h5, args.game)
    if excitement is None:
        print(f"[WARN] No audio features for game_{args.game}: ranking uses class and confidence only")

    video_path = os.path.join(args.video_dir, f"game_{args.game}.mp4")
    score_box, name_boxes = parse_boxes(args.score_box)[0], parse_boxes(args.name_boxes)

    if args.scoreboard_preview is not None:
        out = os.path.join(args.out_dir, f"scoreboard_preview_game_{args.game}.png")
        save_scoreboard_preview(video_path, args.scoreboard_preview, score_box, name_boxes, out)
        print(f"[INFO] Saved {out}  (red = score box, green = fixed team-name boxes)")
        raise SystemExit

    goals = []
    if not args.no_scoreboard and os.path.exists(video_path):
        sb = scan_scoreboard(video_path, score_box, name_boxes,
                             os.path.join(args.out_dir, f"scoreboard_game_{args.game}.json"),
                             change_thresh=6.0, visible_thresh=25.0, persist=10)
        goals = sb['goals']
        print(f"[INFO] Scoreboard on screen {100 * sb['visible_fraction']:.0f}% of the time; "
              f"{len(goals)} score changes: " + ", ".join(
                  f"{g['time'] / 60:.1f} min{' (overturned)' if g['overturned'] else ''}" for g in goals))
        if sb['visible_fraction'] < 0.3:
            print("[WARN] Scoreboard rarely found: check --score_box/--name_boxes with --scoreboard_preview")
            goals = []

    clips = build_clips(args.events_json, args.is_gt, args, tau_s, excitement, goals)
    total = sum(c['end'] - c['start'] for c in clips)
    print(f"[INFO] Selected {len(clips)} clips, {total:.0f} s total")
    for c in clips:
        print(f"    {c['start'] / 60:5.1f}-{c['end'] / 60:5.1f} min  score {c['score']:.2f}  {caption(c['best'])}")

    tag = "gt" if args.is_gt else "pred"
    with open(os.path.join(args.out_dir, f"highlights_game_{args.game}_{tag}.json"), 'w') as f:
        json.dump(to_json(clips), f, indent=2)

    if args.compare_gt:
        gt_clips = build_clips(args.compare_gt, True, args, tau_s, excitement, goals)
        hit = sum(1 for g in gt_clips if any(clip_overlap(g, c) > 0 for c in clips))
        print(f"[EVAL] GT reel has {len(gt_clips)} moments; predicted reel covers {hit} "
              f"({100 * hit / max(len(gt_clips), 1):.1f}% moment recall)")
        with open(os.path.join(args.out_dir, f"highlights_game_{args.game}_gt.json"), 'w') as f:
            json.dump(to_json(gt_clips), f, indent=2)

    if not args.no_video:
        out_path = os.path.join(args.out_dir, f"highlights_game_{args.game}_{tag}.mp4")
        has_audio = render_reel(clips, video_path, out_path, args.fps)
        if not has_audio:
            print(f"[WARN] {video_path} has no audio track: reel is silent (use the full-HD videos via --video_dir)")
        print(f"[INFO] Saved {out_path}")
