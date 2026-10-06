import argparse
import os
import glob
import h5py
import av

from utils.audio_utils import compute_audio_features


def parse_args():

    p = argparse.ArgumentParser(description="Extract per-frame crowd/commentary energy from match videos")

    p.add_argument("--video_dir", type=str, required=True, help="Folder containing game_X.mp4 files (fullHD, with audio)")
    p.add_argument("--save_path", type=str, default=os.path.join("data", "audio_features.h5"), help="Output HDF5 file")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--overwrite", action="store_true", help="Recompute games already present in the output file")

    return p.parse_args()


def get_n_frames(video_path):

    container = av.open(video_path)
    n_frames = container.streams.video[0].frames
    container.close()
    return n_frames


if __name__ == '__main__':

    args = parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    video_files = sorted(glob.glob(os.path.join(args.video_dir, "game_*.mp4")))
    print(f"[INFO] Found {len(video_files)} videos in {args.video_dir}")

    mode = 'a' if os.path.exists(args.save_path) else 'w'

    with h5py.File(args.save_path, mode) as h5f:

        for video_path in video_files:

            game_key = os.path.splitext(os.path.basename(video_path))[0]  # e.g. "game_1"

            if game_key in h5f and not args.overwrite:
                print(f"[SKIP] {game_key} already in {args.save_path}")
                continue

            if game_key in h5f:
                del h5f[game_key]

            n_frames = get_n_frames(video_path)

            try:
                features = compute_audio_features(video_path, n_frames=n_frames, fps=args.fps)
            except ValueError as e:
                print(f"[WARN] {game_key}: {e}")
                continue

            grp = h5f.create_group(game_key)
            grp.create_dataset('crowd_energy', data=features['crowd_energy'], compression='gzip')
            grp.create_dataset('commentary_energy', data=features['commentary_energy'], compression='gzip')
            grp.attrs['n_frames'] = n_frames
            grp.attrs['fps'] = args.fps

            print(f"[OK] {game_key}: {n_frames} frames processed")

    print(f"[INFO] Saved audio features to {args.save_path}")
