"""
Clean check for the final test: HPI trained WITHOUT any TAAD input, scored on the 6 frozen test games.

TAAD trained on the test games, so its input could flatter the main HPI model there. This model never sees TAAD,
so "HPI alone" below involves no TAAD at all. The rule and threshold are the ones chosen on the dev games.

Usage:
  python hpi_tracking_test.py --run v7_33g_track_seed0
"""
import argparse
import json
import os

import h5py

from hpi_exclusivity import score
from hpi_final_test import hpi_events, TEST_H5, TEST_LOGITS, TAAD_THR
from show_hpi_results import macro_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="v7_33g_track_seed0")
    p.add_argument("--ref", default="v7_33g_seed0", help="main HPI model with the same seed, for comparison")
    a = p.parse_args()
    run = os.path.join("runs_hidden", a.run)
    ref = os.path.join("runs_hidden", a.ref)
    ck = json.load(open(os.path.join(run, "exclusivity.json")))["chosen"]
    rc = json.load(open(os.path.join(ref, "exclusivity.json")))["chosen"]
    with h5py.File(TEST_H5, "r") as f:
        keys = sorted(f.keys())
    gt = json.load(open(os.path.join("playbyplay_GT", "playbyplay_train.json")))["events"]
    gt = {k: gt[k] for k in keys}
    taad = json.load(open(os.path.join("runs_hidden", "taad_test_events.json")))["events"]
    track = hpi_events(a.run, TEST_H5, TEST_LOGITS)
    json.dump({"keys": keys, "events": track}, open(os.path.join(run, "hpi_test_events.json"), "w"))
    main_ev = json.load(open(os.path.join(ref, "hpi_test_events.json")))["events"]

    print(f"tracking-only settings: {ck} | main model settings: {rc}\n")
    print(f"{'system':40s} {'micro':>6s} {'macro':>6s} {'hidden found':>18s} {'HPI prec':>8s}")
    out = {}
    for name, t, ev, c in [("Tracking-only HPI alone (no TAAD at all)", {}, track, ck),
                           ("Main HPI alone (TAAD as input)", {}, main_ev, rc),
                           ("TAAD + tracking-only HPI + rule", taad, track, ck),
                           ("TAAD + main HPI + rule", taad, main_ev, rc)]:
        r = score(t, ev, gt, keys, TAAD_THR, c["thr"], c["rule"], c["w"])[1]
        m = macro_f1(t, ev, gt, keys, c["thr"], c["rule"])
        print(f"{name:40s} {r['f1']:6.3f} {m:6.3f} {r['hidden_tp']:6d}/{r['hidden_n']} ({100 * r['hidden_recall']:4.1f}%) {r['hpi_precision']:8.2f}")
        out[name] = dict(r, macro=m)
    json.dump(out, open(os.path.join("runs_hidden", "final_test_tracking.json"), "w"), indent=2, default=float)
    print("\nsaved runs_hidden/final_test_tracking.json")


if __name__ == "__main__":
    main()
