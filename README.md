# FOOTPASS: Audio-Visual Action Spotting and Highlight Generation

### _M.Tech thesis code extending the [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS) baseline: hidden-player inference, an event verifier, an audio branch with tested controls, and a highlight generator_

---

## Overview

This repository extends the official **FOOTPASS** baseline (TAAD + DST, Ochin et al., 2026) for player-centric ball-action spotting in soccer broadcasts: predicting *what* happened, *when*, and *which player* did it.

About one event in five is performed by a player with **no bounding box** (replays, close-ups, off-screen play), so a video detector cannot see it. This work adds:

1. **Hidden-Player event Inference (HPI)**: a 0.44 M-parameter model (Transformer across players + bidirectional GRU over 10 s) that predicts actions of hidden players from tracking data and the context of visible players.
2. **A one-action-at-a-time rule**: removes simultaneous predictions across all players, with no training.
3. **An Event Verifier**: a tracking-only model that checks each TAAD prediction against the movement of the named player and rescores it, measured against a per-class calibration baseline.
4. **An audio branch** with a learned per-class delay, and a controlled study of audio (wrong-match and time-shifted audio, PANNs embeddings, Whisper transcripts).
5. **A highlight generator** that ranks moments by class, confidence and audio excitement.

---

## Main results

Internal test set: 6 held-out matches, scored once with every setting fixed beforehand; mean ± sd of 3 seeds.

| System | Micro F1 | Macro F1 | Hidden recall |
|---|---|---|---|
| TAAD alone | 0.471 | 0.338 | 2.2% |
| TAAD + HPI + rule | 0.543 ± 0.001 | 0.400 ± 0.003 | 39.4 ± 3.0% |
| + per-class calibration | 0.587 ± 0.003 | 0.462 ± 0.010 | |
| **+ Event Verifier** | **0.699 ± 0.006** | **0.549 ± 0.018** | 42.1 ± 3.9% |

- The Verifier adds about +0.11 F1 over calibration alone (95% bootstrap interval +0.10 to +0.12 in every seed). Shifting its input by 5 s or shuffling the players makes F1 collapse to 0.22.
- This is at the level of the published TAAD + DST (0.675), with about 1/50 of its parameters. It is **not** a claim that this system beats DST, because DST was scored on different (challenge) matches.
- TAAD was trained on the internal test matches; HPI and the Verifier were not.

**Audio.** Under wrong-match and time-shifted controls, audio gives **no measurable gain** in spotting events or identifying hidden players. The earlier rise from 0.404 to 0.472 came from extra training epochs, not from the sound. Audio does help to judge importance: goals are louder than 73% of ordinary shots and headers in the same match (50% with wrong-match audio). It is therefore used to rank moments in highlight reels.

---

## Code

| File | Purpose |
|---|---|
| `utils/audio_utils.py`, `extract_audio_features.py` | Crowd (20–500 Hz) and commentary (300–3400 Hz) energy, frame-aligned to the video |
| `models/audio_fusion.py`, `models/model_TAAD_baseline.py` | Audio encoder and per-class learned delay τ, fused into TAAD logits |
| `train_TAAD_Baseline.py` | TAAD training (`--resume_path`, `--fixed_tau` for the τ = 0 ablation) |
| `kaggle_tau_ablation.ipynb` | Kaggle notebook for the τ = 0 ablation (reads credentials from Kaggle Secrets) |
| `build_hidden_dataset.py` | Builds per-half HDF5 grids (frames × 26 slots) of tracking features, TAAD probabilities and labels |
| `train_hidden.py`, `train_hidden_v2.py` | HPI model and training (loss on hidden player-frames only) |
| `hpi_exclusivity.py` | One-action rule; chooses rule, window and threshold on held-out matches |
| `verify_hidden_v2.py`, `show_hpi_results.py`, `hpi_final_test.py` | Evaluation, controls (shifted / shuffled inputs) and the one-time test scoring |
| `event_verifier.py` | Event Verifier: `train`, then `fuse` (dev-set logistic regression, calibration baseline, controls, `--test`) |
| `error_breakdown.py` | Groups the remaining wrong predictions (wrong player / team / class, timing, nothing there) |
| `audio_contribution_test.py`, `audio_probe_panns.py`, `whisper_probe.py`, `goal_audio_test.py`, `score_tau0.py` | Audio tests with controls |
| `generate_highlights.py`, `highlight_comparison.py`, `reel_goal_test.py` | Highlight reels and their shot / goal coverage |
| `annotate_events.py`, `make_thesis_figures.py`, `make_results_table.py` | Demo videos, thesis figures and tables |

---

## Installation

Environment (same as the original FOOTPASS baseline, plus audio dependencies):

```
Python 3.11.5
PyTorch 2.1.0
Torchvision 0.16.0
Albumentations 2.0.8
Numpy 1.26
Opencv-python 4.11
Decord 0.6
h5py 3.14
scipy          # Butterworth filtering
av             # audio extraction from video containers
matplotlib     # figures
```

```bash
git clone https://github.com/sidharth1snair/FOOTPASS-audio-fusion.git
cd FOOTPASS-audio-fusion
pip install -r requirements.txt   # or install the packages above manually
```

Dataset access requires a SoccerNet NDA. See the original [FOOTPASS repository](https://github.com/JeremieOchin/FOOTPASS) for the access form and data layout.

---

## Usage

**1. Audio features and TAAD** (as in the original pipeline, with the audio branch active):
```bash
python extract_audio_features.py --video_dir videos_fullHD --save_path data/audio_features.h5
python train_TAAD_Baseline.py --data_root <path> --run_path <path> --epochs 40
python run_TAAD_on_matches.py --model_checkpoint <ckpt> --set_to_run val
python NPpreds2JSON.py --set_to_run val
```

**2. HPI and the one-action rule:**
```bash
python build_hidden_dataset.py --set train --out hidden_data/train33.h5
python build_hidden_dataset.py --set val --out hidden_data/val.h5
python train_hidden_v2.py --train_h5 hidden_data/train33.h5 --run_name v7_33g_seed0 --seed 0 \
    --window 250 --batch_size 16 --dev_games 6,8,52
python hpi_exclusivity.py --run runs_hidden/v7_33g_seed0 --train_h5 hidden_data/train33.h5 --dev_games 6,8,52
```

**3. Event Verifier** (fit on held-out matches 6, 8, 52; score validation; then the test set once):
```bash
python event_verifier.py train --run ver_seed0 --seed 0
python event_verifier.py fuse  --run ver_seed0 --hpi_run v7_33g_seed0
python event_verifier.py fuse  --run ver_seed0 --hpi_run v7_33g_seed0 --control shift
python event_verifier.py fuse  --run ver_seed0 --hpi_run v7_33g_seed0 --test
```

**4. Highlights:**
```bash
python generate_highlights.py --events_json <predictions.json> --game 18 --checkpoint <taad_audio_ckpt>
```

Run any script with `-h` for all options.

---

## Credits

This project builds directly on the **FOOTPASS** dataset and baseline:

```bibtex
@article{Ochin2026Footpass,
  author  = {Jérémie Ochin and Raphaël Chekroun and Bogdan Stanciulescu and Sotiris Manitsaris},
  title   = {FOOTPASS: A multi-modal multi-agent tactical context dataset for play-by-play action spotting in soccer broadcast videos},
  journal = {Computer Vision and Image Understanding},
  volume  = {269},
  pages   = {104790},
  year    = {2026},
  doi     = {https://doi.org/10.1016/j.cviu.2026.104790},
}
```

Original repository: [github.com/JeremieOchin/FOOTPASS](https://github.com/JeremieOchin/FOOTPASS)
Dataset: [SoccerNet SN-PCBAS-2026](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026)

---

## License

Dataset annotations & baselines released under CC BY-NC 4.0 (per the original FOOTPASS license). Redistribution of SoccerNet broadcast videos is prohibited under the SoccerNet NDA.
