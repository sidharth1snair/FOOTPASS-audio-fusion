# FOOTPASS: Audio-Visual Fusion for Player-Centric Action Spotting

### _An M.Tech thesis extension of the [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS) baseline, adding a learned audio-visual temporal fusion mechanism_

---

## Overview

This repository extends the official **FOOTPASS** baseline (TAAD + DST, Ochin et al., 2026) for player-centric ball-action spotting in soccer broadcast videos. The original FOOTPASS pipeline detects *what* action happened, *who* performed it, and *when* — using video and tactical/positional data only.

This project adds a new **audio branch**: broadcast crowd noise and commentary carry independent, class-specific timing cues about on-field events (a goal's crowd roar builds differently than a foul's brief reaction). This work extracts those audio cues, learns a per-action-class temporal offset between the visual event and its audio reaction, and fuses that signal into the TAAD model's detection logits.

---

## Motivation

The FOOTPASS paper reports that **~20% of ground-truth events** have no usable player bounding box at the event frame (player-tracking failure), which limits the video-only model's detection recall. No existing audio-visual fusion work in the action-spotting literature learns an explicit, per-class timing offset between a visual event and its audio reaction, and none has been applied to FOOTPASS's tactical baseline. This project investigates whether such a mechanism can improve detection performance.

---

## What this adds on top of FOOTPASS

| Component | Description |
|---|---|
| `utils/audio_utils.py` | Extracts crowd-noise and commentary energy from broadcast audio via Butterworth bandpass filtering, frame-aligned to the video timeline |
| `extract_audio_features.py` | Batch-extracts audio features for a folder of match videos into a single HDF5 file |
| `aggregate_audio_by_class.py`, `check_audio_alignment.py` | Sanity-check tools: validate that audio energy actually correlates with ground-truth event timing, per class, before building the full model |
| `models/audio_fusion.py` | `AudioEncoder` (1D temporal CNN) + `EventConditionedAudioHead`: a learned per-class temporal offset (τ), implemented as a differentiable `grid_sample` shift, so the model learns *when* each action class's audio reaction typically lags the visual event |
| `models/model_TAAD_baseline.py` | The audio branch fused directly into TAAD's per-player, per-frame detection logits |
| `train_audio_baseline.py` | A standalone audio-only detector, used to validate the learned-τ mechanism in isolation before integrating it into the full video pipeline |

---

## Key findings so far

**1. The learned-τ mechanism is real and externally validated.** Averaging raw audio energy around ground-truth `shot` events (no model involved) shows commentary excitement peaking ~1.6s after the event. Independently, the learned τ parameter for the `shot` class converges to ~1.1–1.4s through gradient descent on a detection objective. Two unrelated methods agreeing on the same timing offset is strong evidence the mechanism captures something real.

**2. Fusing audio into the full model improves detection.** On the official validation split, at matched optimal confidence threshold:

| Model | F1 |
|---|---|
| TAAD + DST (video only) | 0.404 |
| TAAD + DST + audio fusion | **0.472** |

A ~17% relative improvement, driven primarily by **precision** gains (fewer false alarms) rather than recall — audio corroboration sharpens detections rather than finding entirely new events.

**3. Audio fusion does *not* recover the specific tracking-failure gap it targeted.** Breaking down recall by whether the ground-truth event had a visible player bounding box:

| Event type | Missed by video-only | Recovered by audio fusion |
|---|---|---|
| No visible bbox (tracking failure) | 98.3% | 0.4% — negligible |
| Visible bbox (normal case) | 48.2% | 20.3% — solid recovery |

Audio carries timing/class information but no player-identity information, so it cannot resolve cases where the video branch has no idea *which player* performed the action. The overall F1 improvement comes from sharpening detections for players the model can already see, not from recovering the originally-targeted tracking-failure cases. This is an open problem for future work.

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
```

```bash
git clone https://github.com/sidharth1snair/FOOTPASS-audio-fusion.git
cd FOOTPASS-audio-fusion
pip install -r requirements.txt   # or install the packages above manually
```

Dataset access requires a SoccerNet NDA — see the original [FOOTPASS repository](https://github.com/JeremieOchin/FOOTPASS) for the access form and data layout.

---

## Usage

**1. Extract audio features** from a folder of full-HD match videos (with audio):
```bash
python extract_audio_features.py --video_dir videos_fullHD --save_path data/audio_features.h5
```

**2. Sanity-check the audio signal** against known event timestamps:
```bash
python check_audio_alignment.py --game game_1
python aggregate_audio_by_class.py
```

**3. Train the full TAAD + audio-fusion model** (same interface as the original baseline, now with an audio branch active):
```bash
python train_TAAD_Baseline.py --data_root <path> --run_path <path> --epochs 20
```

**4. Run inference and evaluate**, exactly as in the original pipeline:
```bash
python run_TAAD_on_matches.py --model_checkpoint <ckpt> --set_to_run val
python NPpreds2JSON.py --set_to_run val
python evaluation.py --predictions_file <json> --ground_truth_file <gt_json>
```

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
