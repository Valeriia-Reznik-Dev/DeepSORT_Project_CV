# Modern DeepSORT — Detection & ReID Extension

Course project for *Deep Learning for Computer Vision*. The original
[DeepSORT](https://github.com/nwojke/deep_sort) (Wojke et al.) is extended with
modern, switchable **detection** and **ReID** models, an optional **segmentation**
stage, and a standalone **body-ReID identity database** (additional task). The
system is evaluated on six MOT Challenge sequences using the standard MOTChallenge
HOTA protocol (TrackEval, `trackeval==1.3.0`).

The original DeepSORT core (`deep_sort/`, `application_util/`) is kept **unchanged**
— all new code lives in dedicated packages around it. This preserves the upstream
commit history and avoids copying from modern DeepSORT forks.

## Headline results

**Target metric:** mean **HOTA** over six test videos (arithmetic mean of
per-sequence HOTA from TrackEval, `DO_PREPROC=True`, class = pedestrian). End-to-end
**FPS** is mean throughput on Colab T4 (detector + ReID + tracker; excludes model
load and disk I/O). Reported FPS values are from the Colab notebook run logs
(`notebooks/baseline_colab.ipynb`).

**Final configuration:** YOLOv8n + OSNet, `min_confidence=0.4`,
`max_cosine_distance=0.2` → mean HOTA **52.60** at **17.54 FPS** (**+12.43** vs.
baseline **40.17**), beats the baseline on every test video, real-time (≥5 FPS).

Also implemented and evaluated: standalone identity database (resolved FM **0.657**
at radius **0.4**); optional segmentation pipeline `yolo_seg` + `osnet_seg` (mean
HOTA **51.08** at **10.60 FPS**).

| Configuration | Mean HOTA | Beats baseline on all 6 | Mean FPS (T4) |
|---|---:|---|---:|
| Original DeepSORT (baseline) | 40.17 | — | — |
| **YOLOv8n + OSNet** (defaults) | **51.23** | yes | 16.03 |
| **YOLOv8n + OSNet** (`min_confidence=0.4`, `max_cosine_distance=0.2`) | **52.60** | yes | 17.54 |

Upper bound with **ground-truth detections** + OSNet (ReID/association only): mean
HOTA **80.18** — the gap to the live pipeline is dominated by detector recall, not
by the association stage.

Per-sequence HOTA (same evaluation protocol):

| Sequence | Baseline | YOLOv8n + OSNet | YOLOv8n + OSNet (tuned) |
|---|---:|---:|---:|
| TUD-Campus | 39.86 | 47.96 | 52.96 |
| TUD-Stadtmitte | 36.75 | 55.11 | 54.76 |
| KITTI-17 | 43.41 | 55.71 | 57.68 |
| PETS09-S2L1 | 44.84 | 52.43 | 51.68 |
| MOT16-09 | 36.24 | 44.23 | 45.48 |
| MOT16-11 | 39.95 | 51.95 | 53.01 |
| **Mean (6 videos)** | **40.17** | **51.23** | **52.60** |

HOTA tables match `results/tracking/sweep_summary.csv` and
`results/param_sweep/sweep_min_confidence.csv`. Full ablations (detector × ReID
sweep, segmentation, identity DB, parameter evolution) are in the
[project report](#report-and-results).

## What is new (vs. upstream DeepSORT)

**Detectors** (3 sources): `YOLOv8n` (ultralytics), `NanoDet-Plus`
(RangiLyu/nanodet), `RTMDet` (OpenMMLab / MMDetection).

**ReID** (2 sources, 2 architecture families): `OSNet`, `ResNet50` (torchreid);
`ResNet50-IBN`, `BoT R50` (`fastreid`, fast-reid).

**Segmentation** (optional, 3 sources): `YOLOv8n-seg` (ultralytics),
`Mask R-CNN R50-FPN` (detectron2), `DeepLabV3+` (segmentation-models-pytorch).
Masks remove the background from each crop before ReID.

**Standalone identity database** (additional task): an online gallery with
centroid / kNN representation, radius-based lookup, majority-vote resolution over
a time window, and distance-based conflict resolution between tracks.

**Evaluation harness**: detector P/R/F1 vs. GT, standalone ReID clustering
(Fowlkes–Mallows / Silhouette / Calinski–Harabasz), ReID-inside-tracker HOTA with
GT detections, and full-system HOTA / MOTA / IDF1 via TrackEval.

All detector / ReID / segmentation / identity components are **selectable before
each run** from config files or CLI flags.

## Repository structure

```
deep_sort/            Original DeepSORT core — UNCHANGED (Kalman, matching, Tracker)
application_util/     Original DeepSORT utilities — UNCHANGED
detectors/            Detector adapters: yolo, nanodet, mmdet (+ base)
reid/                 ReID adapters: torchreid_ext, fastreid_ext (+ base)
segmentation/         Segmenters: yolo_seg, detectron2_seg, smp_seg (+ base)
tracking/             pipeline.py (live det+ReID -> DeepSORT core), params
identity/             database.py, manager.py (additional task)
eval/                 trackeval_wrap, detector_metrics, reid_metrics,
                      identity_metrics, overlay_render
configs/              baseline_original.yaml, detectors.yaml, reid.yaml,
                      identity.yaml, tracker_params.yaml
scripts/              run_* (experiments), setup_*_colab, download_* (weights)
notebooks/            baseline_colab.ipynb  <-- main entry point
```

## Quick start (Google Colab — recommended)

The full pipeline runs end-to-end from the notebook:

1. Open `notebooks/baseline_colab.ipynb` in Colab and select a **GPU runtime (T4)**.
2. Run the setup cells. They clone the repo, install dependencies, set up
   torchreid / fast-reid, and mount datasets and model weights from Google Drive
   into `resources/`.
3. Run the experiment cells: baseline evaluation, detector metrics, ReID
   evaluation, the integrated tracker sweep, segmentation, identity-database
   evaluation, parameter tuning, and overlay rendering.

Numerical outputs are written to `results/`; qualitative overlay videos to
`overlays/original/` (baseline) and `overlays/best/` (best configuration).

## Local install

```bash
git clone https://github.com/Valeriia-Reznik-Dev/DeepSORT_Project_CV.git
cd DeepSORT_Project_CV

# Modern pipeline (detectors + ReID + tracking + eval)
pip install -r requirements-colab.txt       # numpy, opencv, scipy, pyyaml, trackeval==1.3.0
pip install -r requirements-detectors.txt   # ultralytics, scikit-learn, torch, torchvision

# torchreid / fast-reid (and NanoDet / MMDet if needed) are installed by the setup scripts:
python scripts/setup_reid_colab.py
python scripts/setup_detectors_colab.py
python scripts/setup_segmentation_colab.py   # optional, for segmentation backends
```

Reproducing the **original** baseline additionally needs the upstream MARS
appearance encoder (`mars-small128.pb`, TensorFlow); see `requirements-gpu.txt`
and `requirements-baseline.txt`.

### Data & weights (`resources/`, git-ignored)

Large binaries are not committed and are mounted from Google Drive into
`resources/`:

```
resources/
  detections/MOT15/train, MOT16/train     MOT video frames + ground truth
  detections/MOT15_train, MOT16_train     Public detections converted to .npy
  networks/mars-small128.pb               Original DeepSORT appearance encoder
  models/torchreid, fastreid, nanodet,    Modern detector / ReID / seg weights
         mmdet, smp
```

Modern detector and ReID weights can be fetched with
`scripts/download_detector_models.py` and `scripts/download_reid_models.py`.

**Test sequences:** TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1 (MOT15);
MOT16-09, MOT16-11 (MOT16).

## Reproducing individual stages (CLI)

Each notebook stage maps to a script. Defaults reproduce the reported setup.

```bash
# 1) Original DeepSORT baseline (public MOT detections -> results/baseline/original)
python scripts/run_baseline.py

# 2) Detector quality vs. GT (Precision / Recall / F1)
python scripts/run_detector_eval.py --detector yolo nanodet mmdet

# 3) Standalone ReID clustering on GT crops (FM / Silhouette / Calinski-Harabasz)
python scripts/run_reid_eval.py --model osnet resnet50 resnet50_ibn fastreid

# 4) Integrated tracker — single run (best global configuration)
python scripts/run_tracker.py --detector yolo --reid osnet \
    --min-confidence 0.4 --max-cosine-distance 0.2

#    ReID inside the tracker with GT detections (perfect-detector upper bound)
python scripts/run_tracker.py --detector yolo --reid osnet --gt-detections

#    Segmentation variant (mask background before ReID)
python scripts/run_tracker.py --detector yolo_seg --reid osnet --mask-background

#    Additional task — enable the standalone identity database
python scripts/run_tracker.py --detector yolo --reid osnet --identity

# 5) Model sweep (detector x ReID combinations)
python scripts/run_sweep.py

# 6) Parameter tuning
python scripts/run_param_sweep.py --param min_confidence     --values 0.2 0.3 0.4 0.5
python scripts/run_param_sweep.py --param max_cosine_distance --values 0.1 0.2 0.3 0.4

# 7) Identity-database ablations
python scripts/run_identity_eval.py
python scripts/run_identity_sweep.py --param radius --values 0.2 0.25 0.3 0.35 0.4

# 8) Score any result folder with TrackEval (HOTA / MOTA / IDF1)
python scripts/run_eval.py --tracker-name yolo_osnet \
    --results-dir results/tracking/yolo_osnet

# 9) Render overlay videos for a result folder
python scripts/run_overlays.py \
    --results-dir results/tracking/yolo_osnet --overlays-dir overlays/best
```

Useful flags: `--device cuda:0|cpu`, `--max-frames N` (quick smoke test),
`--params-config configs/tracker_params.yaml` (per-video tracker parameters).
Identity-DB defaults (radius, window, conflict policy) live in
`configs/identity.yaml`.

## Report and results

* **Report:** `ReznikV_Modern_DeepSORT_Project_Report.docx` — candidate models,
  selection rationale, all numerical experiments, and the optimal
  models/parameters with justification (submitted separately from the repo).
* **Experiment outputs:** produced in Colab under `results/` and `overlays/`, and
  archived in `project_outputs.zip` (the repository version-controls code,
  configs, and scripts only).

## Attribution & license

This project builds on **Deep SORT** by Nicolai Wojke, Alex Bewley, and Dietrich
Paulus — [nwojke/deep_sort](https://github.com/nwojke/deep_sort). The upstream
core and its commit history are preserved unchanged. Released under
**GPL-3.0** (see `LICENSE`).

```bibtex
@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017}, pages={3645--3649}, organization={IEEE}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018}, pages={748--756}, organization={IEEE}
}
```
