# Multi-Object Tracking & Trajectory Analysis

A computer vision project for detecting and tracking multiple objects across video frames, analyzing their trajectories, comparing tracker configurations, and evaluating tracking performance using standard MOT metrics.

## Overview

The project combines **YOLO11n** object detection with **ByteTrack** multi-object tracking to generate persistent track IDs and trajectory data from video. It also includes trajectory visualization, configurable tracking experiments, and quantitative evaluation on a MOT17 sequence.

### Key Features

- Real-time-style object detection and tracking with YOLO11n + ByteTrack
- Track-level statistics and trajectory visualizations
- Comparison of ByteTrack configurations
- Pedestrian tracking evaluation using HOTA, IDF1, and MOTA
- Trajectory analytics: path length, displacement, speed, straightness, and observation gaps
- Automated tests for trajectory-analysis functions

## Pipeline

1. Read an input video.
2. Detect objects with pretrained YOLO11n.
3. Associate detections across frames with ByteTrack.
4. Export detections and track IDs to CSV.
5. Analyze trajectories and visualize tracking statistics.
6. Evaluate pedestrian tracking on a MOT17 sequence using TrackEval.

## Tech Stack

Python · OpenCV · Ultralytics YOLO · ByteTrack · NumPy · Pandas · Matplotlib · TrackEval · pytest

## Setup

```bash
git clone https://github.com/ArwaSaluji01/Multi-Object-Tracking-and-Trajectory-Analysis.git
cd multi-object-tracking

python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download or provide an input video at `data/input/sample.mp4`.

## Usage

Run tracking:

```bash
python src/track_video.py
```

Analyze track statistics:

```bash
python src/analyze_tracks.py
```

Generate visualizations:

```bash
python src/visualize_tracks.py
```

Compare tracker configurations:

```bash
python src/run_experiments.py
```

Analyze trajectories:

```bash
python src/analyze_trajectories.py
```

Run tests:

```bash
python -m pytest -v
```

For MOT17 evaluation, download the MOT17 training sequence `MOT17-02-FRCNN` and place it under `data/MOT17/train/`. Then run:

```bash
python src/evaluate_mot17.py
```

The evaluation script uses TrackEval under `external/TrackEval/`. Follow the repository's evaluation setup instructions if its legacy dependencies need compatibility adjustments.

## Results

### MOT17-02-FRCNN baseline

Evaluation was performed on **one 600-frame sequence**, not the complete MOT17 benchmark.

| Metric | Result |
|---|---:|
| HOTA | 25.147 |
| IDF1 | 27.452 |
| MOTA | 19.396 |
| ID switches | 59 |
| Track fragmentation events | 105 |

These are baseline results for the current detector and tracker configuration. They are not directly comparable to full-benchmark results reported in research papers.

### Tracking configuration experiments

On the separate sample video:

- Default and balanced configurations produced identical summary statistics.
- The persistent configuration produced fewer unique track IDs, but more detection-gap events.
- These comparisons are diagnostic; the sample video does not have ground-truth tracking annotations.

### Trajectory analytics

The sample video produced 35,369 tracked observations across 514 unique track IDs. The analysis exports per-track movement statistics and visualizations, including speed distribution, trajectory straightness, and a spatial occupancy map.

## Limitations

- Image-space distances and speeds are measured in pixels, not calibrated world units.
- Results depend on detector quality, scene density, occlusion, and tracker configuration.
- Evaluation covers one MOT17 sequence only.
- Track gaps are not equivalent to confirmed identity switches.
- No detector training or fine-tuning is performed.

## Future Improvements

- Evaluate across more MOT17 sequences.
- Compare multiple detectors and tracking algorithms.
- Improve detection recall and identity consistency.
- Add camera-motion compensation and calibrated speed estimation.
- Introduce ground-truth-based trajectory and fragmentation analysis.
