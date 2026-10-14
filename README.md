# Multi-Object Tracking & Trajectory Analysis

[![Project CI](https://github.com/ArwaSaluji01/Multi-Object-Tracking-and-Trajectory-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/ArwaSaluji01/Multi-Object-Tracking-and-Trajectory-Analysis/actions/workflows/ci.yml)

An end-to-end computer vision pipeline for detecting and tracking multiple objects across video frames, analyzing their movement trajectories, comparing tracking configurations, and evaluating tracking predictions against MOT17 ground truth.

## Overview

Object detection identifies objects in individual frames, but multi-object tracking associates detections over time to maintain object identities. This project combines YOLO-based detection with ByteTrack to generate object tracks, record movement trajectories, calculate track-level statistics, visualize results, and evaluate tracking performance.

The project also emphasizes engineering reliability through validated input data, safer output handling, experiment metadata, automated tests, and continuous integration.

## Key Features

- **Object detection and tracking:** YOLO11n with ByteTrack assigns track IDs across video frames.
- **Annotated video output:** Visualizes detected objects, bounding boxes, IDs, and trajectories.
- **Track analytics:** Calculates track duration, observed frames, displacement, path length, and confidence statistics.
- **Trajectory analysis:** Computes movement metrics while accounting for gaps between observed frames.
- **Visualization:** Generates plots for class counts, track durations, path lengths, and selected trajectories.
- **Configuration experiments:** Compares default, balanced, and persistent ByteTrack settings.
- **MOT17 evaluation:** Exports predictions and evaluates one MOT17 training sequence with TrackEval.
- **Input validation:** Checks tracking CSVs and MOT-format files for malformed data, invalid values, and duplicate observations.
- **Safer outputs:** Uses atomic output handling to reduce the risk of leaving incomplete result files.
- **Automated quality checks:** Includes pytest tests, Python compilation checks, dependency validation, and GitHub Actions CI.

## Pipeline Architecture

```text
Input Video
    |
    v
YOLO11n Object Detection
    |
    v
ByteTrack Association
    |
    +----------------------+
    |                      |
    v                      v
Annotated Video       Tracking CSV
                           |
                           v
                    Data Validation
                           |
              +------------+------------+
              |            |            |
              v            v            v
         Track Summary  Trajectory   Visualizations
                        Analytics
              |
              v
       Experiment Comparison

MOT17 Ground Truth + Exported Predictions
                    |
                    v
                 TrackEval
                    |
                    v
          MOT Tracking Metrics
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Object detection | Ultralytics YOLO11n |
| Multi-object tracking | ByteTrack |
| Video processing | OpenCV |
| Data analysis | Pandas, NumPy |
| Numerical analysis | SciPy |
| Visualization | Matplotlib |
| Benchmark evaluation | TrackEval, MOTChallenge format |
| Testing | pytest |
| Continuous integration | GitHub Actions |

## Repository Structure

```text
multi-object-tracking/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── bytetrack_balanced.yaml
│   └── bytetrack_persistent.yaml
├── data/
│   └── input/
│       └── sample.mp4
├── patches/
│   ├── trackeval-12c8791-compat.patch
│   └── trackeval-revision.txt
├── src/
│   ├── track_video.py
│   ├── run_experiments.py
│   ├── analyze_tracks.py
│   ├── analyze_trajectories.py
│   ├── visualize_tracks.py
│   ├── evaluate_mot17.py
│   ├── io_validation.py
│   └── pipeline_utils.py
├── tests/
│   ├── test_evaluate_mot17.py
│   ├── test_io_validation.py
│   ├── test_pipeline_utils.py
│   ├── test_run_experiments.py
│   └── test_trajectory_metrics.py
├── outputs/
│   ├── trajectories/
│   └── benchmark/
├── requirements.txt
├── .gitignore
└── README.md
```

Generated outputs and local dataset files may be excluded from Git. The tree above highlights the principal project components rather than every file.

## Getting Started

### Prerequisites

- Python 3.11
- Git
- A compatible environment for the dependencies pinned in `requirements.txt`
- A video file for tracking

CPU inference is supported. An NVIDIA GPU may improve throughput when a compatible CUDA environment is available.

### 1. Clone the repository

```bash
git clone https://github.com/ArwaSaluji01/Multi-Object-Tracking-and-Trajectory-Analysis.git
cd Multi-Object-Tracking-and-Trajectory-Analysis
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The first inference run may download the configured YOLO model weights if they are not already available locally.

## Usage

Run commands from the repository root with the virtual environment activated. The scripts use their configured input paths and settings; update those settings in the relevant source file or configuration if your local paths differ.

### Run object tracking

```bash
python -m src.track_video
```

The tracking pipeline processes the configured video, runs object detection and tracking, and exports an annotated video and tracking observations.

### Analyze track statistics

```bash
python -m src.analyze_tracks
```

This produces a track summary with statistics such as observed frames, first and last frame, duration, displacement, path length, and mean confidence.

### Run advanced trajectory analysis

```bash
python -m src.analyze_trajectories
```

This calculates additional trajectory metrics and saves analysis outputs and plots.

### Generate visualizations

```bash
python -m src.visualize_tracks
```

The visualization script generates plots for class counts, path lengths, top trajectories, and track durations.

### Compare tracker configurations

```bash
python -m src.run_experiments
```

The experiment runner compares the configured default, balanced, and persistent ByteTrack settings and records tracking statistics and timing metadata. Use the same input video when comparing configurations.

### Evaluate on MOT17

```bash
python -m src.evaluate_mot17
```

This command runs the configured evaluation pipeline for the `MOT17-02-FRCNN` sequence. The MOT17 dataset and the compatible local TrackEval checkout must be available in the expected directory structure.

The evaluation script exports predictions, validates MOT-format data, creates an evaluation manifest, and runs TrackEval. Dataset files and model weights are not necessarily included in this repository.

## Tracking Data Format

The tracking CSV contains one observation per tracked object per frame.

| Column | Description |
|---|---|
| `frame` | Frame number |
| `track_id` | Assigned object identity |
| `class_name` | Predicted object class |
| `confidence` | Detection confidence |
| `x1`, `y1` | Top-left bounding-box coordinates |
| `x2`, `y2` | Bottom-right bounding-box coordinates |
| `center_x`, `center_y` | Bounding-box center coordinates |

The observations are used to calculate track-level statistics and reconstruct object trajectories.

## Results

### Sample-video analysis

The sample-video run produced the following results:

| Measurement | Result |
|---|---:|
| Frames processed | 1,762 |
| Tracking observations | 35,369 |
| Unique track IDs | 514 |
| Person track IDs | 307 |
| Car track IDs | 153 |
| Truck track IDs | 23 |
| Bus track IDs | 13 |

The class-specific counts represent unique track IDs grouped by predicted class, not necessarily unique physical objects. Tracking can assign multiple IDs to the same real-world object if identity is lost and later re-established.

### ByteTrack configuration comparison

The following results were measured on the same sample video. FPS and runtime are environment-dependent.

| Configuration | Observations | Track IDs | Mean observed frames per track | Tracks observed for ≤10 frames | Gap events | Processing FPS |
|---|---:|---:|---:|---:|---:|---:|
| Default | 35,369 | 514 | 68.81 | 171 | 290 | 14.86 |
| Balanced | 34,973 | 449 | 77.89 | 129 | 217 | 14.31 |
| Persistent | 35,467 | 501 | 70.79 | 168 | 319 | 11.67 |

The balanced configuration produced fewer short tracks and fewer measured detection-gap events in this run, while also producing fewer total track IDs. These statistics alone do not establish that one configuration is universally more accurate. A ground-truth evaluation is needed for a stronger comparison.

### MOT17 evaluation

The project was evaluated on **one sequence**, `MOT17-02-FRCNN`, from the MOT17 training split—not the full MOT17 benchmark.

| Metric | Result |
|---|---:|
| HOTA | 25.147 |
| Detection Accuracy (DetA) | 19.928 |
| Association Accuracy (AssA) | 32.041 |
| MOTA | 19.396 |
| MOTP | 79.458 |
| IDF1 | 27.452 |
| ID switches | 59 |
| Ground-truth detections | 18,581 |
| Predicted detections | 5,585 |
| Ground-truth identities | 62 |
| Predicted identities | 106 |

These are baseline results for the specified sequence and configuration. They should not be presented as full-benchmark MOT17 scores or as evidence of state-of-the-art performance.

The evaluation manifest records the sequence, tracker, model, configuration, software versions, and relevant input paths to improve traceability.

## Testing and Code Quality

Run the automated tests:

```bash
python -m pytest tests/ -v
```

Run Python compilation checks:

```bash
python -m compileall -q src tests
```

Check dependency consistency:

```bash
python -m pip check
```

Check whitespace errors in pending Git changes:

```bash
git diff --check
```

The test suite covers trajectory metrics, invalid inputs, atomic output handling, experiment configuration, and MOT17 evaluation data validation. GitHub Actions runs automated checks on pushes and pull requests.

## Engineering Considerations

- Input CSV and MOT-format data are validated before analysis or evaluation.
- Integer ranges, frame numbers, track IDs, duplicate observations, and non-finite values are checked where applicable.
- Atomic output handling helps preserve previous completed outputs when a new operation fails.
- Experiment metadata includes configuration information and timing statistics.
- TrackEval compatibility changes are preserved as a patch and revision record rather than relying only on undocumented local modifications.

## Limitations

- Track IDs can fragment when detections are missed, objects overlap, or the tracker loses an identity.
- Image-space displacement and path length are measured in pixels, not real-world distance.
- Results depend on detector quality, confidence thresholds, tracker settings, video conditions, and hardware.
- The experiment comparison is based on one sample video and is not a general performance guarantee.
- The MOT17 evaluation covers one training sequence. Full-benchmark results require evaluating all required sequences with the appropriate ground truth and sequence maps.
- The current pipeline does not claim calibrated physical speed estimation, future trajectory prediction, or production deployment readiness.

## Future Improvements

- Evaluate more MOT17 sequences and compare configurations using HOTA, MOTA, IDF1, and identity-switch counts.
- Tune detection and association parameters against a consistent validation protocol.
- Investigate identity fragmentation and occlusion-related failures using qualitative video review.
- Add representative demo media and a documented, reproducible command-line configuration.
- Explore trajectory clustering, scene-zone analytics, and calibrated real-world motion estimation where camera geometry is available.

## References

- [Ultralytics documentation](https://docs.ultralytics.com/)
- [ByteTrack paper and project](https://github.com/FoundationVision/ByteTrack)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [MOTChallenge](https://motchallenge.net/)
