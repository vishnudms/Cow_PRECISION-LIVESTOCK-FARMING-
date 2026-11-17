<div align="center">

# Cow PLF: Precision Livestock Farming & Computer-Vision Platform

[![GitHub Repo](https://img.shields.io/badge/GitHub-vishnudms%2FCow__PRECISION--LIVESTOCK--FARMING--blue?logo=github)](https://github.com/vishnudms/Cow_PRECISION-LIVESTOCK-FARMING-)
[![Perception](https://img.shields.io/badge/Perception-YOLO26m--seg%20%7C%20ByteTrack-orange)](#-current-vision-stack)
[![Identity](https://img.shields.io/badge/Re--Identification-V5%20Permanent%20ID%20%7C%20Global%20Registry-blue)](#-3-permanent-identity--cross-day-registry)
[![Biometrics](https://img.shields.io/badge/Biometrics-Morphometrics%20%7C%20Contactless%20BCS-green)](#-4-biometric-morphometrics--bcs)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#-license)

**An enterprise-grade, edge-ready computer-vision platform for precision dairy and beef cattle management. Combines deep instance segmentation, multi-target tracking, permanent coat-pattern re-identification, contactless morphometrics, and automated Body Condition Scoring (BCS).**

[Overview](#-1-project-overview) • [Vision Stack](#-2-vision-stack--pipeline) • [Identity Manager](#-3-permanent-identity--cross-day-registry) • [Biometrics & BCS](#-4-biometric-morphometrics--bcs) • [Tech Stack](#-5-technology-stack) • [Repository Structure](#-6-repository-structure) • [Quick Start](#-7-quick-start) • [References](#-8-references--literature)

</div>

---

## 📖 1. Project Overview

Modern dairy and feedlot operations face severe challenges in tracking individual animal health and productivity:
- **Ear-Tag Vulnerability:** Physical RFID and barcode tags frequently snag, tear, or become caked in mud, causing identity loss.
- **Subjective Body Condition Scoring (BCS):** Manual palpation and visual scoring are labor-intensive, stressful for cattle, and introduce high inter-observer variability.
- **Herd Crowding & Occlusion:** Dense pen environments confound standard single-frame tracking when cows cross paths or enter feeding lanes.

**Cow PLF** resolves these bottlenecks by combining deep instance segmentation (**YOLO26m-seg**), robust multi-object tracking (**[ByteTrack](https://github.com/ifzhang/ByteTrack)**), and an invariant **coat-pattern biometric appearance embedder** to establish permanent, cross-day digital identities alongside automated morphological measurement.

---

## 🔬 2. Vision Stack & Pipeline

```
                              ┌────────────────────────────────────────┐
                              │ Video Stream / CCTV Farm Camera Feed   │
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │    YOLO26m-seg Instance Segmentation   │
                              │    (Extracts Cow Bounding Box & Mask)  │
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │    ByteTrack Multi-Object Tracker      │
                              │    (Temporal Frame-to-Frame Trajectory)│
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │    AppearanceExtractor Embeddings      │
                              │    (Coat Pattern & Texture Descriptors)│
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │    V5 IdentityManager                  │
                              │    (Within-Video ID Stabilization)     │
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │    Global Cow Identity Registry        │
                              │    (Permanent Cross-Day / Multi-Camera)│
                              └───────────────────┬────────────────────┘
                                                  │
                        ┌─────────────────────────┴─────────────────────────┐
                        ▼                                                   ▼
         ┌───────────────────────────────┐                   ┌───────────────────────────────┐
         │ Contactless Morphometrics     │                   │ Body Condition Scoring (BCS)  │
         ├───────────────────────────────┤                   ├───────────────────────────────┤
         │ • Body Length Measurement     │                   │ • Feature Map Extraction      │
         │ • View Angle Filter (Profile) │                   │ • Dryad-Trained BCS Engine    │
         │ • CSV Telemetry Export        │                   │ • Fat Reserve Stratification  │
         └──────────────┬────────────────┘                   └──────────────┬────────────────┘
                        │                                                   │
                        └─────────────────────────┬─────────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │ Production Telemetry & Outputs         │
                              │ • Real-Time Annotated Video (.mp4)     │
                              │ • Per-Cow Measurement Logs (.csv)      │
                              └────────────────────────────────────────┘
```

---

## 🏷️ 3. Permanent Identity & Cross-Day Registry

Traditional trackers lose identity when animals leave the field of view or undergo severe occlusion. Cow PLF employs a two-tier identity framework:

1. **V5 IdentityManager (Within-Video Authority):**
   - Continuously fuses bounding-box spatial affinity with deep appearance embeddings.
   - Smooths tracklet fragmentation caused by sudden turns, head dips, or partial occlusions.
2. **Global Cow Identity Registry (Cross-Day / Cross-Camera):**
   - Matches distinctive natural coat pigmentation, patch geometries, and dorsal patterns.
   - Assigns a permanent **Global Cow ID** persisted across recording sessions, enabling longitudinal health monitoring without physical tagging.

---

## 📐 4. Biometric Morphometrics & BCS

- **Contactless Morphometrics:**
  - Extracts principal body contours along anatomical landmarks.
  - Automatically filters for valid side-profile orientations (`view_angle.py`) before logging body length, ensuring perspective distortions do not bias measurements.
- **Contactless Body Condition Scoring (BCS):**
  - Machine-learning pipeline trained on peer-reviewed cattle datasets (including Dryad).
  - Isolates dorsal and pelvic skeletal regions (hooks, pins, spine) to score subcutaneous body fat on standard 1–5 veterinary scales.
- **Behavior & Welfare Monitoring (In-Development):**
  - Posture and activity ethograms (standing, lying, rumination, feeding).
  - Social density analysis and locomotion scoring.

---

## 🛠️ 5. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | [Ultralytics YOLO (v8 / v11 / YOLO26m-seg)](https://github.com/ultralytics/ultralytics) | Cow detection, anatomical bounding boxes, and instance segmentation masks |
| **Tracking Engine** | [ByteTrack](https://github.com/ifzhang/ByteTrack) | Associative multi-target bounding-box tracking and tracklet smoothing |
| **Computer Vision** | [OpenCV (opencv-python)](https://opencv.org/) | Video stream decoding, image-space contour extraction, and HUD visualization |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) | Feature extraction, dimensional reduction, and BCS regression estimators |
| **Data Processing** | [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) | Telemetry serialization, morphological logging, and vector math |
| **Configuration** | [PyYAML](https://pyyaml.org/) | Pipeline hyperparameters, detector thresholds, and camera configuration |

---

## 📂 6. Repository Structure

```text
Cow_PRECISION-LIVESTOCK-FARMING-/
├── src/
│   └── cow_plf/                         # Core modular production package
│       ├── analytics/                   # Morphometrics, view-angle filtering & analytics
│       │   ├── morphometrics.py         # Body length & image-space dimension estimators
│       │   └── view_angle.py            # Side-profile validation filter
│       ├── core/                        # Biometrics & identity infrastructure
│       │   ├── appearance.py            # Deep appearance extractor & coat descriptors
│       │   ├── cow_profile.py           # Per-cow longitudinal telemetry state
│       │   └── identity_manager.py      # V5 local identity association manager
│       ├── identity/                    # Global registry & cross-day re-ID
│       │   └── global_identity.py       # Cross-video permanent cow identity mapper
│       ├── vision/                      # Computer vision & neural inferencing
│       │   └── detector.py              # YOLO26m-seg inference wrapper
│       └── pipeline.py                  # End-to-end unified production pipeline
│
├── bcs/                                 # Body Condition Scoring module
│   ├── bcs_engine.py                    # Inference engine for non-contact BCS scoring
│   ├── estimate_bcs.py                  # Standalone BCS estimation CLI
│   ├── train_bcs_model.py               # Model training script on labeled datasets
│   └── BCS_LABELING_PROTOCOL.txt        # Veterinary annotation protocol documentation
│
├── behavior/                            # Cattle behavior classification & ethograms
│   ├── build_behavior_dataset.py        # Posture & activity dataset generator
│   └── train_behavior_model.py          # Classifier training for rumination/feeding/lying
│
├── morphometrics/                       # Morphological dataset builders & calculators
│   ├── build_dataset.py                 # Ground-truth paired dataset processor
│   └── build_master_dataset.py          # Aggregator for dimensional telemetry
│
├── tests/                               # Comprehensive unit & regression test suite
│   ├── test_appearance.py               # Invariance tests for coat pattern embeddings
│   ├── test_detector.py                 # Segmentation & bounding-box validation
│   ├── test_identity_manager.py         # Tracklet association unit tests
│   └── test_v5_identity_regression.py   # Identity persistence regression tests
│
├── main.py                              # Primary CLI production entry point
├── run_production.py                    # Automated multi-video production runner
├── cow_detection_v2.py                  # Standalone detection utility script
├── cow_length_measurement_v1.py         # Standalone length measurement prototype
├── requirements.txt                     # Pinned Python package dependencies
└── README.md                            # Comprehensive system documentation
```

---

## ⚡ 7. Quick Start

### 1. Environment Setup
Clone the repository and set up a Python 3.12 virtual environment:
```bash
git clone https://github.com/vishnudms/Cow_PRECISION-LIVESTOCK-FARMING-.git
cd Cow_PRECISION-LIVESTOCK-FARMING-

python -m venv venv
.\venv\Scripts\Activate.ps1       # On Linux/macOS: source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Production Video Analysis
Process a cattle pen video through the full segmentation, tracking, and identity pipeline:
```bash
# Set PYTHONPATH to include the src directory
$env:PYTHONPATH="src"             # On Linux/macOS: export PYTHONPATH=src

# Run main inference pipeline
python main.py --video path/to/cow_video.mp4
```

Outputs are automatically saved to `output/`:
- `annotated_<video_name>.mp4`: Processed video with masks, bounding boxes, Local IDs, and Global IDs.
- `measurements_<video_name>.csv`: Frame-by-frame morphological dimensions and identity logs.

### 3. Run BCS Estimation
```bash
python bcs/estimate_bcs.py --image path/to/cow_dorsal.jpg
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```

---

## 📋 8. Implementation Status & Roadmap

- [x] YOLO instance segmentation integration with robust mask extraction.
- [x] ByteTrack multi-target associative tracking.
- [x] V5 Permanent Identity Manager for within-video continuity.
- [x] Global Cow Identity Registry for cross-day re-identification.
- [x] Contactless image-space morphometrics & view-angle filtering.
- [x] Body Condition Scoring (BCS) training and evaluation engine.
- [ ] Metric-scale stereoscopic camera calibration for 3D measurements (cm).
- [ ] Automated lameness detection and locomotion scoring.
- [ ] Real-time RTSP multi-camera streaming ingestion.

---

## 📚 9. References & Literature

1. **ByteTrack Multi-Object Tracking:**  
   Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.  
   📄 [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) • 💻 [GitHub: ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)

2. **Precision Livestock Farming (PLF) Principles:**  
   Berckmans, D. (2014). *Precision livestock farming technologies for welfare, health and production control of animals.* Veterinary Record, 175(25), 648–653.

3. **Automated Body Condition Scoring in Cattle:**  
   Spoliansky, R., et al. (2016). *Development of automatic body condition scoring system for dairy cows using 3D cameras.* Computers and Electronics in Agriculture, 128, 97–103.

4. **Ultralytics YOLO Architecture:**  
   Jocher, G., Qiu, J., et al. (2024). *Ultralytics Real-Time Computer Vision.*  
   💻 [GitHub: Ultralytics](https://github.com/ultralytics/ultralytics)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
