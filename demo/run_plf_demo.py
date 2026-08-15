from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

VIDEO_PATH = (
    PROJECT_ROOT
    / "videos"
    / "cow_video10.mp4"
)

YOLO_MODEL = (
    PROJECT_ROOT
    / "models"
    / "yolo26m-seg.pt"
)

BEHAVIOR_MODEL = (
    PROJECT_ROOT
    / "output"
    / "behavior"
    / "model"
    / "best_behavior_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "demo"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "cow_plf_demo.mp4"
)

EVENTS_CSV = (
    OUTPUT_DIR
    / "cow_plf_demo_events.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "cow_plf_cow_summary.csv"
)


# ============================================================
# CONFIG
# ============================================================

CONFIDENCE = 0.40
IOU = 0.50

IMAGE_SIZE = 224

BEHAVIORS = [
    "standing",
    "lying",
    "foraging",
    "drinking",
    "rumination",
]

SMOOTHING_WINDOW = 12

# Save one event observation per cow every N seconds.
EVENT_INTERVAL_SEC = 0.50

COW_CLASS_ID = 19


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

behavior_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# LOAD BEHAVIOR MODEL
# ============================================================

def load_behavior_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(
            p=0.25
        ),
        torch.nn.Linear(
            model.fc.in_features,
            len(BEHAVIORS),
        ),
    )

    checkpoint = torch.load(
        BEHAVIOR_MODEL,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        DEVICE
    )

    model.eval()

    threshold = float(
        checkpoint.get(
            "threshold",
            0.40,
        )
    )

    return model, threshold


# ============================================================
# SAFE CROP
# ============================================================

def safe_crop(
    frame,
    x1,
    y1,
    x2,
    y2,
):

    height, width = frame.shape[:2]

    x1 = max(
        0,
        min(
            int(x1),
            width - 1,
        ),
    )

    y1 = max(
        0,
        min(
            int(y1),
            height - 1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            int(x2),
            width,
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            int(y2),
            height,
        ),
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# BEHAVIOR PREDICTION
# ============================================================

@torch.no_grad()
def predict_behavior(
    model,
    crop,
):

    if crop is None:
        return None

    if crop.size == 0:
        return None

    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(
        rgb
    )

    tensor = behavior_transform(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    logits = model(
        tensor
    )

    probabilities = torch.sigmoid(
        logits
    )[0].cpu().numpy()

    return probabilities


# ============================================================
# MASK MORPHOMETRICS
# ============================================================

def extract_mask_features(
    mask,
):

    if mask is None:
        return None

    binary = (
        mask > 0.5
    ).astype(
        np.uint8
    )

    ys, xs = np.where(
        binary > 0
    )

    if len(xs) < 20:
        return None

    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    )

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    )

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    body_length = max(
        width,
        height,
    )

    body_depth = min(
        width,
        height,
    )

    area = int(
        binary.sum()
    )

    return {
        "body_length_px": body_length,
        "body_depth_px": body_depth,
        "mask_area_px": area,
        "bbox_width_px": width,
        "bbox_height_px": height,
    }


# ============================================================
# TEXT
# ============================================================

def draw_text(
    frame,
    text,
    x,
    y,
    scale=0.55,
    thickness=2,
):

    cv2.putText(
        frame,
        str(text),
        (
            int(x),
            int(y),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("COW PLF - COMPANY DEMO")
    print("=" * 80)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    if not YOLO_MODEL.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{YOLO_MODEL}"
        )

    if not BEHAVIOR_MODEL.exists():
        raise FileNotFoundError(
            f"Behavior model not found:\n{BEHAVIOR_MODEL}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print()
    print(
        "[MODEL] Loading YOLO26..."
    )

    yolo = YOLO(
        str(YOLO_MODEL)
    )

    print(
        "[MODEL] Loading behavior model..."
    )

    behavior_model, behavior_threshold = (
        load_behavior_model()
    )

    print(
        f"[DEVICE] {DEVICE}"
    )

    print(
        f"[BEHAVIOR] Threshold: "
        f"{behavior_threshold:.2f}"
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open:\n{VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration_sec = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print()
    print(
        f"[VIDEO] Resolution: "
        f"{width} x {height}"
    )

    print(
        f"[VIDEO] FPS: {fps:.2f}"
    )

    print(
        f"[VIDEO] Frames: {total_frames}"
    )

    print(
        f"[VIDEO] Duration: "
        f"{duration_sec:.1f} sec"
    )

    # --------------------------------------------------------
    # Writer
    # --------------------------------------------------------

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create:\n{OUTPUT_VIDEO}"
        )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    behavior_history = defaultdict(
        list
    )

    event_last_time = {}

    event_rows = []

    # Observation history used for cow summary.
    cow_observations = defaultdict(
        list
    )

    frame_number = 0

    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        timestamp_sec = (
            frame_number - 1
        ) / fps

        results = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            iou=IOU,
            classes=[
                COW_CLASS_ID
            ],
            verbose=False,
        )

        result = results[0]

        detection_count = 0

        behavior_counter = Counter()

        # ----------------------------------------------------
        # Detections
        # ----------------------------------------------------

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            boxes = result.boxes

            masks = (
                result.masks.data
                if result.masks is not None
                else None
            )

            for index in range(
                len(boxes)
            ):

                detection_confidence = (
                    float(
                        boxes.conf[index]
                        .cpu()
                        .item()
                    )
                )

                if (
                    detection_confidence
                    < CONFIDENCE
                ):
                    continue

                # ------------------------------------------------
                # Track ID
                # ------------------------------------------------

                if boxes.id is not None:

                    track_id = int(
                        boxes.id[index]
                        .cpu()
                        .item()
                    )

                else:

                    track_id = (
                        index + 1
                    )

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                x1, y1, x2, y2 = (
                    boxes.xyxy[index]
                    .cpu()
                    .numpy()
                )

                crop = safe_crop(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                )

                probabilities = (
                    predict_behavior(
                        behavior_model,
                        crop,
                    )
                )

                if probabilities is None:
                    continue

                # ------------------------------------------------
                # Temporal smoothing
                # ------------------------------------------------

                history = (
                    behavior_history[
                        track_id
                    ]
                )

                history.append(
                    probabilities
                )

                if len(history) > SMOOTHING_WINDOW:

                    del history[
                        :-
                        SMOOTHING_WINDOW
                    ]

                smoothed = np.mean(
                    history,
                    axis=0,
                )

                # ------------------------------------------------
                # Primary behavior
                # ------------------------------------------------

                primary_index = int(
                    np.argmax(
                        smoothed
                    )
                )

                primary_behavior = (
                    BEHAVIORS[
                        primary_index
                    ]
                )

                primary_confidence = (
                    float(
                        smoothed[
                            primary_index
                        ]
                    )
                )

                behavior_counter[
                    primary_behavior
                ] += 1

                # ------------------------------------------------
                # Active behaviors
                # ------------------------------------------------

                active_behaviors = []

                for behavior_index, behavior in enumerate(
                    BEHAVIORS
                ):

                    probability = float(
                        smoothed[
                            behavior_index
                        ]
                    )

                    if (
                        probability
                        >= behavior_threshold
                    ):

                        active_behaviors.append(
                            behavior
                        )

                # ------------------------------------------------
                # Morphometrics from YOLO mask
                # ------------------------------------------------

                morph = None

                if (
                    masks is not None
                    and index < len(masks)
                ):

                    mask = (
                        masks[index]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    morph = (
                        extract_mask_features(
                            mask
                        )
                    )

                # ------------------------------------------------
                # Event logging
                #
                # We record an observation every 0.5 sec per
                # tracked cow instead of creating a row for
                # every single video frame.
                # ------------------------------------------------

                previous_event_time = (
                    event_last_time.get(
                        track_id,
                        -1e9,
                    )
                )

                if (
                    timestamp_sec
                    - previous_event_time
                    >= EVENT_INTERVAL_SEC
                ):

                    event_last_time[
                        track_id
                    ] = timestamp_sec

                    event = {
                        "timestamp_sec":
                            round(
                                timestamp_sec,
                                2,
                            ),

                        "frame_number":
                            frame_number,

                        "track_id":
                            track_id,

                        "behavior":
                            primary_behavior.upper(),

                        "behavior_confidence":
                            round(
                                primary_confidence,
                                3,
                            ),

                        "active_behaviors":
                            "|".join(
                                active_behaviors
                            ),

                        "detection_confidence":
                            round(
                                detection_confidence,
                                3,
                            ),

                        "body_length_px":
                            (
                                morph[
                                    "body_length_px"
                                ]
                                if morph
                                else None
                            ),

                        "body_depth_px":
                            (
                                morph[
                                    "body_depth_px"
                                ]
                                if morph
                                else None
                            ),

                        "mask_area_px":
                            (
                                morph[
                                    "mask_area_px"
                                ]
                                if morph
                                else None
                            ),
                    }

                    event_rows.append(
                        event
                    )

                # ------------------------------------------------
                # Summary observation
                # ------------------------------------------------

                cow_observations[
                    track_id
                ].append(
                    {
                        "timestamp_sec":
                            timestamp_sec,

                        "behavior":
                            primary_behavior,

                        "confidence":
                            primary_confidence,

                        "body_length_px":
                            (
                                morph[
                                    "body_length_px"
                                ]
                                if morph
                                else np.nan
                            ),

                        "body_depth_px":
                            (
                                morph[
                                    "body_depth_px"
                                ]
                                if morph
                                else np.nan
                            ),
                    }
                )

                # ------------------------------------------------
                # Draw cow box
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (
                        int(x1),
                        int(y1),
                    ),
                    (
                        int(x2),
                        int(y2),
                    ),
                    (0, 255, 0),
                    2,
                )

                # ------------------------------------------------
                # Cow overlay
                # ------------------------------------------------

                text_x = int(
                    x1
                )

                text_y = max(
                    25,
                    int(y1) - 10,
                )

                draw_text(
                    frame,
                    f"COW {track_id}",
                    text_x,
                    text_y,
                    scale=0.60,
                )

                draw_text(
                    frame,
                    (
                        f"{primary_behavior.upper()} "
                        f"{primary_confidence:.2f}"
                    ),
                    text_x,
                    text_y + 24,
                    scale=0.46,
                )

                if morph:

                    draw_text(
                        frame,
                        (
                            f"L:{morph['body_length_px']} "
                            f"D:{morph['body_depth_px']}"
                        ),
                        text_x,
                        text_y + 44,
                        scale=0.42,
                    )

                detection_count += 1

        # --------------------------------------------------------
        # DASHBOARD
        # --------------------------------------------------------

        dashboard_width = 380

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (
                width - dashboard_width,
                0,
            ),
            (
                width,
                height,
            ),
            (20, 20, 20),
            -1,
        )

        frame = cv2.addWeighted(
            overlay,
            0.72,
            frame,
            0.28,
            0,
        )

        dashboard_x = (
            width
            - dashboard_width
            + 20
        )

        draw_text(
            frame,
            "COW PLF AI",
            dashboard_x,
            42,
            scale=0.85,
        )

        draw_text(
            frame,
            "PRECISION LIVESTOCK INTELLIGENCE",
            dashboard_x,
            70,
            scale=0.38,
            thickness=1,
        )

        draw_text(
            frame,
            f"Cows detected: {detection_count}",
            dashboard_x,
            112,
            scale=0.50,
        )

        draw_text(
            frame,
            f"Time: {timestamp_sec:07.1f}s",
            dashboard_x,
            138,
            scale=0.46,
        )

        # ----------------------------------------------------
        # Current behaviors
        # ----------------------------------------------------

        draw_text(
            frame,
            "CURRENT BEHAVIOR",
            dashboard_x,
            185,
            scale=0.60,
        )

        y_dashboard = 215

        for behavior in BEHAVIORS:

            count = (
                behavior_counter[
                    behavior
                ]
            )

            draw_text(
                frame,
                f"{behavior.title():12s} {count}",
                dashboard_x,
                y_dashboard,
                scale=0.45,
            )

            y_dashboard += 26

        # ----------------------------------------------------
        # System status
        # ----------------------------------------------------

        draw_text(
            frame,
            "SYSTEM STATUS",
            dashboard_x,
            y_dashboard + 18,
            scale=0.60,
        )

        status_y = (
            y_dashboard
            + 48
        )

        statuses = [
            "YOLO26 Detection       READY",
            "ByteTrack Tracking     READY",
            "Behavior AI            READY",
            "Morphometrics          READY",
            "BCS Engine             PROVISIONAL",
        ]

        for text in statuses:

            draw_text(
                frame,
                text,
                dashboard_x,
                status_y,
                scale=0.38,
            )

            status_y += 25

        # ----------------------------------------------------
        # Bottom information
        # ----------------------------------------------------

        draw_text(
            frame,
            (
                "Company Prototype Demo"
            ),
            20,
            height - 20,
            scale=0.43,
        )

        writer.write(
            frame
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            frame_number % 100
            == 0
        ):

            progress = (
                frame_number
                / max(
                    1,
                    total_frames,
                )
                * 100
            )

            print(
                f"[VIDEO] "
                f"{frame_number}/{total_frames} "
                f"({progress:.1f}%)"
            )

    # ========================================================
    # RELEASE VIDEO
    # ========================================================

    cap.release()
    writer.release()

    # ========================================================
    # EVENT CSV
    # ========================================================

    events_df = pd.DataFrame(
        event_rows
    )

    if events_df.empty:

        print(
            "[WARNING] No behavior events generated."
        )

        events_df = pd.DataFrame(
            columns=[
                "timestamp_sec",
                "frame_number",
                "track_id",
                "behavior",
                "behavior_confidence",
                "active_behaviors",
                "detection_confidence",
                "body_length_px",
                "body_depth_px",
                "mask_area_px",
            ]
        )

    events_df.to_csv(
        EVENTS_CSV,
        index=False,
    )

    # ========================================================
    # PER-COW SUMMARY
    # ========================================================

    summary_rows = []

    for track_id, observations in sorted(
        cow_observations.items()
    ):

        if not observations:
            continue

        observation_count = len(
            observations
        )

        behavior_counts = Counter(
            observation[
                "behavior"
            ]
            for observation in observations
        )

        first_time = min(
            observation[
                "timestamp_sec"
            ]
            for observation in observations
        )

        last_time = max(
            observation[
                "timestamp_sec"
            ]
            for observation in observations
        )

        tracked_seconds = max(
            0.0,
            last_time - first_time,
        )

        row = {
            "track_id":
                track_id,

            "first_seen_sec":
                round(
                    first_time,
                    2,
                ),

            "last_seen_sec":
                round(
                    last_time,
                    2,
                ),

            "tracked_seconds":
                round(
                    tracked_seconds,
                    2,
                ),

            "observations":
                observation_count,
        }

        # ----------------------------------------------------
        # Behavior percentages
        # ----------------------------------------------------

        for behavior in BEHAVIORS:

            count = (
                behavior_counts[
                    behavior
                ]
            )

            percentage = (
                count
                / observation_count
                * 100.0
            )

            row[
                f"{behavior}_pct"
            ] = round(
                percentage,
                2,
            )

        # ----------------------------------------------------
        # Dominant behavior
        # ----------------------------------------------------

        dominant_behavior = (
            behavior_counts.most_common(
                1
            )[0][0]
        )

        row[
            "dominant_behavior"
        ] = dominant_behavior.upper()

        # ----------------------------------------------------
        # Average confidence
        # ----------------------------------------------------

        mean_confidence = np.mean(
            [
                observation[
                    "confidence"
                ]
                for observation in observations
            ]
        )

        row[
            "mean_behavior_confidence"
        ] = round(
            float(
                mean_confidence
            ),
            3,
        )

        # ----------------------------------------------------
        # Average morphometrics
        # ----------------------------------------------------

        length_values = [
            observation[
                "body_length_px"
            ]
            for observation in observations
            if np.isfinite(
                observation[
                    "body_length_px"
                ]
            )
        ]

        depth_values = [
            observation[
                "body_depth_px"
            ]
            for observation in observations
            if np.isfinite(
                observation[
                    "body_depth_px"
                ]
            )
        ]

        row[
            "mean_body_length_px"
        ] = round(
            float(
                np.mean(
                    length_values
                )
            ),
            2,
        ) if length_values else None

        row[
            "mean_body_depth_px"
        ] = round(
            float(
                np.mean(
                    depth_values
                )
            ),
            2,
        ) if depth_values else None

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("COW PLF DEMO COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Video:"
    )

    print(
        OUTPUT_VIDEO
    )

    print()
    print(
        f"Event CSV:"
    )

    print(
        EVENTS_CSV
    )

    print()
    print(
        f"Cow Summary CSV:"
    )

    print(
        SUMMARY_CSV
    )

    print()
    print(
        f"Event rows : {len(events_df)}"
    )

    print(
        f"Cow rows   : {len(summary_df)}"
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "track_id represents the tracker identity in this demo."
    )

    print(
        "BCS is intentionally not attached to video track IDs."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()