import cv2
import os
import numpy as np
from ultralytics import YOLO

# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/cow_video1.mp4"
MODEL_PATH = "yolo11x.pt"        # bigger model = better on hard/unusual angles than yolo11m
OUTPUT_PATH = "output/goat_detection.mp4"

# Animal classes to accept (COCO ids). Top-down packed livestock
# frequently gets misclassified across these - we accept all of them
# and just draw the box, regardless of which label it picked.
ANIMAL_CLASSES = [17, 18, 19, 21]   # horse, sheep, cow, bear

CONFIDENCE = 0.08                   # low on purpose - unusual camera angle
IMG_SIZE = 1280                     # bigger inference size = smaller animals stay visible
DEVICE = 0

# Tiling: split each frame into overlapping tiles and run detection
# on each tile separately, then merge results. This matters a lot
# here because the animals are small relative to the full 1920x1080
# frame once resized down to imgsz, and packed together.
USE_TILING = True
TILE_ROWS = 2
TILE_COLS = 2
TILE_OVERLAP = 0.2                  # 20% overlap between tiles

IOU_MERGE_THRESHOLD = 0.4           # for de-duping boxes across tiles


# ============================================================
# TILING HELPERS
# ============================================================

def make_tiles(frame_w, frame_h, rows, cols, overlap):
    tile_w = frame_w / cols
    tile_h = frame_h / rows
    ov_w = tile_w * overlap
    ov_h = tile_h * overlap

    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tile_w - ov_w))
            y1 = max(0, int(r * tile_h - ov_h))
            x2 = min(frame_w, int((c + 1) * tile_w + ov_w))
            y2 = min(frame_h, int((r + 1) * tile_h + ov_h))
            tiles.append((x1, y1, x2, y2))
    return tiles


def iou(box1, box2):
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def merge_boxes(all_boxes, iou_threshold):
    # all_boxes: list of (x1, y1, x2, y2, conf, cls_id)
    all_boxes = sorted(all_boxes, key=lambda b: b[4], reverse=True)
    kept = []

    for box in all_boxes:
        duplicate = False
        for kept_box in kept:
            if iou(box[:4], kept_box[:4]) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    return kept


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: could not open video:", VIDEO_PATH)
    exit()

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_fps, (video_width, video_height))

tiles = make_tiles(video_width, video_height, TILE_ROWS, TILE_COLS, TILE_OVERLAP) if USE_TILING else None

frame_number = 0
max_detections_seen = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_number += 1

    all_boxes = []

    if USE_TILING:
        for (tx1, ty1, tx2, ty2) in tiles:
            tile_img = frame[ty1:ty2, tx1:tx2]

            results = model.predict(
                tile_img,
                conf=CONFIDENCE,
                imgsz=IMG_SIZE,
                classes=ANIMAL_CLASSES,
                device=DEVICE,
                agnostic_nms=True,
                verbose=False,
            )
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)

                for (bx1, by1, bx2, by2), conf, cls_id in zip(boxes, confs, clss):
                    # shift tile-local coords back to full-frame coords
                    all_boxes.append((
                        bx1 + tx1, by1 + ty1,
                        bx2 + tx1, by2 + ty1,
                        float(conf), int(cls_id)
                    ))
    else:
        results = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            classes=ANIMAL_CLASSES,
            device=DEVICE,
            agnostic_nms=True,
            verbose=False,
        )
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for (bx1, by1, bx2, by2), conf, cls_id in zip(boxes, confs, clss):
                all_boxes.append((bx1, by1, bx2, by2, float(conf), int(cls_id)))

    merged = merge_boxes(all_boxes, IOU_MERGE_THRESHOLD)
    max_detections_seen = max(max_detections_seen, len(merged))

    class_names = model.names
    for (x1, y1, x2, y2, conf, cls_id) in merged:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Detections: {len(merged)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    writer.write(frame)

    if frame_number % 30 == 0:
        print(f"frame {frame_number} | detections this frame: {len(merged)}")

cap.release()
writer.release()

print()
print("Done.")
print("Max simultaneous detections seen in any frame:", max_detections_seen)
print("Output video:", OUTPUT_PATH)

if max_detections_seen == 0:
    print()
    print("Still zero detections across the whole video with a bigger model,")
    print("lower confidence, tiling, and a wider class list. That's a strong")
    print("signal the stock COCO model genuinely cannot see this camera angle,")
    print("and fine-tuning on your own labeled frames is the fix, not further")
    print("threshold tuning.")
