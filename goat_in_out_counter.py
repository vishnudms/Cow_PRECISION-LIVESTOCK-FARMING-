"""
===============================================================================
DIAGNOSTIC: WHY IS THE DETECTOR RETURNING 0 GOATS?
===============================================================================
This bypasses ALL filtering (class list, confidence threshold) so we can see
literally everything YOLO thinks might be in the frame, at any confidence,
of any class. This tells us which of these is actually happening:

  A) YOLO sees goat-shaped things but scores them very low / wrong class
     (e.g. calls them "horse", "dog", "bird", or nothing above ~0.01)
     -> fixable by lowering conf further, and confirms a fine-tune is
        needed for reliable production use.

  B) YOLO detects almost NOTHING in the frame at all (empty results even
     at conf=0.001) -> points to a loading/frame/path problem rather
     than a detection-quality problem. Worth ruling out first.

  C) The model file loaded is not what you think it is (wrong path,
     corrupted download, etc).

USAGE
-----
    python diagnose_detection.py --source D:\\COW\\videos\\cow_video1.mp4 --frame 950

    --frame is the frame NUMBER to test on. Pick one you know visually has
    goats in it (scrub through the video in VLC/media player and note the
    frame or timestamp first).
===============================================================================
"""

import argparse
import os

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video path")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to test (0-indexed)")
    parser.add_argument("--model", default=None, help="Model path (defaults to searching D:\\COW)")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    # ---- find model -----------------------------------------------------
    model_path = args.model
    if model_path is None:
        for candidate in [r"D:\COW\models\yolo11m.pt", r"D:\COW\yolo11m.pt"]:
            if os.path.exists(candidate):
                model_path = candidate
                break
    if model_path is None or not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        return

    print(f"[MODEL] Loading: {model_path}")
    model = YOLO(model_path)
    print(f"[MODEL] Loaded OK. This model knows {len(model.names)} classes.")
    print(f"[MODEL] Class 18 = {model.names.get(18)}   Class 19 = {model.names.get(19)}")

    # ---- grab the requested frame ---------------------------------------
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.source}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= total:
        print(f"[ERROR] Frame {args.frame} out of range (video has {total} frames).")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[ERROR] Could not read that frame.")
        return
    print(f"[VIDEO] Frame {args.frame} loaded, shape={frame.shape}")

    cv2.imwrite("diag_input_frame.jpg", frame)
    print("[SAVED] diag_input_frame.jpg  <- open this and confirm it actually shows goats")

    # ---- run with NO filtering at all ------------------------------------
    print()
    print("=" * 70)
    print("RUNNING YOLO WITH conf=0.001, NO CLASS FILTER, imgsz=" + str(args.imgsz))
    print("=" * 70)

    results = model.predict(
        source=frame,
        conf=0.001,       # essentially "show me everything"
        iou=0.45,
        imgsz=args.imgsz,
        classes=None,      # no class restriction at all
        verbose=False,
    )
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print()
        print("!!! ZERO detections even at conf=0.001 with no class filter !!!")
        print("This points to (B) or (C) above — not a confidence-tuning issue.")
        print("Check:")
        print("  1. Open diag_input_frame.jpg — does it actually show goats?")
        print("     If it's blank/black/wrong content, the --frame number or")
        print("     video path is off, not the model.")
        print("  2. Confirm model_path above is really yolo11m.pt and not a")
        print("     0-byte / corrupted / partial download (check file size).")
        print("  3. Try imgsz=1920 or even the raw frame size, in case objects")
        print("     are being lost to downscaling before detection.")
    else:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        print(f"\n[RESULT] {len(boxes)} raw detections found (any class, any confidence).")
        print(f"{'CLASS ID':>8} {'CLASS NAME':<15} {'CONFIDENCE':>10}  {'BOX (x1,y1,x2,y2)'}")
        print("-" * 70)

        order = confs.argsort()[::-1]  # highest confidence first
        for i in order:
            cid = int(classes[i])
            cname = model.names.get(cid, "?")
            conf = confs[i]
            x1, y1, x2, y2 = boxes[i]
            print(f"{cid:>8} {cname:<15} {conf:>10.4f}  ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        # Annotated save so you can SEE what it's calling everything.
        annotated = result.plot()
        cv2.imwrite("diag_annotated_frame.jpg", annotated)
        print("\n[SAVED] diag_annotated_frame.jpg  <- open this to see every box YOLO drew")

        goat_dets = [i for i in order if int(classes[i]) == 18]
        cow_dets = [i for i in order if int(classes[i]) == 19]
        print(f"\nOf these: {len(goat_dets)} were class 18 (sheep/goat-proxy), "
              f"{len(cow_dets)} were class 19 (cow).")
        if goat_dets:
            print(f"Highest goat-class confidence seen: {confs[goat_dets[0]]:.4f}")
        else:
            print("No detections were ever classified as sheep/cow at all — "
                  "YOLO is likely calling the animals something else entirely "
                  "(check the class names printed above), or not seeing them "
                  "as objects at all.")


if __name__ == "__main__":
    main()
