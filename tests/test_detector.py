import sys
from pathlib import Path

import cv2


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from cow_plf.vision.detector import CowDetector


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo26m-seg.pt"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "videos"
    / "cow_video6.mp4"
)


def main():

    print()
    print("=" * 70)
    print("PRODUCTION VISION MODULE TEST")
    print("=" * 70)

    detector = CowDetector(
        model_path=MODEL_PATH,
        confidence=0.20,
        iou=0.50,
        image_size=1280,
        device=0,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        print(
            "[ERROR] Could not open video:"
        )

        print(
            VIDEO_PATH
        )

        return

    frame_number = 0

    max_frames = 100

    try:

        while frame_number < max_frames:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            detections = detector.predict(
                frame
            )

            print(
                f"Frame {frame_number:03d}: "
                f"{len(detections)} detections"
            )

            for detection in detections:

                mask_pixels = 0

                if detection.mask is not None:

                    mask_pixels = int(
                        detection.mask.sum()
                    )

                print(
                    f"  "
                    f"Tracker={detection.tracker_id} "
                    f""
                    f"Conf={detection.confidence:.3f} "
                    f""
                    f"Box={detection.box} "
                    f""
                    f"Mask={mask_pixels}"
                )

            cv2.imshow(
                "Production Vision Test",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print(
        "PRODUCTION VISION TEST COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
