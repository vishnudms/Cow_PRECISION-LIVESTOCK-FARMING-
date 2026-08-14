import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

"""
============================================================
APPEARANCE EXTRACTOR TEST
============================================================
"""

import cv2

from cow_plf.core.appearance import AppearanceExtractor


def main():

    print()
    print("=" * 60)
    print("COW APPEARANCE EXTRACTOR TEST")
    print("=" * 60)
    print()

    extractor = AppearanceExtractor()

    # --------------------------------------------------------
    # Load a frame from the cow video
    # --------------------------------------------------------

    video_path = (
        "videos\\cow_video6.mp4"
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "ERROR: Cannot open video"
        )

        return

    success, frame = cap.read()

    cap.release()

    if not success:

        print(
            "ERROR: Cannot read frame"
        )

        return

    print(
        f"Frame shape: {frame.shape}"
    )

    # --------------------------------------------------------
    # Take a central crop
    #
    # This is only a test.
    # Later YOLO will provide the actual cow box.
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    x1 = int(
        width * 0.25
    )

    y1 = int(
        height * 0.25
    )

    x2 = int(
        width * 0.75
    )

    y2 = int(
        height * 0.75
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    feature = extractor.extract(
        crop
    )

    if feature is None:

        print(
            "ERROR: Feature extraction failed"
        )

        return

    print(
        f"Feature length: {len(feature)}"
    )

    print(
        f"Feature dtype : {feature.dtype}"
    )

    print(
        f"Feature norm  : "
        f"{float((feature ** 2).sum()) ** 0.5:.4f}"
    )

    # --------------------------------------------------------
    # Self similarity
    # --------------------------------------------------------

    similarity = extractor.similarity(
        feature,
        feature
    )

    print(
        f"Self similarity: {similarity:.4f}"
    )

    print()

    if similarity > 0.99:

        print(
            "PASS: Appearance extractor works."
        )

    else:

        print(
            "FAIL: Unexpected similarity."
        )

    print()
    print("=" * 60)


if __name__ == "__main__":

    main()

