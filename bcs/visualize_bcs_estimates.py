from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(r"D:\cow")

ESTIMATE_FILE = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
    / "provisional_bcs_estimates.csv"
)

# Images used by the morphometrics pipeline
IMAGE_DIR = PROJECT_ROOT / "dataset"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
    / "visualization"
)


def main():

    print("=" * 80)
    print("COW PLF - BCS VISUALIZATION")
    print("=" * 80)

    if not ESTIMATE_FILE.exists():
        raise FileNotFoundError(
            f"Estimate file not found:\n{ESTIMATE_FILE}"
        )

    df = pd.read_csv(ESTIMATE_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"[LOAD] Estimates : {len(df)}")

    generated = 0
    missing = 0

    for _, row in df.iterrows():

        image_name = str(row["image_name"])

        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            print(
                f"[WARNING] Image not found: "
                f"{image_path}"
            )
            missing += 1
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"[WARNING] Could not read: "
                f"{image_path}"
            )
            continue

        bcs = float(
            row["provisional_bcs"]
        )

        confidence = float(
            row["provisional_bcs_confidence"]
        )

        cow_id = int(
            row["cow_id"]
        )

        # ----------------------------------------------------
        # Text panel
        # ----------------------------------------------------

        lines = [
            f"COW ID: {cow_id}",
            f"PROVISIONAL BCS: {bcs:.2f}",
            f"CONFIDENCE: {confidence:.2f}",
            "STATUS: PROVISIONAL",
        ]

        y = 60

        for line in lines:

            cv2.putText(
                image,
                line,
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                line,
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            y += 55

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_name = (
            image_path.stem
            + "_bcs.jpg"
        )

        output_path = (
            OUTPUT_DIR / output_name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        generated += 1

    print()
    print("=" * 80)
    print("BCS VISUALIZATION COMPLETE")
    print("=" * 80)

    print(
        f"Generated : {generated}"
    )

    print(
        f"Missing   : {missing}"
    )

    print(
        f"Output    : {OUTPUT_DIR}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()