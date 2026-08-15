from pathlib import Path
import pandas as pd

DATASET_DIR = Path(r"D:\cow\dataset")
MEASUREMENTS_FILE = DATASET_DIR / "measurements.xlsx"
OUTPUT_DIR = Path(r"D:\cow\output\morphometrics_dataset")
OUTPUT_FILE = OUTPUT_DIR / "paired_cow_dataset.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main():
    print("=" * 80)
    print("COW PLF - PAIRED COW DATASET BUILDER")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load ground-truth measurements
    # ------------------------------------------------------------------

    df = pd.read_excel(
        MEASUREMENTS_FILE,
        sheet_name="Sheet1"
    )

    required_columns = [
        "Num",
        "Oblique body length (cm)",
        "Withers height(cm)",
        "Heart girth(cm)",
        "Hip length (cm)",
        "Body weight (kg)",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing Excel columns: {missing_columns}"
        )

    # ------------------------------------------------------------------
    # Find images
    # ------------------------------------------------------------------

    image_map = {}

    for path in DATASET_DIR.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Expected format: 1.png, 2.png, ..., 72.png
        try:
            cow_id = int(path.stem)
        except ValueError:
            continue

        image_map[cow_id] = path.name

    print(f"\nMeasurement records : {len(df)}")
    print(f"Images found        : {len(image_map)}")

    # ------------------------------------------------------------------
    # Build paired dataset
    # ------------------------------------------------------------------

    paired_rows = []
    missing_images = []

    for _, record in df.iterrows():

        cow_id = int(record["Num"])

        image_name = image_map.get(cow_id)

        if image_name is None:
            missing_images.append(cow_id)
            continue

        paired_rows.append({
            "cow_id": cow_id,
            "image_name": image_name,

            "oblique_body_length_cm":
                float(record["Oblique body length (cm)"]),

            "withers_height_cm":
                float(record["Withers height(cm)"]),

            "heart_girth_cm":
                float(record["Heart girth(cm)"]),

            "hip_length_cm":
                float(record["Hip length (cm)"]),

            "actual_weight_kg":
                float(record["Body weight (kg)"]),
        })

    paired_df = pd.DataFrame(paired_rows)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    paired_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PAIRING COMPLETE")
    print("=" * 80)

    print(
        f"Paired cows         : {len(paired_df)}"
    )

    print(
        f"Missing images      : {missing_images}"
    )

    print(
        f"Output              : {OUTPUT_FILE}"
    )

    print("\nFirst 10 rows:")
    print(
        paired_df.head(10).to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()