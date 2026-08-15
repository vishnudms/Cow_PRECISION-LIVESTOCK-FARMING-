from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"D:\cow")

MASTER_FILE = (
    PROJECT_ROOT
    / "output"
    / "morphometrics_dataset"
    / "master_cow_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "bcs"
    / "labels"
    / "bcs_labeling_dataset.csv"
)


def main():

    print("=" * 80)
    print("COW PLF - BCS LABELING DATASET")
    print("=" * 80)

    df = pd.read_csv(MASTER_FILE)

    # Only valid side-view measurements
    df = df[
        df["measurement_status"]
        == "MEASUREMENT_VALID"
    ].copy()

    columns = [
        "cow_id",
        "image_name",
        "actual_weight_kg",
        "oblique_body_length_cm",
        "withers_height_cm",
        "heart_girth_cm",
        "hip_length_cm",
    ]

    output = df[columns].copy()

    # Empty expert-label fields
    output["bcs_score"] = ""
    output["bcs_source"] = ""
    output["assessor_id"] = ""
    output["assessment_notes"] = ""

    output["label_status"] = "UNLABELED"

    output = output.sort_values("cow_id")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Cows : {len(output)}")
    print(f"File : {OUTPUT_FILE}")

    print()
    print("Columns:")
    for column in output.columns:
        print(f"  - {column}")

    print()
    print("=" * 80)
    print("BCS LABELING DATASET CREATED")
    print("=" * 80)


if __name__ == "__main__":
    main()