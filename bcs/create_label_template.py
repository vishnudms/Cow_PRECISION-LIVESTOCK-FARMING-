from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"D:\cow")

FEATURE_FILE = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
    / "bcs_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "bcs"
    / "labels"
    / "bcs_labels.csv"
)


def main():

    df = pd.read_csv(FEATURE_FILE)

    labels = df[
        [
            "cow_id",
            "image_name",
            "actual_weight_kg",
            "oblique_body_length_cm",
            "withers_height_cm",
            "heart_girth_cm",
            "hip_length_cm",
        ]
    ].copy()

    # Expert-provided BCS will be entered here later.
    labels["bcs_score"] = ""

    # Optional metadata for quality control.
    labels["bcs_source"] = ""
    labels["assessor_id"] = ""
    labels["assessment_notes"] = ""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 70)
    print("BCS LABEL TEMPLATE CREATED")
    print("=" * 70)

    print(f"Cows : {len(labels)}")
    print(f"File : {OUTPUT_FILE}")

    print()
    print("BCS labels are intentionally EMPTY.")
    print("They must be supplied from a valid reference assessment.")


if __name__ == "__main__":
    main()