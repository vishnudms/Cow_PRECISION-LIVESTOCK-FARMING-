from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(
    r"D:\cow\dataset2\Total_sorted_DGE_images\Total_sorted_DGE_images"
)

OUTPUT = Path(
    r"D:\cow\output\bcs_training\dryad_manifest.csv"
)


def main():

    rows = []

    for class_dir in sorted(
        ROOT.iterdir(),
        key=lambda p: int(p.name) if p.name.isdigit() else 999,
    ):

        if not class_dir.is_dir():
            continue

        source_class = class_dir.name

        for cow_dir in sorted(class_dir.iterdir()):

            if not cow_dir.is_dir():
                continue

            images = sorted(
                cow_dir.glob("*.tif")
            )

            for image_path in images:

                # DGE = processed depth-derived image.
                # Keep source representation explicit.
                rows.append(
                    {
                        "image_path": str(image_path),
                        "source_class": int(source_class),
                        "cow_group": cow_dir.name,
                        "image_name": image_path.name,
                        "image_type": "DGE",
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No .tif images found.")

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    if df["cow_group"].isna().any():
        raise RuntimeError("Missing cow_group values.")

    ownership = (
        df.groupby("cow_group")["source_class"]
        .nunique()
    )

    shared = ownership[ownership > 1]

    if not shared.empty:
        raise RuntimeError(
            "Cow groups appear in multiple source classes:\n"
            + shared.to_string()
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 80)
    print("DRYAD TRAINING MANIFEST")
    print("=" * 80)

    print(f"Images       : {len(df)}")
    print(f"Cow groups   : {df['cow_group'].nunique()}")
    print(f"Source class : {df['source_class'].nunique()}")

    print()
    print("SOURCE CLASS COUNTS")
    print("-" * 80)

    print(
        df.groupby("source_class")
        .agg(
            images=("image_name", "count"),
            cow_groups=("cow_group", "nunique"),
        )
        .to_string()
    )

    print()
    print(f"Saved to:\n{OUTPUT}")

    print("=" * 80)


if __name__ == "__main__":
    main()