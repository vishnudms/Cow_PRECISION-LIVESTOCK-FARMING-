from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(
    r"D:\cow\dataset2\Total_sorted_DGE_images\Total_sorted_DGE_images"
)

OUTPUT_DIR = Path(
    r"D:\cow\output\bcs_training"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for class_dir in sorted(
        ROOT.iterdir(),
        key=lambda p: (
            int(p.name)
            if p.name.isdigit()
            else 999
        ),
    ):

        if not class_dir.is_dir():
            continue

        source_class = class_dir.name

        for cow_dir in sorted(
            class_dir.iterdir()
        ):

            if not cow_dir.is_dir():
                continue

            images = sorted(
                cow_dir.glob("*.tif")
            )

            for image_path in images:

                rows.append(
                    {
                        "image_path": str(
                            image_path
                        ),
                        "source_class": source_class,
                        "cow_group": cow_dir.name,
                        "image_name": image_path.name,
                    }
                )

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Save complete audit
    # --------------------------------------------------------

    audit_path = (
        OUTPUT_DIR
        / "dryad_audit.csv"
    )

    df.to_csv(
        audit_path,
        index=False,
    )

    # --------------------------------------------------------
    # Cow-group summary
    # --------------------------------------------------------

    cow_summary = (
        df.groupby(
            [
                "source_class",
                "cow_group",
            ],
            as_index=False,
        )
        .agg(
            image_count=(
                "image_name",
                "count",
            )
        )
        .sort_values(
            [
                "source_class",
                "cow_group",
            ]
        )
    )

    cow_summary_path = (
        OUTPUT_DIR
        / "cow_groups.csv"
    )

    cow_summary.to_csv(
        cow_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("DRYAD BCS DATASET AUDIT")
    print("=" * 80)

    print()
    print(
        f"Total images       : {len(df)}"
    )

    print(
        f"Unique cow groups  : "
        f"{df['cow_group'].nunique()}"
    )

    print()
    print("-" * 80)
    print("SOURCE CLASS SUMMARY")
    print("-" * 80)

    summary = (
        df.groupby(
            "source_class"
        )
        .agg(
            images=(
                "image_name",
                "count",
            ),
            cow_groups=(
                "cow_group",
                "nunique",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("-" * 80)
    print("COW GROUP INTEGRITY")
    print("-" * 80)

    ownership = (
        df.groupby(
            "cow_group"
        )["source_class"]
        .nunique()
    )

    shared = ownership[
        ownership > 1
    ]

    if len(shared) == 0:

        print(
            "[OK] No cow group crosses source classes."
        )

    else:

        print(
            "[WARNING] Cow groups crossing classes:"
        )

        print(
            shared.to_string()
        )

    print()
    print("-" * 80)
    print("OUTPUT")
    print("-" * 80)

    print(
        f"Audit : {audit_path}"
    )

    print(
        f"Cows  : {cow_summary_path}"
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()