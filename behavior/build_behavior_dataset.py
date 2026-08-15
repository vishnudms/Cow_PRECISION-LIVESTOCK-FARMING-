from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "new"
    / "CBVD-5.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "new"
    / "labelframes"
    / "labelframes"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "behavior"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "behavior_annotations.csv"
)


# ============================================================
# BEHAVIOR LABELS
# ============================================================

BEHAVIORS = {
    0: "standing",
    1: "lying",
    2: "foraging",
    3: "drinking",
    4: "rumination",
}


# ============================================================
# VIA CSV COLUMNS
# ============================================================

VIA_COLUMNS = [
    "metadata_id",
    "file_list",
    "flags",
    "temporal_coordinates",
    "spatial_coordinates",
    "metadata",
]


# ============================================================
# HELPERS
# ============================================================

def parse_list(value: str):
    """
    Parse a VIA list such as:

        ["618_00002.jpg"]

    or:

        [2,877.618,387.741,321.541,349.912]
    """

    if value is None:
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        return ast.literal_eval(
            text
        )

    except Exception:

        try:
            return json.loads(
                text
            )

        except Exception:

            return []


def parse_metadata(value: str) -> dict:
    """
    Parse VIA metadata.

    Example:

        {"1":"0,4"}

    becomes:

        {
            "1": "0,4"
        }
    """

    if value is None:
        return {}

    text = str(value).strip()

    if not text:
        return {}

    # JSON should normally work directly.
    try:
        result = json.loads(
            text
        )

        if isinstance(result, dict):
            return result

    except Exception:
        pass

    # Fallback for slightly unusual VIA formatting.
    try:
        result = ast.literal_eval(
            text
        )

        if isinstance(result, dict):
            return result

    except Exception:
        pass

    return {}


def parse_behavior_labels(metadata: dict) -> list[int]:
    """
    Extract behavior IDs from VIA attribute "1".

    Example:

        {"1": "0,4"}

    -> [0, 4]
    """

    if not metadata:
        return []

    value = (
        metadata.get("1")
        if "1" in metadata
        else metadata.get(1)
    )

    if value is None:
        return []

    text = str(value).strip()

    if not text:
        return []

    labels = []

    for part in text.split(","):

        part = part.strip()

        if not part:
            continue

        try:
            label = int(part)

        except ValueError:
            continue

        if label in BEHAVIORS:
            labels.append(label)

    return sorted(
        set(labels)
    )


def parse_filename(file_list: str) -> str | None:
    """
    Extract the actual image filename from:

        ["618_00002.jpg"]
    """

    values = parse_list(
        file_list
    )

    if not values:
        return None

    filename = str(
        values[0]
    ).strip()

    if not filename:
        return None

    return filename


def parse_bbox(spatial_coordinates: str):
    """
    VIA rectangle format:

        [2, x, y, width, height]

    """

    values = parse_list(
        spatial_coordinates
    )

    if len(values) != 5:
        return None

    try:

        shape_id = int(
            values[0]
        )

        x = float(
            values[1]
        )

        y = float(
            values[2]
        )

        width = float(
            values[3]
        )

        height = float(
            values[4]
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    # We expect rectangle annotations.
    if shape_id != 2:
        return None

    if (
        width <= 0
        or height <= 0
    ):
        return None

    return (
        x,
        y,
        width,
        height,
    )


def get_sequence_id(image_name: str) -> str:
    """
    Extract the source sequence/video identifier.

    Example:

        618_00002.jpg -> 618
    """

    stem = Path(
        image_name
    ).stem

    match = re.match(
        r"^([^_]+)_",
        stem,
    )

    if match:
        return match.group(1)

    return stem


def get_frame_id(image_name: str) -> str:
    """
    Extract frame identifier.

    Example:

        618_00002.jpg -> 00002
    """

    stem = Path(
        image_name
    ).stem

    parts = stem.split(
        "_",
        1,
    )

    if len(parts) == 2:
        return parts[1]

    return stem


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("COW PLF - BEHAVIOR DATASET BUILDER")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not ANNOTATION_FILE.exists():

        raise FileNotFoundError(
            f"Annotation file not found:\n"
            f"{ANNOTATION_FILE}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{IMAGE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Annotations : {ANNOTATION_FILE}"
    )

    print(
        f"Images      : {IMAGE_DIR}"
    )

    # --------------------------------------------------------
    # Read VIA CSV
    #
    # The CSV header is commented out, therefore names are
    # supplied explicitly.
    # --------------------------------------------------------

    df = pd.read_csv(
        ANNOTATION_FILE,
        comment="#",
        header=None,
        names=VIA_COLUMNS,
        dtype=str,
    )

    print()
    print(
        f"Raw annotation rows : {len(df)}"
    )

    # --------------------------------------------------------
    # Parse annotations
    # --------------------------------------------------------

    rows = []

    missing_images = 0
    invalid_bbox = 0
    invalid_metadata = 0
    no_behavior = 0

    for index, row in df.iterrows():

        image_name = parse_filename(
            row["file_list"]
        )

        if image_name is None:
            continue

        image_path = (
            IMAGE_DIR
            / image_name
        )

        if not image_path.exists():

            missing_images += 1

            continue

        bbox = parse_bbox(
            row["spatial_coordinates"]
        )

        if bbox is None:

            invalid_bbox += 1

            continue

        metadata = parse_metadata(
            row["metadata"]
        )

        if not metadata:

            invalid_metadata += 1

        behavior_ids = parse_behavior_labels(
            metadata
        )

        if not behavior_ids:

            no_behavior += 1

            continue

        x, y, width, height = bbox

        # ----------------------------------------------------
        # Multi-hot encoding
        # ----------------------------------------------------

        label_values = {
            behavior: 0
            for behavior in BEHAVIORS.values()
        }

        for behavior_id in behavior_ids:

            behavior_name = BEHAVIORS[
                behavior_id
            ]

            label_values[
                behavior_name
            ] = 1

        # ----------------------------------------------------
        # Build row
        # ----------------------------------------------------

        output_row = {

            "annotation_id": str(
                row["metadata_id"]
            ),

            "image_name": image_name,

            "image_path": str(
                image_path
            ),

            "sequence_id": get_sequence_id(
                image_name
            ),

            "frame_id": get_frame_id(
                image_name
            ),

            "x": x,
            "y": y,
            "width": width,
            "height": height,

            "standing": label_values[
                "standing"
            ],

            "lying": label_values[
                "lying"
            ],

            "foraging": label_values[
                "foraging"
            ],

            "drinking": label_values[
                "drinking"
            ],

            "rumination": label_values[
                "rumination"
            ],

            "behavior_ids": ",".join(
                map(
                    str,
                    behavior_ids,
                )
            ),
        }

        rows.append(
            output_row
        )

    # --------------------------------------------------------
    # Output dataframe
    # --------------------------------------------------------

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        raise RuntimeError(
            "No valid behavior annotations were produced."
        )

    # --------------------------------------------------------
    # Numeric normalization
    # --------------------------------------------------------

    for column in [
        "x",
        "y",
        "width",
        "height",
    ]:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    for column in BEHAVIORS.values():

        result[column] = result[
            column
        ].astype(int)

    # --------------------------------------------------------
    # Validate boxes
    # --------------------------------------------------------

    result = result[
        (
            result["width"] > 0
        )
        &
        (
            result["height"] > 0
        )
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("BEHAVIOR DATASET SUMMARY")
    print("=" * 80)

    print()
    print(
        f"Valid annotations : {len(result)}"
    )

    print(
        f"Unique images     : "
        f"{result['image_name'].nunique()}"
    )

    print(
        f"Sequences         : "
        f"{result['sequence_id'].nunique()}"
    )

    print(
        f"Unique annotation boxes : "
        f"{len(result)}"
    )

    # --------------------------------------------------------
    # Behavior counts
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("BEHAVIOR LABEL COUNTS")
    print("-" * 80)

    for behavior in BEHAVIORS.values():

        count = int(
            result[behavior].sum()
        )

        print(
            f"{behavior:15s}: {count}"
        )

    # --------------------------------------------------------
    # Combination counts
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("BEHAVIOR COMBINATIONS")
    print("-" * 80)

    combinations = (
        result["behavior_ids"]
        .value_counts()
        .head(20)
    )

    print(
        combinations.to_string()
    )

    # --------------------------------------------------------
    # Sequence distribution
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("TOP SEQUENCES")
    print("-" * 80)

    sequence_counts = (
        result["sequence_id"]
        .value_counts()
        .head(20)
    )

    print(
        sequence_counts.to_string()
    )

    # --------------------------------------------------------
    # Parsing diagnostics
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("PARSING DIAGNOSTICS")
    print("-" * 80)

    print(
        f"Missing images skipped    : {missing_images}"
    )

    print(
        f"Invalid boxes skipped     : {invalid_bbox}"
    )

    print(
        f"Invalid metadata rows     : {invalid_metadata}"
    )

    print(
        f"No behavior rows skipped  : {no_behavior}"
    )

    print()
    print(
        f"Saved:\n{OUTPUT_FILE}"
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()