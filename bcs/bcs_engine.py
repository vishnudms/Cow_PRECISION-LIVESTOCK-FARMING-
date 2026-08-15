from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

PROVISIONAL_FILE = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
    / "provisional_bcs_estimates.csv"
)


# ============================================================
# RESULT
# ============================================================

@dataclass
class BCSEngineResult:
    cow_id: Any
    bcs: Optional[float]
    confidence: float
    confidence_status: str
    category: str
    measurement_valid: bool
    status: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# ENGINE
# ============================================================

class BCSEngine:
    """
    Production interface for the current PROVISIONAL BCS system.

    IMPORTANT:
    The current estimator is a silhouette-based heuristic.
    It is NOT a veterinary-validated BCS prediction model.
    """

    MIN_BCS = 1.0
    MAX_BCS = 5.0

    def __init__(
        self,
        provisional_file: Path = PROVISIONAL_FILE,
    ):
        self.provisional_file = Path(
            provisional_file
        )

        self.df = None

        self.load()

    # ========================================================
    # LOAD
    # ========================================================

    def load(self):

        if not self.provisional_file.exists():

            raise FileNotFoundError(
                "Provisional BCS dataset not found:\n"
                f"{self.provisional_file}"
            )

        self.df = pd.read_csv(
            self.provisional_file
        )

        required = [
            "cow_id",
            "image_name",
            "provisional_bcs",
            "provisional_bcs_confidence",
            "bcs_estimation_method",
            "bcs_confidence_type",
            "bcs_source",
            "label_status",
        ]

        missing = [
            column
            for column in required
            if column not in self.df.columns
        ]

        if missing:

            raise RuntimeError(
                "Missing BCS engine columns:\n"
                + "\n".join(missing)
            )

    # ========================================================
    # BCS CATEGORY
    # ========================================================

    @staticmethod
    def classify_bcs(
        bcs: float,
    ) -> str:

        if bcs < 2.0:
            return "UNDER_CONDITIONED"

        if bcs < 3.0:
            return "LEAN"

        if bcs < 3.5:
            return "MODERATE"

        return "HIGH_CONDITION"

    # ========================================================
    # CONFIDENCE CATEGORY
    # ========================================================

    @staticmethod
    def confidence_status(
        confidence: float,
    ) -> str:

        # This is feature-quality confidence,
        # NOT veterinary prediction confidence.

        if confidence >= 0.80:
            return "HIGH_FEATURE_QUALITY"

        if confidence >= 0.60:
            return "MEDIUM_FEATURE_QUALITY"

        if confidence >= 0.40:
            return "LOW_FEATURE_QUALITY"

        return "VERY_LOW_FEATURE_QUALITY"

    # ========================================================
    # VALIDATE BCS
    # ========================================================

    @staticmethod
    def validate_bcs(
        bcs: float,
    ) -> bool:

        return (
            BCSEngine.MIN_BCS
            <= float(bcs)
            <= BCSEngine.MAX_BCS
        )

    # ========================================================
    # FIND COW
    # ========================================================

    def _find_cow(
        self,
        cow_id: Any,
    ):

        matches = self.df[
            self.df["cow_id"].astype(str)
            == str(cow_id)
        ]

        if len(matches) == 0:
            return None

        return matches.iloc[0]

    # ========================================================
    # PREDICT / GET RESULT
    # ========================================================

    def predict(
        self,
        cow_id: Any,
        measurement_valid: bool = True,
        view_class: str = "SIDE_VIEW_GOOD",
    ) -> BCSEngineResult:

        # ----------------------------------------------------
        # Measurement gate
        # ----------------------------------------------------

        if not measurement_valid:

            return BCSEngineResult(
                cow_id=cow_id,
                bcs=None,
                confidence=0.0,
                confidence_status="VERY_LOW_FEATURE_QUALITY",
                category="UNAVAILABLE",
                measurement_valid=False,
                status="REJECTED",
                reason=(
                    "Morphometric measurement or segmentation "
                    "was invalid."
                ),
            )

        # ----------------------------------------------------
        # View gate
        # ----------------------------------------------------

        if view_class not in {
            "SIDE_VIEW_GOOD",
            "SIDE_VIEW_ACCEPTABLE",
        }:

            return BCSEngineResult(
                cow_id=cow_id,
                bcs=None,
                confidence=0.0,
                confidence_status="VERY_LOW_FEATURE_QUALITY",
                category="UNAVAILABLE",
                measurement_valid=False,
                status="REJECTED",
                reason=(
                    f"Invalid view for BCS: {view_class}"
                ),
            )

        # ----------------------------------------------------
        # Find cow
        # ----------------------------------------------------

        row = self._find_cow(
            cow_id
        )

        if row is None:

            return BCSEngineResult(
                cow_id=cow_id,
                bcs=None,
                confidence=0.0,
                confidence_status="VERY_LOW_FEATURE_QUALITY",
                category="UNAVAILABLE",
                measurement_valid=False,
                status="NOT_FOUND",
                reason=(
                    f"No provisional BCS estimate found "
                    f"for cow {cow_id}."
                ),
            )

        # ----------------------------------------------------
        # Read estimate
        # ----------------------------------------------------

        try:

            bcs = float(
                row["provisional_bcs"]
            )

            confidence = float(
                row["provisional_bcs_confidence"]
            )

        except (
            TypeError,
            ValueError,
        ):

            return BCSEngineResult(
                cow_id=cow_id,
                bcs=None,
                confidence=0.0,
                confidence_status="VERY_LOW_FEATURE_QUALITY",
                category="UNAVAILABLE",
                measurement_valid=False,
                status="REJECTED",
                reason="Invalid provisional BCS data.",
            )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not self.validate_bcs(bcs):

            return BCSEngineResult(
                cow_id=cow_id,
                bcs=None,
                confidence=0.0,
                confidence_status="VERY_LOW_FEATURE_QUALITY",
                category="UNAVAILABLE",
                measurement_valid=False,
                status="REJECTED",
                reason="BCS outside valid range.",
            )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        # ----------------------------------------------------
        # Category
        # ----------------------------------------------------

        category = self.classify_bcs(
            bcs
        )

        confidence_label = (
            self.confidence_status(
                confidence
            )
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return BCSEngineResult(
            cow_id=cow_id,
            bcs=round(
                bcs,
                2,
            ),
            confidence=round(
                confidence,
                3,
            ),
            confidence_status=confidence_label,
            category=category,
            measurement_valid=True,
            status="PROVISIONAL",
            reason=(
                "Image-derived silhouette heuristic estimate. "
                "No independent veterinary/reference BCS label."
            ),
        )

    # ========================================================
    # ALL COWS
    # ========================================================

    def predict_all(self):

        results = []

        for _, row in self.df.iterrows():

            result = self.predict(
                cow_id=row["cow_id"],
                measurement_valid=True,
                view_class="SIDE_VIEW_GOOD",
            )

            results.append(
                result.to_dict()
            )

        return pd.DataFrame(
            results
        )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("COW PLF - BCS ENGINE TEST")
    print("=" * 70)

    engine = BCSEngine()

    # Test Cow 16 using the CURRENT estimator.
    result = engine.predict(
        cow_id=16,
        measurement_valid=True,
        view_class="SIDE_VIEW_GOOD",
    )

    for key, value in result.to_dict().items():
        print(
            f"{key:22}: {value}"
        )

    print("=" * 70)