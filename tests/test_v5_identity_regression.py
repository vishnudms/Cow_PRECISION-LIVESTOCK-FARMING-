import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cow_plf.core.identity_manager import IdentityManager


def main():
    manager = IdentityManager(
        max_distance=220,
        max_missed_frames=600,
        appearance_threshold=0.70,
        strong_rejection_threshold=0.40,
        position_weight=0.25,
        appearance_weight=0.60,
        size_weight=0.15,
        verbose=False,
    )

    # Frame 1: tracker 76 appears.
    manager.start_frame(1)

    used = set()

    cow_1 = manager.update_detection(
        tracker_id=76,
        center=(100, 100),
        box=(50, 50, 150, 150),
        confidence=0.95,
        frame_number=1,
        used_cow_ids=used,
    )

    used.add(cow_1)

    manager.end_frame(
        active_cow_ids=used,
        frame_number=1,
    )

    assert cow_1 == 1, f"Expected Cow 1, got {cow_1}"

    # Frame 2: same tracker must remain Cow 1.
    manager.start_frame(2)

    used = set()

    cow_1_again = manager.update_detection(
        tracker_id=76,
        center=(105, 100),
        box=(55, 50, 155, 150),
        confidence=0.95,
        frame_number=2,
        used_cow_ids=used,
    )

    used.add(cow_1_again)

    manager.end_frame(
        active_cow_ids=used,
        frame_number=2,
    )

    assert cow_1_again == 1, (
        f"Expected Cow 1 on tracker continuation, got {cow_1_again}"
    )

    print("=" * 60)
    print("V5 IDENTITY REGRESSION TEST PASSED")
    print("=" * 60)
    print("Frame 1: Tracker 76 -> Cow 1")
    print("Frame 2: Tracker 76 -> Cow 1")
    print("Permanent identity preserved.")
    print("=" * 60)


if __name__ == "__main__":
    main()