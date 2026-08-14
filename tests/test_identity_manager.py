import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from cow_plf.core.identity_manager import IdentityManager


print()
print("=" * 60)
print("PERMANENT COW IDENTITY V5 TEST")
print("=" * 60)


# ============================================================
# CREATE IDENTITY MANAGER
# ============================================================

manager = IdentityManager(

    max_distance=350,

    max_missed_frames=1200,

    appearance_threshold=0.72,

    strong_appearance_threshold=0.88,

    max_gallery_size=12

)


# ============================================================
# CREATE TEST APPEARANCE FEATURES
# ============================================================

# ------------------------------------------------------------
# Cow 1
# ------------------------------------------------------------

rng1 = np.random.RandomState(1)

cow1_feature = rng1.randn(
    1286
).astype(
    np.float32
)

cow1_feature /= np.linalg.norm(
    cow1_feature
)


# ------------------------------------------------------------
# Cow 2
# ------------------------------------------------------------

rng2 = np.random.RandomState(2)

cow2_feature = rng2.randn(
    1286
).astype(
    np.float32
)

cow2_feature /= np.linalg.norm(
    cow2_feature
)


# ============================================================
# FRAME 1
# ============================================================

cow_id_1 = manager.update_detection(

    tracker_id=76,

    center=(500, 500),

    box=(450, 450, 550, 550),

    confidence=0.90,

    frame_number=1,

    used_cow_ids=set(),

    appearance=cow1_feature

)

print(
    f"Frame 1: Tracker 76 -> COW {cow_id_1}"
)

manager.end_frame(

    {cow_id_1},

    1

)


# ============================================================
# FRAME 2
# ============================================================

cow_id_2 = manager.update_detection(

    tracker_id=76,

    center=(510, 510),

    box=(460, 460, 560, 560),

    confidence=0.92,

    frame_number=2,

    used_cow_ids=set(),

    appearance=cow1_feature

)

print(
    f"Frame 2: Tracker 76 -> COW {cow_id_2}"
)

manager.end_frame(

    {cow_id_2},

    2

)


# ============================================================
# FRAME 3
#
# Tracker 76 disappears.
#
# ByteTrack gives tracker 184.
#
# BUT:
#
# Appearance = Cow 1
#
# Therefore:
#
# Tracker 184 -> COW 1
# ============================================================

cow_id_3 = manager.update_detection(

    tracker_id=184,

    center=(525, 525),

    box=(475, 475, 575, 575),

    confidence=0.91,

    frame_number=3,

    used_cow_ids=set(),

    appearance=cow1_feature

)

print(
    f"Frame 3: Tracker 184 -> COW {cow_id_3}"
)

manager.end_frame(

    {cow_id_3},

    3

)


# ============================================================
# FRAME 4
#
# Completely different cow.
#
# Appearance = Cow 2
# Location = far away
#
# Therefore:
#
# Tracker 200 -> COW 2
# ============================================================

cow_id_4 = manager.update_detection(

    tracker_id=200,

    center=(900, 900),

    box=(850, 850, 950, 950),

    confidence=0.91,

    frame_number=4,

    used_cow_ids=set(),

    appearance=cow2_feature

)

print(
    f"Frame 4: Tracker 200 -> COW {cow_id_4}"
)

manager.end_frame(

    {cow_id_4},

    4

)


# ============================================================
# STATISTICS
# ============================================================

stats = manager.get_statistics()


print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Active cows      : "
    f"{stats['active_cows']}"
)

print(
    f"Total known cows : "
    f"{stats['total_known_cows']}"
)

print(
    f"New cows         : "
    f"{stats['total_new_cows']}"
)

print(
    f"Reconnections    : "
    f"{stats['total_reconnections']}"
)

print(
    f"Updates          : "
    f"{stats['total_updates']}"
)


# ============================================================
# EXPECTED
# ============================================================

print()
print("EXPECTED:")
print(
    "Tracker 76  -> COW 1"
)

print(
    "Tracker 76  -> COW 1"
)

print(
    "Tracker 184 -> COW 1"
)

print(
    "Tracker 200 -> COW 2"
)


# ============================================================
# VALIDATION
# ============================================================

passed = True


# ------------------------------------------------------------
# Exactly 2 permanent cows
# ------------------------------------------------------------

if stats["total_known_cows"] != 2:

    print()
    print(
        "FAIL: Expected exactly 2 known cows."
    )

    passed = False


# ------------------------------------------------------------
# Exactly 2 new identities
# ------------------------------------------------------------

if stats["total_new_cows"] != 2:

    print()
    print(
        "FAIL: Expected exactly 2 new cows."
    )

    passed = False


# ------------------------------------------------------------
# Tracker 184 must reconnect to Cow 1
# ------------------------------------------------------------

if stats["total_reconnections"] < 1:

    print()
    print(
        "FAIL: Tracker 184 was not "
        "recognized as a reconnection."
    )

    passed = False


# ------------------------------------------------------------
# Cow 1
# ------------------------------------------------------------

cow1 = manager.get_cow(1)

if cow1 is None:

    print()
    print(
        "FAIL: COW 1 does not exist."
    )

    passed = False

else:

    if cow1.observation_count < 3:

        print()
        print(
            "FAIL: COW 1 should have "
            "at least 3 observations."
        )

        passed = False


# ------------------------------------------------------------
# Cow 2
# ------------------------------------------------------------

cow2 = manager.get_cow(2)

if cow2 is None:

    print()
    print(
        "FAIL: COW 2 does not exist."
    )

    passed = False


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print()

if passed:

    print(
        "PASS: V5 identity manager correctly "
        "recognized the same cow after "
        "the tracker ID changed."
    )

else:

    print(
        "FAIL: Identity manager still needs "
        "improvement."
    )


print()
print("=" * 60)
