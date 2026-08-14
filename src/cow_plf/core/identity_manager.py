"""
===========================================================================
PERMANENT COW IDENTITY MANAGER - V6
===========================================================================

Purpose
-------
Convert temporary ByteTrack IDs into permanent COW IDs.

V6 improvements:
    1. Permanent identity
    2. Appearance matching
    3. Motion prediction
    4. Velocity estimation
    5. Spatial gating
    6. One-to-one identity locking
    7. Missed-frame memory
    8. Appearance history
    9. Stable identity updates
   10. Backward-compatible API

Existing API preserved:
    update_detection(...)
    end_frame(...)
    get_statistics()
    get_cow()
    get_cow_id()
    get_active_cows()

Example:

    Tracker 76  -> COW 1
    Tracker 76  -> COW 1
    Tracker 184 -> COW 1
    Tracker 200 -> COW 2

=========================================================================== 
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import math
import numpy as np


# ============================================================================
# BASIC HELPERS
# ============================================================================

def normalize_feature(
    feature
):

    if feature is None:
        return None

    try:
        feature = np.asarray(
            feature,
            dtype=np.float32
        ).reshape(-1)

    except Exception:
        return None

    if feature.size == 0:
        return None

    norm = np.linalg.norm(
        feature
    )

    if norm <= 1e-8:
        return None

    return feature / norm


def cosine_similarity(
    feature_a,
    feature_b
):

    if (
        feature_a is None
        or
        feature_b is None
    ):
        return 0.0

    a = normalize_feature(
        feature_a
    )

    b = normalize_feature(
        feature_b
    )

    if a is None or b is None:
        return 0.0

    if a.shape != b.shape:
        return 0.0

    return float(
        np.clip(
            np.dot(a, b),
            -1.0,
            1.0
        )
    )


def normalize_box(
    box
):

    if box is None:
        return None

    try:

        arr = np.asarray(
            box,
            dtype=np.float32
        ).reshape(-1)

    except Exception:

        return None

    if arr.size < 4:
        return None

    return arr[:4]


def box_center(
    box
):

    box = normalize_box(
        box
    )

    if box is None:
        return None

    x1, y1, x2, y2 = box

    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        ],
        dtype=np.float32
    )


def box_size(
    box
):

    box = normalize_box(
        box
    )

    if box is None:
        return None

    x1, y1, x2, y2 = box

    return (
        max(1.0, x2 - x1),
        max(1.0, y2 - y1)
    )


def euclidean_distance(
    point_a,
    point_b
):

    if (
        point_a is None
        or
        point_b is None
    ):
        return float("inf")

    return float(
        np.linalg.norm(
            np.asarray(point_a)
            -
            np.asarray(point_b)
        )
    )


def size_similarity(
    box_a,
    box_b
):

    size_a = box_size(
        box_a
    )

    size_b = box_size(
        box_b
    )

    if (
        size_a is None
        or
        size_b is None
    ):
        return 0.0

    wa, ha = size_a
    wb, hb = size_b

    width_score = (
        min(wa, wb)
        /
        max(wa, wb)
    )

    height_score = (
        min(ha, hb)
        /
        max(ha, hb)
    )

    return float(
        (
            width_score
            +
            height_score
        )
        /
        2.0
    )


# ============================================================================
# COW IDENTITY
# ============================================================================

@dataclass
class CowIdentity:

    cow_id: int

    current_tracker_id: Optional[int] = None

    previous_tracker_id: Optional[int] = None

    appearance: Optional[np.ndarray] = None

    # Appearance history
    appearance_history: list = field(
        default_factory=list
    )

    # Position history
    last_box: Optional[np.ndarray] = None

    last_center: Optional[np.ndarray] = None

    previous_center: Optional[np.ndarray] = None

    # Velocity in pixels/frame
    velocity: Optional[np.ndarray] = None

    # Frame information
    first_frame: int = -1

    last_frame: int = -1

    missed_frames: int = 0

    # Statistics
    observations: int = 0

    updates: int = 0

    reconnections: int = 0

    # Current state
    active: bool = False

    # Tracker history
    tracker_history: list = field(
        default_factory=list
    )

    # ------------------------------------------------------------
    # Compatibility property
    # ------------------------------------------------------------

    @property
    def observation_count(self):

        return self.observations

    @observation_count.setter
    def observation_count(
        self,
        value
    ):

        self.observations = int(
            value
        )

    @property
    def tracker_id(self):

        return self.current_tracker_id

    @property
    def permanent_id(self):

        return self.cow_id


# ============================================================================
# IDENTITY MANAGER V6
# ============================================================================

class IdentityManager:

    def __init__(
        self,

        # ------------------------------------------------------------
        # Appearance
        # ------------------------------------------------------------

        appearance_threshold=0.70,

        strong_appearance_threshold=0.86,

        strong_rejection_threshold=0.45,

        # ------------------------------------------------------------
        # Spatial
        # ------------------------------------------------------------

        max_distance=300.0,

        # ------------------------------------------------------------
        # Missed frames
        # ------------------------------------------------------------

        max_missed_frames=180,

        max_reid_frames=900,

        # ------------------------------------------------------------
        # Weights
        # ------------------------------------------------------------

        appearance_weight=0.55,

        motion_weight=0.25,

        position_weight=0.10,

        size_weight=0.10,

        # ------------------------------------------------------------
        # Appearance update
        # ------------------------------------------------------------

        appearance_update_rate=0.10,

        max_appearance_history=10,

        # ------------------------------------------------------------
        # Debug
        # ------------------------------------------------------------

        verbose=True,

        **kwargs
    ):

        # ============================================================
        # SETTINGS
        # ============================================================

        self.appearance_threshold = float(
            appearance_threshold
        )

        self.strong_appearance_threshold = float(
            strong_appearance_threshold
        )

        self.strong_rejection_threshold = float(
            strong_rejection_threshold
        )

        self.max_distance = float(
            max_distance
        )

        self.max_missed_frames = int(
            max_missed_frames
        )

        self.max_reid_frames = int(
            max_reid_frames
        )

        self.appearance_weight = float(
            appearance_weight
        )

        self.motion_weight = float(
            motion_weight
        )

        self.position_weight = float(
            position_weight
        )

        self.size_weight = float(
            size_weight
        )

        self.appearance_update_rate = float(
            appearance_update_rate
        )

        self.max_appearance_history = int(
            max_appearance_history
        )

        self.verbose = bool(
            verbose
        )

        # ============================================================
        # PERMANENT COW DATABASE
        # ============================================================

        self.cows: Dict[
            int,
            CowIdentity
        ] = {}

        # ============================================================
        # TRACKER -> COW
        # ============================================================

        self.tracker_to_cow: Dict[
            int,
            int
        ] = {}

        # ============================================================
        # ID COUNTER
        # ============================================================

        self.next_cow_id = 1

        # ============================================================
        # FRAME
        # ============================================================

        self.current_frame = 0

        # ============================================================
        # CURRENT FRAME ASSIGNMENTS
        #
        # This is the important V6 one-to-one protection.
        #
        # Example:
        #
        # Detection A -> COW 7
        #
        # Detection B cannot also become COW 7.
        # ============================================================

        self.current_tracker_ids = set()

        self.current_cow_ids = set()

        # ============================================================
        # STATISTICS
        # ============================================================

        self.total_new_cows = 0

        self.total_reconnections = 0

        self.total_updates = 0

        self.total_observations = 0

        # Compatibility
        self.new_cows = 0

        self.reconnections = 0

        self.updates = 0


    # ========================================================================
    # START FRAME
    # ========================================================================

    def start_frame(
        self,
        frame_number
    ):

        self.current_frame = int(
            frame_number
        )

        self.current_tracker_ids.clear()

        self.current_cow_ids.clear()

        for cow in self.cows.values():

            cow.active = False


    # ========================================================================
    # CREATE NEW COW
    # ========================================================================

    def _create_cow(
        self,
        tracker_id,
        box,
        appearance,
        frame_number
    ):

        cow_id = (
            self.next_cow_id
        )

        self.next_cow_id += 1

        feature = normalize_feature(
            appearance
        )

        box = normalize_box(
            box
        )

        center = box_center(
            box
        )

        cow = CowIdentity(

            cow_id=cow_id,

            current_tracker_id=int(
                tracker_id
            ),

            previous_tracker_id=None,

            appearance=feature,

            appearance_history=(
                [feature.copy()]
                if feature is not None
                else []
            ),

            last_box=(
                box.copy()
                if box is not None
                else None
            ),

            last_center=(
                center.copy()
                if center is not None
                else None
            ),

            previous_center=None,

            velocity=None,

            first_frame=int(
                frame_number
            ),

            last_frame=int(
                frame_number
            ),

            missed_frames=0,

            observations=1,

            updates=1 if feature is not None else 0,

            reconnections=0,

            active=True,

            tracker_history=[
                int(tracker_id)
            ]
        )

        self.cows[
            cow_id
        ] = cow

        self.tracker_to_cow[
            int(tracker_id)
        ] = cow_id

        self.total_new_cows += 1

        self.new_cows += 1

        self.current_tracker_ids.add(
            int(tracker_id)
        )

        self.current_cow_ids.add(
            cow_id
        )

        if self.verbose:

            print(
                f"[NEW COW] "
                f"Tracker {tracker_id} "
                f"-> COW {cow_id}"
            )

        return cow_id


    # ========================================================================
    # APPEARANCE SCORE
    # ========================================================================

    def _appearance_score(
        self,
        cow,
        appearance
    ):

        feature = normalize_feature(
            appearance
        )

        if feature is None:

            return 0.0

        best_score = 0.0

        if cow.appearance is not None:

            best_score = max(
                best_score,
                cosine_similarity(
                    feature,
                    cow.appearance
                )
            )

        for stored in cow.appearance_history:

            score = cosine_similarity(
                feature,
                stored
            )

            if score > best_score:

                best_score = score

        return float(
            best_score
        )


    # ========================================================================
    # MOTION PREDICTION
    # ========================================================================

    def _predict_center(
        self,
        cow,
        frame_number
    ):

        if cow.last_center is None:

            return None

        if cow.velocity is None:

            return cow.last_center.copy()

        frames_ahead = max(
            1,
            int(
                frame_number
                -
                cow.last_frame
            )
        )

        # Limit extrapolation.
        frames_ahead = min(
            frames_ahead,
            12
        )

        predicted = (
            cow.last_center
            +
            cow.velocity
            *
            frames_ahead
        )

        return predicted.astype(
            np.float32
        )


    # ========================================================================
    # MOTION SCORE
    # ========================================================================

    def _motion_score(
        self,
        cow,
        new_box,
        frame_number
    ):

        new_center = box_center(
            new_box
        )

        predicted = self._predict_center(
            cow,
            frame_number
        )

        if (
            new_center is None
            or
            predicted is None
        ):

            return 0.0

        distance = euclidean_distance(
            new_center,
            predicted
        )

        allowed = (
            self.max_distance
            *
            max(
                1.0,
                min(
                    1.0
                    +
                    cow.missed_frames
                    *
                    0.15,
                    3.0
                )
            )
        )

        if distance >= allowed:

            return 0.0

        return float(
            max(
                0.0,
                1.0
                -
                distance
                /
                allowed
            )
        )


    # ========================================================================
    # POSITION SCORE
    # ========================================================================

    def _position_score(
        self,
        cow,
        box
    ):

        if (
            cow.last_box is None
            or
            box is None
        ):

            return 0.0

        distance = euclidean_distance(
            box_center(box),
            cow.last_center
        )

        if distance >= self.max_distance:

            return 0.0

        return float(
            max(
                0.0,
                1.0
                -
                distance
                /
                self.max_distance
            )
        )


    # ========================================================================
    # FIND BEST MATCH
    # ========================================================================

    def _find_best_match(
        self,
        tracker_id,
        appearance,
        box,
        frame_number
    ):

        best_cow = None

        best_score = -1.0

        best_appearance = 0.0

        best_motion = 0.0

        # ==============================================================
        # CHECK EVERY COW
        # ==============================================================

        for cow in self.cows.values():

            # ----------------------------------------------------------
            # ONE-TO-ONE PROTECTION
            # ----------------------------------------------------------

            if cow.cow_id in self.current_cow_ids:

                continue

            # ----------------------------------------------------------
            # Recently active cow owned by another tracker
            # ----------------------------------------------------------

            if (
                cow.active
                and
                cow.current_tracker_id is not None
                and
                cow.current_tracker_id != tracker_id
            ):

                continue

            # ----------------------------------------------------------
            # Too old
            # ----------------------------------------------------------

            missed = max(
                0,
                frame_number
                -
                cow.last_frame
            )

            if missed > self.max_reid_frames:

                continue

            # ----------------------------------------------------------
            # Appearance
            # ----------------------------------------------------------

            appearance_score = (
                self._appearance_score(
                    cow,
                    appearance
                )
            )

            # ----------------------------------------------------------
            # Motion
            # ----------------------------------------------------------

            motion_score = (
                self._motion_score(
                    cow,
                    box,
                    frame_number
                )
            )

            # ----------------------------------------------------------
            # Position
            # ----------------------------------------------------------

            position_score = (
                self._position_score(
                    cow,
                    box
                )
            )

            # ----------------------------------------------------------
            # Size
            # ----------------------------------------------------------

            size_score = (
                size_similarity(
                    box,
                    cow.last_box
                )
            )

            # ==========================================================
            # STRONG APPEARANCE
            # ==========================================================

            if (
                appearance_score
                >=
                self.strong_appearance_threshold
            ):

                appearance_weight = (
                    0.70
                )

                motion_weight = (
                    0.20
                )

                position_weight = (
                    0.05
                )

                size_weight = (
                    0.05
                )

            else:

                appearance_weight = (
                    self.appearance_weight
                )

                motion_weight = (
                    self.motion_weight
                )

                position_weight = (
                    self.position_weight
                )

                size_weight = (
                    self.size_weight
                )

            # ==========================================================
            # COMBINED SCORE
            # ==========================================================

            score = (

                appearance_weight
                *
                appearance_score

                +

                motion_weight
                *
                motion_score

                +

                position_weight
                *
                position_score

                +

                size_weight
                *
                size_score

            )

            # ----------------------------------------------------------
            # Hard rejection
            # ----------------------------------------------------------

            if (
                appearance is not None
                and
                appearance_score
                <
                self.strong_rejection_threshold
            ):

                # Motion alone cannot identify a cow.
                if motion_score < 0.75:

                    continue

            # ----------------------------------------------------------
            # Long disappearance
            # ----------------------------------------------------------

            if missed > 90:

                if (
                    appearance_score
                    <
                    self.strong_appearance_threshold
                ):

                    continue

            # ----------------------------------------------------------
            # Minimum evidence
            # ----------------------------------------------------------

            accepted = False

            # Strong appearance.
            if (
                appearance_score
                >=
                self.strong_appearance_threshold
            ):

                accepted = True

            # Normal appearance + reasonable motion.
            elif (
                appearance_score
                >=
                self.appearance_threshold
                and
                (
                    motion_score >= 0.25
                    or
                    position_score >= 0.35
                )
            ):

                accepted = True

            if not accepted:

                continue

            # ----------------------------------------------------------
            # Keep highest score
            # ----------------------------------------------------------

            if score > best_score:

                best_score = score

                best_cow = cow

                best_appearance = (
                    appearance_score
                )

                best_motion = (
                    motion_score
                )

        return (
            best_cow,
            best_score,
            best_appearance,
            best_motion
        )


    # ========================================================================
    # UPDATE EXISTING COW
    # ========================================================================

    def _update_cow(
        self,
        cow,
        tracker_id,
        appearance,
        box,
        frame_number,
        reconnection=False,
        match_score=None
    ):

        old_tracker = (
            cow.current_tracker_id
        )

        new_center = box_center(
            box
        )

        # ==============================================================
        # VELOCITY
        # ==============================================================

        if (
            cow.last_center is not None
            and
            new_center is not None
        ):

            delta = (
                new_center
                -
                cow.last_center
            )

            frame_delta = max(
                1,
                frame_number
                -
                cow.last_frame
            )

            instantaneous_velocity = (
                delta
                /
                frame_delta
            )

            if cow.velocity is None:

                cow.velocity = (
                    instantaneous_velocity
                    .astype(
                        np.float32
                    )
                )

            else:

                # Smooth velocity.
                cow.velocity = (
                    0.70
                    *
                    cow.velocity
                    +
                    0.30
                    *
                    instantaneous_velocity
                )

        # ==============================================================
        # TRACKER HISTORY
        # ==============================================================

        if (
            old_tracker is not None
            and
            old_tracker != tracker_id
        ):

            cow.previous_tracker_id = (
                old_tracker
            )

        if (
            not cow.tracker_history
            or
            cow.tracker_history[-1]
            !=
            tracker_id
        ):

            cow.tracker_history.append(
                int(tracker_id)
            )

        if len(
            cow.tracker_history
        ) > 100:

            cow.tracker_history = (
                cow.tracker_history[-100:]
            )

        # ==============================================================
        # APPEARANCE UPDATE
        # ==============================================================

        feature = normalize_feature(
            appearance
        )

        if feature is not None:

            if cow.appearance is None:

                cow.appearance = (
                    feature.copy()
                )

            else:

                alpha = (
                    self.appearance_update_rate
                )

                updated = (
                    (1.0 - alpha)
                    *
                    cow.appearance
                    +
                    alpha
                    *
                    feature
                )

                cow.appearance = (
                    normalize_feature(
                        updated
                    )
                )

            # Store feature sample.
            cow.appearance_history.append(
                feature.copy()
            )

            if len(
                cow.appearance_history
            ) > self.max_appearance_history:

                cow.appearance_history = (
                    cow.appearance_history[
                        -self.max_appearance_history:
                    ]
                )

            cow.updates += 1

            self.total_updates += 1

            self.updates += 1

        # ==============================================================
        # UPDATE POSITION
        # ==============================================================

        if box is not None:

            if cow.last_center is not None:

                cow.previous_center = (
                    cow.last_center.copy()
                )

            cow.last_box = normalize_box(
                box
            )

            cow.last_center = (
                new_center
                if new_center is not None
                else cow.last_center
            )

        # ==============================================================
        # STATE
        # ==============================================================

        cow.current_tracker_id = (
            int(tracker_id)
        )

        cow.last_frame = int(
            frame_number
        )

        cow.missed_frames = 0

        cow.observations += 1

        cow.active = True

        # ==============================================================
        # MAP
        # ==============================================================

        self.tracker_to_cow[
            int(tracker_id)
        ] = cow.cow_id

        # ==============================================================
        # CURRENT FRAME ASSIGNMENT
        # ==============================================================

        self.current_tracker_ids.add(
            int(tracker_id)
        )

        self.current_cow_ids.add(
            cow.cow_id
        )

        # ==============================================================
        # RECONNECTION
        # ==============================================================

        if (
            reconnection
            and
            old_tracker != tracker_id
        ):

            cow.reconnections += 1

            self.total_reconnections += 1

            self.reconnections += 1

            if self.verbose:

                if match_score is None:

                    print(
                        f"[RECONNECTED] "
                        f"Tracker {tracker_id} "
                        f"-> COW {cow.cow_id} "
                        f"| Previous tracker "
                        f"{old_tracker}"
                    )

                else:

                    print(
                        f"[RECONNECTED] "
                        f"Tracker {tracker_id} "
                        f"-> COW {cow.cow_id} "
                        f"| Match={match_score:.3f} "
                        f"| Previous tracker "
                        f"{old_tracker}"
                    )

        return cow.cow_id


    # ========================================================================
    # MAIN PUBLIC API
    # ========================================================================

    def update_detection(
        self,
        tracker_id=None,
        appearance=None,
        box=None,
        frame_number=None,
        confidence=0.0,
        **kwargs
    ):

        # --------------------------------------------------------------
        # Compatibility aliases
        # --------------------------------------------------------------

        if tracker_id is None:

            tracker_id = kwargs.get(
                "track_id",
                kwargs.get(
                    "id",
                    kwargs.get(
                        "tracker",
                        None
                    )
                )
            )

        if appearance is None:

            appearance = kwargs.get(
                "feature",
                kwargs.get(
                    "appearance_feature",
                    kwargs.get(
                        "embedding",
                        None
                    )
                )
            )

        if box is None:

            box = kwargs.get(
                "bbox",
                kwargs.get(
                    "xyxy",
                    None
                )
            )

        if frame_number is None:

            frame_number = kwargs.get(
                "frame",
                kwargs.get(
                    "frame_id",
                    self.current_frame + 1
                )
            )

        if tracker_id is None:

            raise ValueError(
                "tracker_id is required"
            )

        tracker_id = int(
            tracker_id
        )

        frame_number = int(
            frame_number
        )

        self.current_frame = max(
            self.current_frame,
            frame_number
        )

        box = normalize_box(
            box
        )

        # ==============================================================
        # 1. SAME TRACKER
        # ==============================================================

        cow_id = self.tracker_to_cow.get(
            tracker_id
        )

        if cow_id is not None:

            cow = self.cows.get(
                cow_id
            )

            if cow is not None:

                return self._update_cow(

                    cow=cow,

                    tracker_id=tracker_id,

                    appearance=appearance,

                    box=box,

                    frame_number=frame_number,

                    reconnection=False
                )

        # ==============================================================
        # 2. FIND OLD COW
        # ==============================================================

        (
            best_cow,
            score,
            appearance_score,
            motion_score
        ) = self._find_best_match(

            tracker_id=tracker_id,

            appearance=appearance,

            box=box,

            frame_number=frame_number
        )

        # ==============================================================
        # 3. RECONNECT
        # ==============================================================

        if best_cow is not None:

            # Important:
            # Once assigned in this frame, don't reuse it.
            if (
                best_cow.cow_id
                in
                self.current_cow_ids
            ):

                best_cow = None

            else:

                return self._update_cow(

                    cow=best_cow,

                    tracker_id=tracker_id,

                    appearance=appearance,

                    box=box,

                    frame_number=frame_number,

                    reconnection=True,

                    match_score=appearance_score
                )

        # ==============================================================
        # 4. NEW COW
        # ==============================================================

        return self._create_cow(

            tracker_id=tracker_id,

            box=box,

            appearance=appearance,

            frame_number=frame_number
        )


    # ========================================================================
    # END FRAME
    # ========================================================================

    def end_frame(
        self,
        frame_number=None,
        active_tracker_ids=None,
        *args,
        **kwargs
    ):
        """
        Flexible API.

        Supports:

            end_frame()

            end_frame(10)

            end_frame({1, 2, 3})

            end_frame(10, {1, 2, 3})

            end_frame(
                frame_number=10,
                active_tracker_ids={1,2,3}
            )
        """

        # --------------------------------------------------------------
        # First positional argument can accidentally be the tracker set.
        # --------------------------------------------------------------

        if isinstance(
            frame_number,
            (
                set,
                list,
                tuple
            )
        ):

            if active_tracker_ids is None:

                active_tracker_ids = (
                    frame_number
                )

            frame_number = (
                self.current_frame
            )

        # --------------------------------------------------------------
        # Handle optional positional args
        # --------------------------------------------------------------

        if len(args) > 0:

            first_extra = args[0]

            if active_tracker_ids is None:

                if isinstance(
                    first_extra,
                    (
                        set,
                        list,
                        tuple,
                        np.ndarray
                    )
                ):

                    active_tracker_ids = (
                        first_extra
                    )

            if frame_number is None:

                if isinstance(
                    first_extra,
                    (
                        int,
                        np.integer,
                        float,
                        np.floating
                    )
                ):

                    frame_number = int(
                        first_extra
                    )

        # --------------------------------------------------------------
        # Frame aliases
        # --------------------------------------------------------------

        if frame_number is None:

            frame_number = kwargs.get(
                "frame",
                kwargs.get(
                    "frame_id",
                    self.current_frame
                )
            )

        frame_number = int(
            frame_number
        )

        self.current_frame = max(
            self.current_frame,
            frame_number
        )

        # --------------------------------------------------------------
        # Active trackers
        # --------------------------------------------------------------

        if active_tracker_ids is None:

            active_tracker_ids = set(
                self.current_tracker_ids
            )

        elif isinstance(
            active_tracker_ids,
            (
                int,
                np.integer
            )
        ):

            active_tracker_ids = {
                int(active_tracker_ids)
            }

        else:

            try:

                active_tracker_ids = {
                    int(x)
                    for x in active_tracker_ids
                }

            except Exception:

                active_tracker_ids = set()

        # ==============================================================
        # UPDATE ALL COWS
        # ==============================================================

        for cow in self.cows.values():

            if (
                cow.current_tracker_id
                in
                active_tracker_ids
            ):

                cow.active = True

                cow.missed_frames = 0

            else:

                cow.active = False

                cow.missed_frames = max(
                    0,
                    frame_number
                    -
                    cow.last_frame
                )

        # ==============================================================
        # KEEP PERMANENT COW DATABASE
        # ==============================================================
        #
        # We deliberately do NOT delete identities.
        #
        # A cow can disappear and later be recovered.
        #

        # ==============================================================
        # RESET FRAME ASSIGNMENTS
        # ==============================================================

        self.current_tracker_ids.clear()

        self.current_cow_ids.clear()


    # ========================================================================
    # GET COW
    # ========================================================================

    def get_cow(
        self,
        cow_id
    ):

        try:

            cow_id = int(
                cow_id
            )

        except Exception:

            return None

        return self.cows.get(
            cow_id
        )


    # ========================================================================
    # GET COW ID
    # ========================================================================

    def get_cow_id(
        self,
        tracker_id
    ):

        try:

            tracker_id = int(
                tracker_id
            )

        except Exception:

            return None

        return self.tracker_to_cow.get(
            tracker_id
        )


    # ========================================================================
    # ACTIVE COWS
    # ========================================================================

    def get_active_cows(self):

        return [
            cow
            for cow in self.cows.values()
            if cow.active
        ]


    # ========================================================================
    # TOTAL KNOWN COWS
    # ========================================================================

    def total_known_cows(self):

        return len(
            self.cows
        )


    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(self):

        active = sum(
            1
            for cow in self.cows.values()
            if cow.active
        )

        return {

            "active_cows":
                active,

            "total_known_cows":
                len(self.cows),

            "total_new_cows":
                self.total_new_cows,

            "new_cows":
                self.total_new_cows,

            "total_reconnections":
                self.total_reconnections,

            "reconnections":
                self.total_reconnections,

            "total_updates":
                self.total_updates,

            "updates":
                self.total_updates,

            "total_observations":
                self.total_observations,

            "current_frame":
                self.current_frame,

        }


    # ========================================================================
    # STATISTICS PROPERTY
    # ========================================================================

    @property
    def statistics(self):

        return self.get_statistics()


    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self):

        self.cows.clear()

        self.tracker_to_cow.clear()

        self.next_cow_id = 1

        self.current_frame = 0

        self.current_tracker_ids.clear()

        self.current_cow_ids.clear()

        self.total_new_cows = 0

        self.total_reconnections = 0

        self.total_updates = 0

        self.total_observations = 0

        self.new_cows = 0

        self.reconnections = 0

        self.updates = 0


# ============================================================================
# V6 SELF TEST
# ============================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("IDENTITY MANAGER V6 SELF TEST")
    print("=" * 70)

    manager = IdentityManager(
        appearance_threshold=0.70,
        strong_appearance_threshold=0.86,
        strong_rejection_threshold=0.45,
        max_distance=300,
        max_missed_frames=180,
        verbose=True
    )

    # ------------------------------------------------------------
    # Cow 1 appearance
    # ------------------------------------------------------------

    feature_a = np.ones(
        128,
        dtype=np.float32
    )

    feature_a /= np.linalg.norm(
        feature_a
    )

    # ------------------------------------------------------------
    # Cow 2 appearance
    # ------------------------------------------------------------

    feature_b = np.zeros(
        128,
        dtype=np.float32
    )

    feature_b[0] = 1.0

    # ------------------------------------------------------------
    # Frame 1
    # ------------------------------------------------------------

    manager.start_frame(1)

    cow1 = manager.update_detection(

        tracker_id=76,

        appearance=feature_a,

        box=[
            100,
            100,
            200,
            200
        ],

        frame_number=1

    )

    manager.end_frame(
        1,
        {76}
    )

    # ------------------------------------------------------------
    # Frame 2
    # ------------------------------------------------------------

    manager.start_frame(2)

    cow1_again = manager.update_detection(

        tracker_id=76,

        appearance=feature_a,

        box=[
            110,
            105,
            210,
            205
        ],

        frame_number=2

    )

    manager.end_frame(
        2,
        {76}
    )

    # ------------------------------------------------------------
    # Frame 3
    # Tracker changes
    # ------------------------------------------------------------

    manager.start_frame(3)

    cow1_reconnected = (
        manager.update_detection(

            tracker_id=184,

            appearance=feature_a,

            box=[
                120,
                110,
                220,
                210
            ],

            frame_number=3
        )
    )

    manager.end_frame(
        3,
        {184}
    )

    # ------------------------------------------------------------
    # Frame 4
    # Different cow
    # ------------------------------------------------------------

    manager.start_frame(4)

    cow2 = manager.update_detection(

        tracker_id=200,

        appearance=feature_b,

        box=[
            700,
            700,
            800,
            800
        ],

        frame_number=4
    )

    manager.end_frame(
        4,
        {200}
    )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    stats = (
        manager.get_statistics()
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Tracker 76  -> COW {cow1}"
    )

    print(
        f"Tracker 76  -> COW {cow1_again}"
    )

    print(
        f"Tracker 184 -> COW {cow1_reconnected}"
    )

    print(
        f"Tracker 200 -> COW {cow2}"
    )

    print()

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

    cow1_obj = manager.get_cow(
        1
    )

    if cow1_obj is not None:

        print(
            f"COW 1 observations : "
            f"{cow1_obj.observation_count}"
        )

    print()

    if (
        cow1 == 1
        and
        cow1_again == 1
        and
        cow1_reconnected == 1
        and
        cow2 == 2
        and
        stats["total_known_cows"] == 2
    ):

        print(
            "PASS: V6 identity manager works."
        )

    else:

        print(
            "FAIL: V6 identity manager needs investigation."
        )

    print(
        "=" * 70
    )