"""
============================================================
V6 COW PROFILE
============================================================

Stores all information associated with one persistent Cow ID.

This is the central data structure that future modules will use:

    Identity
    Tracking
    Morphometrics
    Behavior
    Welfare
    Social Dynamics
    Flow Analytics
    Events

IMPORTANT:
This module does NOT create or change Cow IDs.
Your existing V5 Identity Manager remains responsible for that.
============================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import math


@dataclass
class CowProfile:

    # ========================================================
    # IDENTITY
    # ========================================================

    cow_id: int

    tracker_id: Optional[int] = None

    identity_confidence: float = 0.0

    first_seen_frame: int = 0
    last_seen_frame: int = 0

    first_seen_time: float = field(
        default_factory=time.time
    )

    last_seen_time: float = field(
        default_factory=time.time
    )

    total_frames: int = 0

    # ========================================================
    # TRACK STATE
    # ========================================================

    state: str = "NEW"

    missing_frames: int = 0

    # ========================================================
    # POSITION
    # ========================================================

    center_x: float = 0.0
    center_y: float = 0.0

    previous_x: float = 0.0
    previous_y: float = 0.0

    # ========================================================
    # MOVEMENT
    # ========================================================

    distance_travelled: float = 0.0

    current_speed: float = 0.0

    maximum_speed: float = 0.0

    # ========================================================
    # FLOW / ZONES
    # ========================================================

    current_zone: str = "UNKNOWN"

    previous_zone: str = "UNKNOWN"

    zone_enter_frame: Optional[int] = None

    zone_exit_frame: Optional[int] = None

    zone_time: Dict[str, float] = field(
        default_factory=dict
    )

    # ========================================================
    # BEHAVIOR
    # Future module
    # ========================================================

    behavior: str = "UNKNOWN"

    behavior_confidence: float = 0.0

    behavior_history: List[dict] = field(
        default_factory=list
    )

    # ========================================================
    # MORPHOMETRICS
    # Future module
    # ========================================================

    body_area: float = 0.0

    body_length: float = 0.0

    body_width: float = 0.0

    body_height: float = 0.0

    body_volume: float = 0.0

    # ========================================================
    # WELFARE
    # Future module
    # ========================================================

    welfare_score: float = 0.0

    welfare_status: str = "UNKNOWN"

    # ========================================================
    # SOCIAL DYNAMICS
    # Future module
    # ========================================================

    nearby_cows: List[int] = field(
        default_factory=list
    )

    interaction_history: List[dict] = field(
        default_factory=list
    )

    # ========================================================
    # EVENTS
    # ========================================================

    events: List[dict] = field(
        default_factory=list
    )

    # ========================================================
    # POSITION UPDATE
    # ========================================================

    def update_position(
        self,
        x: float,
        y: float
    ):

        self.previous_x = self.center_x
        self.previous_y = self.center_y

        self.center_x = float(x)
        self.center_y = float(y)

        # If this is not the first position
        if (
            self.previous_x != 0
            or self.previous_y != 0
        ):

            dx = (
                self.center_x
                - self.previous_x
            )

            dy = (
                self.center_y
                - self.previous_y
            )

            distance = math.sqrt(
                dx * dx + dy * dy
            )

            self.distance_travelled += distance

            self.current_speed = distance

            if (
                self.current_speed
                > self.maximum_speed
            ):

                self.maximum_speed = (
                    self.current_speed
                )

    # ========================================================
    # SEEN UPDATE
    # ========================================================

    def update_seen(
        self,
        frame_number: int
    ):

        if self.first_seen_frame == 0:

            self.first_seen_frame = frame_number

            self.first_seen_time = time.time()

        self.last_seen_frame = frame_number

        self.last_seen_time = time.time()

        self.total_frames += 1

        self.missing_frames = 0

    # ========================================================
    # MISSING UPDATE
    # ========================================================

    def update_missing(self):

        self.missing_frames += 1

    # ========================================================
    # STATE CHANGE
    # ========================================================

    def change_state(
        self,
        new_state: str,
        frame_number: int
    ):

        if self.state == new_state:
            return

        old_state = self.state

        self.state = new_state

        self.add_event(
            event_type="STATE_CHANGE",
            frame_number=frame_number,
            data={
                "from": old_state,
                "to": new_state
            }
        )

    # ========================================================
    # ZONE CHANGE
    # ========================================================

    def change_zone(
        self,
        new_zone: str,
        frame_number: int
    ):

        if new_zone == self.current_zone:
            return False

        old_zone = self.current_zone

        self.previous_zone = old_zone

        self.current_zone = new_zone

        self.zone_enter_frame = frame_number

        self.add_event(
            event_type="ZONE_CHANGE",
            frame_number=frame_number,
            data={
                "from": old_zone,
                "to": new_zone
            }
        )

        return True

    # ========================================================
    # BEHAVIOR UPDATE
    # ========================================================

    def update_behavior(
        self,
        behavior: str,
        confidence: float,
        frame_number: int
    ):

        if behavior != self.behavior:

            self.behavior_history.append({
                "behavior": behavior,
                "confidence": confidence,
                "frame": frame_number,
                "time": time.time()
            })

        self.behavior = behavior

        self.behavior_confidence = float(
            confidence
        )

    # ========================================================
    # EVENT
    # ========================================================

    def add_event(
        self,
        event_type: str,
        frame_number: int,
        data=None
    ):

        event = {
            "event": event_type,
            "frame": frame_number,
            "time": time.time(),
            "data": data or {}
        }

        self.events.append(event)

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self):

        return {

            "cow_id": self.cow_id,

            "tracker_id": self.tracker_id,

            "state": self.state,

            "identity_confidence":
                self.identity_confidence,

            "first_seen_frame":
                self.first_seen_frame,

            "last_seen_frame":
                self.last_seen_frame,

            "total_frames":
                self.total_frames,

            "center": (
                self.center_x,
                self.center_y
            ),

            "distance_travelled":
                self.distance_travelled,

            "speed":
                self.current_speed,

            "zone":
                self.current_zone,

            "behavior":
                self.behavior,

            "welfare":
                self.welfare_status
        }