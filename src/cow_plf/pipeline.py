"""
===============================================================================
COW PLF - PRODUCTION PIPELINE
===============================================================================

Pipeline

    Video
      |
      v
    YOLO26m-seg
      |
      v
    ByteTrack
      |
      v
    AppearanceExtractor
      |
      v
    V5 IdentityManager
      |
      v
    Local Cow ID
      |
      v
    Global Cow Registry
      |
      v
    Permanent Global Cow ID
      |
      +--------------------------+
      |                          |
      v                          v
 Morphometrics              Cow Profile
      |                          |
      +-------------+------------+
                    |
                    v
               CSV / Video

Important
---------
The V5 IdentityManager remains the authority for within-video identity.

The GlobalCowIdentityRegistry provides cross-video / cross-day identity.

This module does not modify the V5 identity algorithm.
===============================================================================
"""

from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np

from .vision.detector import CowDetector
from .core.appearance import AppearanceExtractor
from .core.identity_manager import IdentityManager
from .core.cow_profile import CowProfile

from .identity.global_identity_registry import (
    GlobalCowIdentityRegistry
)

from .analytics.morphometrics import (
    CowMorphometrics
)


class CowPLFPipeline:

    def __init__(
        self,
        project_root,
        model_path="models/yolo26m-seg.pt",
        global_database="database/global_cows.json",
        confidence=0.20,
        iou=0.50,
        image_size=1280,
        device=0,
        tracker="bytetrack.yaml",
        appearance_interval=3,
        min_crop_size=30,
        global_similarity_threshold=0.82,
        global_min_observations=3,
        global_max_embeddings=20,
        morphometric_interval=3,
        v5_max_distance=220,
        v5_max_missed_frames=600,
        v5_appearance_threshold=0.70,
        v5_strong_rejection_threshold=0.40,
    ):

        self.project_root = Path(
            project_root
        ).resolve()

        # =====================================================================
        # CONFIG
        # =====================================================================

        self.appearance_interval = int(
            appearance_interval
        )

        self.min_crop_size = int(
            min_crop_size
        )

        self.global_min_observations = int(
            global_min_observations
        )

        self.global_update_interval = 30

        self.morphometric_interval = int(
            morphometric_interval
        )

        # =====================================================================
        # VISION
        # =====================================================================

        self.detector = CowDetector(
            model_path=(
                self.project_root / model_path
            ),
            confidence=confidence,
            iou=iou,
            image_size=image_size,
            device=device,
            tracker=tracker,
        )

        # =====================================================================
        # APPEARANCE
        # =====================================================================

        self.appearance_extractor = (
            AppearanceExtractor()
        )

        # =====================================================================
        # V5 IDENTITY
        # =====================================================================

        self.identity_manager = IdentityManager(

            max_distance=v5_max_distance,

            max_missed_frames=(
                v5_max_missed_frames
            ),

            appearance_threshold=(
                v5_appearance_threshold
            ),

            strong_rejection_threshold=(
                v5_strong_rejection_threshold
            ),

            position_weight=0.25,

            appearance_weight=0.60,

            size_weight=0.15,
        )

        # =====================================================================
        # GLOBAL IDENTITY
        # =====================================================================

        self.global_registry = (
            GlobalCowIdentityRegistry(

                database_path=str(
                    self.project_root
                    / global_database
                ),

                similarity_threshold=(
                    global_similarity_threshold
                ),

                max_embeddings_per_cow=(
                    global_max_embeddings
                ),
            )
        )

        # =====================================================================
        # MORPHOMETRICS
        # =====================================================================

        self.morphometrics = (
            CowMorphometrics(
                min_mask_pixels=100
            )
        )

        # =====================================================================
        # PER-COW STATE
        # =====================================================================

        self.profiles = {}

        self.local_to_global = {}

        self.global_scores = {}

        self.appearance_cache = {}

        self.last_appearance_frame = {}

        self.embedding_buffers = defaultdict(
            list
        )

        self.last_morphometric_frame = {}

        self.reserved_global_ids = set()

        # =====================================================================
        # RUNTIME
        # =====================================================================

        self.frame_number = 0

        self.video_name = ""

        self.total_detections = 0

        self.total_measurements = 0

        self.total_global_assignments = 0

        self.total_new_global_cows = 0

        self.total_reused_global_cows = 0

    # =========================================================================
    # START VIDEO
    # =========================================================================

    def start_video(self, video_name):

        self.video_name = str(
            video_name
        )

        self.frame_number = 0

        self.profiles.clear()

        self.local_to_global.clear()

        self.global_scores.clear()

        self.appearance_cache.clear()

        self.last_appearance_frame.clear()

        self.embedding_buffers.clear()

        self.last_morphometric_frame.clear()

        self.reserved_global_ids.clear()

        self.total_detections = 0

        self.total_measurements = 0

        self.total_global_assignments = 0

        self.total_new_global_cows = 0

        self.total_reused_global_cows = 0

    # =========================================================================
    # APPEARANCE
    # =========================================================================

    def _get_appearance(
        self,
        frame,
        detection
    ):

        tracker_id = detection.tracker_id

        x1, y1, x2, y2 = detection.box

        crop = frame[
            y1:y2,
            x1:x2
        ]

        appearance = None

        should_extract = (

            tracker_id not in
            self.last_appearance_frame

            or

            (
                self.frame_number
                -
                self.last_appearance_frame[
                    tracker_id
                ]
            )
            >=
            self.appearance_interval
        )

        if (

            crop is not None
            and
            crop.size > 0
            and
            (x2 - x1) >= self.min_crop_size
            and
            (y2 - y1) >= self.min_crop_size
            and
            should_extract
        ):

            try:

                appearance = (
                    self.appearance_extractor.extract(
                        crop
                    )
                )

            except Exception as exc:

                print(
                    f"[WARNING] "
                    f"Appearance extraction failed "
                    f"for tracker {tracker_id}: "
                    f"{exc}"
                )

                appearance = None

            if appearance is not None:

                self.appearance_cache[
                    tracker_id
                ] = appearance

                self.last_appearance_frame[
                    tracker_id
                ] = self.frame_number

        if appearance is None:

            appearance = (
                self.appearance_cache.get(
                    tracker_id
                )
            )

        return appearance

    # =========================================================================
    # EMBEDDING BUFFER
    # =========================================================================

    def _update_embedding_buffer(
        self,
        local_cow_id,
        appearance
    ):

        if appearance is None:
            return

        try:

            vector = np.asarray(
                appearance,
                dtype=np.float32
            ).reshape(-1)

            norm = np.linalg.norm(
                vector
            )

            if norm <= 1e-12:
                return

            vector = (
                vector /
                norm
            )

            self.embedding_buffers[
                local_cow_id
            ].append(
                vector
            )

            if len(
                self.embedding_buffers[
                    local_cow_id
                ]
            ) > 5:

                self.embedding_buffers[
                    local_cow_id
                ] = (
                    self.embedding_buffers[
                        local_cow_id
                    ][-5:]
                )

        except Exception:
            return

    # =========================================================================
    # GLOBAL ID
    # =========================================================================

    def _assign_global_id(
        self,
        local_cow_id
    ):

        if local_cow_id in self.local_to_global:

            return (
                self.local_to_global[
                    local_cow_id
                ]
            )

        samples = (
            self.embedding_buffers[
                local_cow_id
            ]
        )

        if len(samples) < (
            self.global_min_observations
        ):

            return None

        try:

            mean_embedding = np.mean(
                np.stack(samples),
                axis=0
            )

            norm = np.linalg.norm(
                mean_embedding
            )

            if norm <= 1e-12:
                return None

            mean_embedding = (
                mean_embedding /
                norm
            )

            (
                global_id,
                score,
                created
            ) = self.global_registry.identify(

                embedding=mean_embedding,

                source_video=(
                    self.video_name
                ),

                local_cow_id=(
                    local_cow_id
                ),

                frame_number=(
                    self.frame_number
                ),

                excluded_global_ids=(
                    self.reserved_global_ids
                )
            )

        except Exception as exc:

            print(
                f"[WARNING] Global ID matching failed "
                f"for local Cow {local_cow_id}: "
                f"{exc}"
            )

            return None

        if global_id is None:
            return None

        self.local_to_global[
            local_cow_id
        ] = global_id

        self.global_scores[
            local_cow_id
        ] = float(score)

        self.reserved_global_ids.add(
            global_id
        )

        self.total_global_assignments += 1

        if created:

            self.total_new_global_cows += 1

            print(
                f"[GLOBAL NEW] "
                f"Local Cow {local_cow_id} "
                f"-> {global_id} "
                f"| Score={score:.4f}"
            )

        else:

            self.total_reused_global_cows += 1

            print(
                f"[GLOBAL MATCH] "
                f"Local Cow {local_cow_id} "
                f"-> {global_id} "
                f"| Score={score:.4f}"
            )

        return global_id

    # =========================================================================
    # PROCESS FRAME
    # =========================================================================

    def process_frame(
        self,
        frame,
        csv_writer=None
    ):

        self.frame_number += 1

        self.identity_manager.start_frame(
            self.frame_number
        )

        current_cow_ids = set()

        detections = (
            self.detector.predict(
                frame
            )
        )

        results = []

        # =====================================================================
        # DETECTIONS
        # =====================================================================

        for detection in detections:

            appearance = (
                self._get_appearance(
                    frame,
                    detection
                )
            )

            # ================================================================
            # V5 IDENTITY
            # ================================================================

            cow_id = (
                self.identity_manager
                .update_detection(

                    tracker_id=(
                        detection.tracker_id
                    ),

                    center=(
                        detection.center
                    ),

                    box=(
                        detection.box
                    ),

                    confidence=(
                        detection.confidence
                    ),

                    frame_number=(
                        self.frame_number
                    ),

                    used_cow_ids=(
                        current_cow_ids
                    ),

                    appearance=appearance,
                )
            )

            if cow_id is None:
                continue

            cow_id = int(
                cow_id
            )

            current_cow_ids.add(
                cow_id
            )

            self.total_detections += 1

            # ================================================================
            # PROFILE
            # ================================================================

            if cow_id not in self.profiles:

                self.profiles[
                    cow_id
                ] = CowProfile(
                    cow_id=cow_id
                )

                self.profiles[
                    cow_id
                ].add_event(
                    "LOCAL_COW_CREATED",
                    self.frame_number,
                    {
                        "tracker_id":
                            detection.tracker_id
                    }
                )

            profile = (
                self.profiles[
                    cow_id
                ]
            )

            profile.tracker_id = (
                detection.tracker_id
            )

            profile.identity_confidence = (
                detection.confidence
            )

            profile.change_state(
                "ACTIVE",
                self.frame_number
            )

            profile.update_position(
                detection.center[0],
                detection.center[1]
            )

            profile.update_seen(
                self.frame_number
            )

            # ================================================================
            # EMBEDDING
            # ================================================================

            self._update_embedding_buffer(
                cow_id,
                appearance
            )

            # ================================================================
            # GLOBAL ID
            # ================================================================

            global_id = (
                self._assign_global_id(
                    cow_id
                )
            )

            # ================================================================
            # PERIODIC GLOBAL EMBEDDING UPDATE
            # ================================================================

            if (
                global_id is not None
                and
                appearance is not None
                and
                self.frame_number
                %
                self.global_update_interval
                == 0
            ):

                self.global_registry.add_observation(

                    global_id=global_id,

                    embedding=appearance,

                    source_video=(
                        self.video_name
                    ),

                    local_cow_id=cow_id,

                    frame_number=(
                        self.frame_number
                    )
                )

            # ================================================================
            # MORPHOMETRICS
            # ================================================================

            features = None

            should_measure = (

                cow_id not in
                self.last_morphometric_frame

                or

                (
                    self.frame_number
                    -
                    self.last_morphometric_frame[
                        cow_id
                    ]
                )
                >=
                self.morphometric_interval
            )

            if (
                should_measure
                and
                detection.mask is not None
            ):

                try:

                    features = (
                        self.morphometrics.extract(

                            mask=detection.mask,

                            bbox=detection.box,

                            frame_width=(
                                frame.shape[1]
                            ),

                            frame_height=(
                                frame.shape[0]
                            ),

                            clean_mask=True
                        )
                    )

                except Exception as exc:

                    print(
                        f"[WARNING] "
                        f"Morphometrics failed "
                        f"for Cow {cow_id}: "
                        f"{exc}"
                    )

                    features = None

            if features is not None:

                self.last_morphometric_frame[
                    cow_id
                ] = self.frame_number

                self.total_measurements += 1

                profile.body_area = (
                    features[
                        "mask_area_px"
                    ]
                )

                profile.body_length = (
                    features[
                        "body_length_px"
                    ]
                )

                profile.body_width = (
                    features[
                        "body_width_px"
                    ]
                )

                profile.add_event(
                    "MORPHOMETRICS",
                    self.frame_number,
                    {
                        "mask_area_px":
                            features[
                                "mask_area_px"
                            ],

                        "body_length_px":
                            features[
                                "body_length_px"
                            ],

                        "body_width_px":
                            features[
                                "body_width_px"
                            ]
                    }
                )

            # ================================================================
            # CSV
            # ================================================================

            if (
                csv_writer is not None
                and
                features is not None
            ):

                row = {

                    "timestamp":
                        datetime.now().isoformat(
                            timespec="milliseconds"
                        ),

                    "video":
                        self.video_name,

                    "frame":
                        self.frame_number,

                    "tracker_id":
                        detection.tracker_id,

                    "local_cow_id":
                        cow_id,

                    "global_cow_id":
                        (
                            global_id
                            or
                            "PENDING"
                        ),

                    "confidence":
                        detection.confidence,

                    "global_score":
                        self.global_scores.get(
                            cow_id,
                            0.0
                        ),
                }

                row.update(
                    features
                )

                csv_writer.writerow(row)

            # ================================================================
            # RESULT
            # ================================================================

            results.append({

                "local_cow_id":
                    cow_id,

                "global_cow_id":
                    global_id,

                "tracker_id":
                    detection.tracker_id,

                "confidence":
                    detection.confidence,

                "box":
                    detection.box,

                "center":
                    detection.center,

                "mask":
                    detection.mask,

                "morphometrics":
                    features,

                "profile":
                    profile,
            })

        # =====================================================================
        # V5 FRAME END
        # =====================================================================

        self.identity_manager.end_frame(

            active_cow_ids=(
                current_cow_ids
            ),

            frame_number=(
                self.frame_number
            )
        )

        return results

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summary(self):

        return {

            "video":
                self.video_name,

            "frames":
                self.frame_number,

            "detections":
                self.total_detections,

            "measurements":
                self.total_measurements,

            "local_cows":
                len(
                    self.profiles
                ),

            "global_assignments":
                self.total_global_assignments,

            "new_global_cows":
                self.total_new_global_cows,

            "reused_global_cows":
                self.total_reused_global_cows,
        }

    # =========================================================================
    # CLOSE
    # =========================================================================

    def close(self):

        self.global_registry.save()