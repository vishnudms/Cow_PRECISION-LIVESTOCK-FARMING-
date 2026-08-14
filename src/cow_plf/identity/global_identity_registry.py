"""
===============================================================================
GLOBAL COW IDENTITY REGISTRY
===============================================================================

Permanent cross-video / cross-day identity.

V5:
    Local identity inside one video.

V6:
    Global identity across videos and days.

Example:

    Day 1:
        local Cow 1 -> COW-0001

    Day 2:
        local Cow 7 -> COW-0001

    Day 3:
        local Cow 3 -> COW-0001

Important:
    Matching is based on appearance embeddings.
    A real-world deployment should combine appearance with additional cues
    and confidence checks before treating an identity as certain.
===============================================================================
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np


class GlobalCowIdentityRegistry:

    def __init__(
        self,
        database_path="database/global_cows.json",
        similarity_threshold=0.82,
        max_embeddings_per_cow=20,
    ):

        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.similarity_threshold = float(
            similarity_threshold
        )

        self.max_embeddings_per_cow = int(
            max_embeddings_per_cow
        )

        self.cows = {}

        self.next_global_id = 1

        self.load()

    # =========================================================================
    # GLOBAL ID
    # =========================================================================

    def _new_global_id(self):

        global_id = (
            f"COW-{self.next_global_id:04d}"
        )

        self.next_global_id += 1

        return global_id

    # =========================================================================
    # NORMALIZE
    # =========================================================================

    @staticmethod
    def _normalize_embedding(embedding):

        if embedding is None:
            return None

        try:
            vector = np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)

        except Exception:
            return None

        if vector.size == 0:
            return None

        norm = np.linalg.norm(vector)

        if norm <= 1e-12:
            return None

        return vector / norm

    # =========================================================================
    # COSINE
    # =========================================================================

    @staticmethod
    def cosine_similarity(a, b):

        a = np.asarray(
            a,
            dtype=np.float32
        ).reshape(-1)

        b = np.asarray(
            b,
            dtype=np.float32
        ).reshape(-1)

        if a.size == 0 or b.size == 0:
            return 0.0

        if a.shape != b.shape:
            return 0.0

        denominator = (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )

        if denominator <= 1e-12:
            return 0.0

        return float(
            np.dot(a, b)
            /
            denominator
        )

    # =========================================================================
    # LOAD
    # =========================================================================

    def load(self):

        if not self.database_path.exists():

            self.cows = {}
            self.next_global_id = 1
            return

        try:

            with open(
                self.database_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            self.next_global_id = int(
                data.get(
                    "next_global_id",
                    1
                )
            )

            self.cows = data.get(
                "cows",
                {}
            )

        except Exception as exc:

            print(
                f"[GLOBAL ID] Database load failed: {exc}"
            )

            self.cows = {}
            self.next_global_id = 1

    # =========================================================================
    # SAVE
    # =========================================================================

    def save(self):

        data = {
            "next_global_id": self.next_global_id,
            "cows": self.cows
        }

        temp_path = (
            self.database_path.with_suffix(".tmp")
        )

        try:

            with open(
                temp_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    data,
                    file,
                    indent=2
                )

            temp_path.replace(
                self.database_path
            )

        except Exception as exc:

            print(
                f"[GLOBAL ID] Database save failed: {exc}"
            )

    # =========================================================================
    # CREATE NEW COW
    # =========================================================================

    def create_cow(
        self,
        embedding,
        source_video="",
        local_cow_id=None,
        frame_number=0
    ):

        vector = self._normalize_embedding(
            embedding
        )

        if vector is None:
            return None

        global_id = self._new_global_id()

        timestamp = datetime.now().isoformat()

        self.cows[global_id] = {

            "global_id": global_id,

            "created_at": timestamp,

            "last_seen": timestamp,

            "observation_count": 1,

            "videos": (
                [source_video]
                if source_video
                else []
            ),

            "local_ids": (
                [int(local_cow_id)]
                if local_cow_id is not None
                else []
            ),

            "last_frame": int(frame_number),

            "embeddings": [
                vector.tolist()
            ]
        }

        self.save()

        return global_id

    # =========================================================================
    # MATCH EXISTING GLOBAL COW
    # =========================================================================

    def match(
        self,
        embedding,
        excluded_global_ids=None
    ):

        vector = self._normalize_embedding(
            embedding
        )

        if vector is None:
            return None, 0.0

        if excluded_global_ids is None:
            excluded_global_ids = set()

        best_id = None

        best_score = -1.0

        for global_id, cow in self.cows.items():

            if global_id in excluded_global_ids:
                continue

            for stored_embedding in cow.get(
                "embeddings",
                []
            ):

                stored = np.asarray(
                    stored_embedding,
                    dtype=np.float32
                )

                score = self.cosine_similarity(
                    vector,
                    stored
                )

                if score > best_score:

                    best_score = score
                    best_id = global_id

        if (
            best_id is not None
            and
            best_score >= self.similarity_threshold
        ):

            return best_id, best_score

        return None, best_score

    # =========================================================================
    # ADD OBSERVATION
    # =========================================================================

    def add_observation(
        self,
        global_id,
        embedding,
        source_video="",
        local_cow_id=None,
        frame_number=0
    ):

        if global_id not in self.cows:
            return False

        vector = self._normalize_embedding(
            embedding
        )

        if vector is None:
            return False

        cow = self.cows[global_id]

        embeddings = cow.setdefault(
            "embeddings",
            []
        )

        embeddings.append(
            vector.tolist()
        )

        if len(embeddings) > (
            self.max_embeddings_per_cow
        ):

            del embeddings[
                :-self.max_embeddings_per_cow
            ]

        cow[
            "observation_count"
        ] = (
            int(
                cow.get(
                    "observation_count",
                    0
                )
            )
            + 1
        )

        cow[
            "last_seen"
        ] = datetime.now().isoformat()

        cow[
            "last_frame"
        ] = int(frame_number)

        if (
            source_video
            and
            source_video not in cow.get(
                "videos",
                []
            )
        ):

            cow.setdefault(
                "videos",
                []
            ).append(
                source_video
            )

        if (
            local_cow_id is not None
            and
            int(local_cow_id)
            not in cow.get(
                "local_ids",
                []
            )
        ):

            cow.setdefault(
                "local_ids",
                []
            ).append(
                int(local_cow_id)
            )

        self.save()

        return True

    # =========================================================================
    # IDENTIFY
    # =========================================================================

    def identify(
        self,
        embedding,
        source_video="",
        local_cow_id=None,
        frame_number=0,
        excluded_global_ids=None
    ):

        global_id, score = self.match(
            embedding,
            excluded_global_ids=(
                excluded_global_ids
            )
        )

        if global_id is not None:

            self.add_observation(
                global_id=global_id,
                embedding=embedding,
                source_video=source_video,
                local_cow_id=local_cow_id,
                frame_number=frame_number
            )

            return (
                global_id,
                score,
                False
            )

        global_id = self.create_cow(
            embedding=embedding,
            source_video=source_video,
            local_cow_id=local_cow_id,
            frame_number=frame_number
        )

        return (
            global_id,
            1.0,
            True
        )

    # =========================================================================
    # GET
    # =========================================================================

    def get_cow(self, global_id):

        return self.cows.get(
            global_id
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def statistics(self):

        return {

            "total_global_cows":
                len(self.cows),

            "next_global_id":
                self.next_global_id,

            "database":
                str(
                    self.database_path
                )
        }


if __name__ == "__main__":

    print()
    print("=" * 75)
    print("GLOBAL COW IDENTITY REGISTRY")
    print("=" * 75)

    registry = GlobalCowIdentityRegistry()

    print()
    print(
        registry.statistics()
    )

    print()
    print(
        "Database:",
        registry.database_path
    )

    print()
    print("=" * 75)