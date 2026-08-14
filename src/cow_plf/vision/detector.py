"""
===============================================================================
COW PLF - VISION DETECTOR
===============================================================================

Production YOLO26m-seg + ByteTrack wrapper.

Responsibilities:
    - Load YOLO26m-seg
    - Run instance segmentation
    - Run ByteTrack
    - Return clean detection records

This module does NOT:
    - assign permanent Cow IDs
    - assign Global Cow IDs
    - calculate BCS
    - calculate welfare

Those belong to higher layers.
===============================================================================
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    tracker_id: int
    confidence: float
    box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    mask: Optional[np.ndarray]

    @property
    def x1(self):
        return self.box[0]

    @property
    def y1(self):
        return self.box[1]

    @property
    def x2(self):
        return self.box[2]

    @property
    def y2(self):
        return self.box[3]


class CowDetector:
    """
    YOLO26m-seg + ByteTrack production wrapper.
    """

    def __init__(
        self,
        model_path,
        confidence=0.20,
        iou=0.50,
        image_size=1280,
        device=0,
        tracker="bytetrack.yaml",
    ):

        self.model_path = str(
            Path(model_path)
        )

        self.confidence = float(
            confidence
        )

        self.iou = float(
            iou
        )

        self.image_size = int(
            image_size
        )

        self.device = device

        self.tracker = tracker

        print(
            f"[VISION] Loading model: "
            f"{self.model_path}"
        )

        self.model = YOLO(
            self.model_path
        )

        print(
            "[VISION] YOLO26m-seg loaded."
        )

    def predict(self, frame) -> List[Detection]:
        """
        Run YOLO26m-seg + ByteTrack on one frame.
        """

        results = self.model.track(

            frame,

            persist=True,

            tracker=self.tracker,

            conf=self.confidence,

            iou=self.iou,

            imgsz=self.image_size,

            device=self.device,

            verbose=False,

            retina_masks=True,
        )

        if not results:
            return []

        result = results[0]

        boxes = result.boxes

        masks = result.masks

        if boxes is None or len(boxes) == 0:
            return []

        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        confs = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        if boxes.id is not None:

            tracker_ids = (
                boxes.id
                .int()
                .detach()
                .cpu()
                .tolist()
            )

        else:

            tracker_ids = [
                -1
                for _ in xyxy
            ]

        mask_data = None

        if masks is not None:

            mask_data = (
                masks.data
                .detach()
                .cpu()
                .numpy()
            )

        height, width = frame.shape[:2]

        detections = []

        for index, (
            box,
            confidence,
            tracker_id
        ) in enumerate(
            zip(
                xyxy,
                confs,
                tracker_ids
            )
        ):

            tracker_id = int(
                tracker_id
            )

            # No ByteTrack ID -> do not pass into identity layer.
            if tracker_id < 0:
                continue

            x1, y1, x2, y2 = map(
                int,
                box
            )

            x1 = max(
                0,
                min(
                    width - 1,
                    x1
                )
            )

            y1 = max(
                0,
                min(
                    height - 1,
                    y1
                )
            )

            x2 = max(
                0,
                min(
                    width,
                    x2
                )
            )

            y2 = max(
                0,
                min(
                    height,
                    y2
                )
            )

            if x2 <= x1 or y2 <= y1:
                continue

            cx = int(
                (x1 + x2) / 2
            )

            cy = int(
                (y1 + y2) / 2
            )

            mask = None

            if (
                mask_data is not None
                and
                index < len(mask_data)
            ):

                raw_mask = mask_data[index]

                mask = cv2.resize(

                    raw_mask,

                    (
                        width,
                        height
                    ),

                    interpolation=cv2.INTER_NEAREST
                )

                mask = (
                    mask > 0.5
                )

            detections.append(
                Detection(

                    tracker_id=tracker_id,

                    confidence=float(
                        confidence
                    ),

                    box=(
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    center=(
                        cx,
                        cy
                    ),

                    mask=mask
                )
            )

        return detections