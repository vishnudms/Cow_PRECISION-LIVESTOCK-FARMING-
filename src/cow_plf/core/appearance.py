"""
============================================================
COW APPEARANCE FEATURE EXTRACTOR
============================================================

Purpose:

Convert a detected cow image into an appearance vector.

This module is intentionally kept separate from YOLO and
IdentityManager.

Pipeline:

    Cow crop
       ↓
    AppearanceExtractor
       ↓
    Feature vector
       ↓
    IdentityManager
============================================================
"""

import cv2
import numpy as np


class AppearanceExtractor:

    def __init__(
        self,
        image_size=128
    ):

        self.image_size = image_size


    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(
        self,
        crop
    ):

        if crop is None:

            return None

        if crop.size == 0:

            return None

        # Resize

        image = cv2.resize(
            crop,
            (
                self.image_size,
                self.image_size
            )
        )

        # Convert BGR → RGB

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return image


    # ========================================================
    # COLOR HISTOGRAM
    # ========================================================

    def color_histogram(
        self,
        image
    ):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2HSV
        )

        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [16, 16],
            [0, 180, 0, 256]
        )

        hist = cv2.normalize(
            hist,
            hist
        )

        return hist.flatten()


    # ========================================================
    # EDGE FEATURE
    # ========================================================

    def edge_feature(
        self,
        image
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        edges = cv2.resize(
            edges,
            (
                32,
                32
            )
        )

        feature = (
            edges
            /
            255.0
        )

        return feature.flatten()


    # ========================================================
    # COLOR MEAN
    # ========================================================

    def color_statistics(
        self,
        image
    ):

        mean = np.mean(
            image,
            axis=(0, 1)
        )

        std = np.std(
            image,
            axis=(0, 1)
        )

        return np.concatenate(
            [
                mean,
                std
            ]
        )


    # ========================================================
    # EXTRACT FEATURE
    # ========================================================

    def extract(
        self,
        crop
    ):

        image = self.preprocess(
            crop
        )

        if image is None:

            return None

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        color = self.color_histogram(
            image
        )

        edges = self.edge_feature(
            image
        )

        statistics = self.color_statistics(
            image
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        feature = np.concatenate(
            [
                color,
                edges,
                statistics
            ]
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        norm = np.linalg.norm(
            feature
        )

        if norm > 0:

            feature = (
                feature
                /
                norm
            )

        return feature


    # ========================================================
    # SIMILARITY
    # ========================================================

    @staticmethod
    def similarity(
        feature1,
        feature2
    ):

        if feature1 is None:

            return 0.0

        if feature2 is None:

            return 0.0

        feature1 = np.asarray(
            feature1,
            dtype=np.float32
        )

        feature2 = np.asarray(
            feature2,
            dtype=np.float32
        )

        if (
            feature1.shape
            !=
            feature2.shape
        ):

            return 0.0

        norm1 = np.linalg.norm(
            feature1
        )

        norm2 = np.linalg.norm(
            feature2
        )

        if (
            norm1 == 0
            or
            norm2 == 0
        ):

            return 0.0

        return float(

            np.dot(
                feature1,
                feature2
            )
            /
            (
                norm1
                *
                norm2
            )

        )