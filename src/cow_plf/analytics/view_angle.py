"""
===============================================================================
COW PLF - ROBUST COW VIEW ANGLE ANALYZER
===============================================================================

Purpose
-------
Determine whether a detected cow is suitable for contactless morphometric
measurement.

Main idea
---------
For a true side view, the cow's main body axis is approximately horizontal.

The previous implementation relied too heavily on silhouette elongation.
That caused real cows with visible legs/head/neck to be incorrectly rejected.

This version:

    1. Prepares the YOLO segmentation mask
    2. Removes small components
    3. Finds the largest cow contour
    4. Calculates PCA body axis
    5. Measures angle relative to horizontal
    6. Uses elongation only as a supporting feature
    7. Classifies the view

Classes
-------
    SIDE_VIEW_GOOD
    SIDE_VIEW_ACCEPTABLE
    NON_SIDE_VIEW

Measurement
-----------
    SIDE_VIEW_GOOD       -> True
    SIDE_VIEW_ACCEPTABLE -> True
    NON_SIDE_VIEW        -> False

Important
---------
This module determines whether the image geometry is suitable for
measurement. It does NOT calculate real-world length or weight.

Pixel measurements still require camera calibration for centimetres/metres.
===============================================================================
"""

import math
import cv2
import numpy as np


class CowViewAngleAnalyzer:

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    GOOD_ANGLE_DEG = 20.0
    ACCEPTABLE_ANGLE_DEG = 40.0

    # Much lower than the old threshold because real cow silhouettes include
    # legs, neck and head.
    GOOD_ELONGATION = 1.20
    ACCEPTABLE_ELONGATION = 1.10

    MIN_MASK_PIXELS = 500

    # =========================================================================
    # MASK PREPARATION
    # =========================================================================

    @staticmethod
    def prepare_mask(
        mask,
        width=None,
        height=None
    ):
        """
        Convert a YOLO mask into a binary uint8 mask.

        Compatible with the image-testing pipeline:

            prepare_mask(mask, width=..., height=...)

        Accepted:
            bool
            float probability mask
            uint8 mask
            0/1 mask
        """

        if mask is None:
            return None

        try:
            mask = np.asarray(mask)
        except Exception:
            return None

        if mask.size == 0:
            return None

        # ---------------------------------------------------------------------
        # Probability mask
        # ---------------------------------------------------------------------

        if mask.dtype.kind == "f":
            binary = (
                mask > 0.5
            ).astype(np.uint8) * 255

        else:
            binary = (
                mask > 0
            ).astype(np.uint8) * 255

        # ---------------------------------------------------------------------
        # Ensure 2D
        # ---------------------------------------------------------------------

        if binary.ndim > 2:
            binary = np.squeeze(binary)

        if binary.ndim != 2:
            return None

        # ---------------------------------------------------------------------
        # Resize if requested
        # ---------------------------------------------------------------------

        if (
            width is not None
            and height is not None
            and (
                binary.shape[1] != int(width)
                or
                binary.shape[0] != int(height)
            )
        ):

            binary = cv2.resize(
                binary,
                (
                    int(width),
                    int(height)
                ),
                interpolation=cv2.INTER_NEAREST
            )

        return binary

    # =========================================================================
    # LARGEST COMPONENT
    # =========================================================================

    @staticmethod
    def keep_largest_component(mask):

        if mask is None:
            return None

        if cv2.countNonZero(mask) == 0:
            return None

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        if num_labels <= 1:
            return mask

        largest_label = (
            1
            +
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )

        cleaned = np.zeros_like(mask)

        cleaned[
            labels == largest_label
        ] = 255

        return cleaned

    # =========================================================================
    # LARGEST CONTOUR
    # =========================================================================

    @staticmethod
    def get_largest_contour(mask):

        if mask is None:
            return None

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return None

        return max(
            contours,
            key=cv2.contourArea
        )

    # =========================================================================
    # PCA BODY AXIS
    # =========================================================================

    @staticmethod
    def calculate_pca(mask):

        ys, xs = np.where(
            mask > 0
        )

        if len(xs) < 20:
            return None

        points = np.column_stack(
            (
                xs.astype(np.float32),
                ys.astype(np.float32)
            )
        )

        mean = np.mean(
            points,
            axis=0
        )

        centered = points - mean

        covariance = np.cov(
            centered,
            rowvar=False
        )

        eigenvalues, eigenvectors = (
            np.linalg.eigh(covariance)
        )

        order = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[
            order
        ]

        eigenvectors = eigenvectors[
            :,
            order
        ]

        # Principal body axis.
        vector = eigenvectors[:, 0]

        angle = math.degrees(
            math.atan2(
                float(vector[1]),
                float(vector[0])
            )
        )

        # Normalize to [-180, 180]
        while angle > 180:
            angle -= 360

        while angle < -180:
            angle += 360

        # The PCA axis has no direction. Therefore:
        # 180 degrees and 0 degrees both represent horizontal.
        horizontal_angle = min(
            abs(angle),
            abs(180.0 - abs(angle))
        )

        # PCA standard-deviation based shape ratio.
        major = math.sqrt(
            max(
                0.0,
                float(eigenvalues[0])
            )
        )

        minor = math.sqrt(
            max(
                0.0,
                float(eigenvalues[1])
            )
        )

        if minor > 1e-8:
            elongation = (
                major / minor
            )
        else:
            elongation = 0.0

        return {
            "orientation_deg": float(angle),

            "horizontal_angle_deg":
                float(horizontal_angle),

            "elongation_ratio":
                float(elongation)
        }

    # =========================================================================
    # MINIMUM AREA RECTANGLE
    # =========================================================================

    @staticmethod
    def calculate_rectangle(contour):

        if contour is None:
            return None

        rect = cv2.minAreaRect(
            contour
        )

        (
            _center,
            (w, h),
            angle
        ) = rect

        w = float(w)
        h = float(h)

        if w <= 0 or h <= 0:
            return None

        long_side = max(
            w,
            h
        )

        short_side = min(
            w,
            h
        )

        ratio = (
            long_side /
            max(
                short_side,
                1e-8
            )
        )

        return {
            "rotated_width_px":
                w,

            "rotated_height_px":
                h,

            "rectangle_ratio":
                float(ratio),

            "rectangle_angle_deg":
                float(angle)
        }

    # =========================================================================
    # BODY CENTER
    # =========================================================================

    @staticmethod
    def calculate_center(contour):

        moments = cv2.moments(
            contour
        )

        if abs(
            moments["m00"]
        ) > 1e-8:

            cx = (
                moments["m10"]
                /
                moments["m00"]
            )

            cy = (
                moments["m01"]
                /
                moments["m00"]
            )

        else:

            x, y, w, h = (
                cv2.boundingRect(
                    contour
                )
            )

            cx = x + w / 2.0
            cy = y + h / 2.0

        return (
            float(cx),
            float(cy)
        )

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    def analyze(
        self,
        mask,
        frame_width=None,
        frame_height=None
    ):
        """
        Analyze the view angle of one cow.

        Returns a dictionary containing:

            view_class
            confidence
            measurement_valid
            orientation_deg
            horizontal_angle_deg
            elongation_ratio
            rectangle_ratio
            mask_area_px
            center_x
            center_y
        """

        binary = self.prepare_mask(
            mask,
            width=frame_width,
            height=frame_height
        )

        if binary is None:

            return {
                "view_class":
                    "NON_SIDE_VIEW",

                "confidence":
                    0.0,

                "measurement_valid":
                    False,

                "orientation_deg":
                    0.0,

                "horizontal_angle_deg":
                    90.0,

                "elongation_ratio":
                    0.0,

                "rectangle_ratio":
                    0.0,

                "mask_area_px":
                    0,

                "center_x":
                    0.0,

                "center_y":
                    0.0
            }

        # ---------------------------------------------------------------------
        # Clean mask
        # ---------------------------------------------------------------------

        binary = (
            self.keep_largest_component(
                binary
            )
        )

        if binary is None:

            return {
                "view_class":
                    "NON_SIDE_VIEW",

                "confidence":
                    0.0,

                "measurement_valid":
                    False,

                "orientation_deg":
                    0.0,

                "horizontal_angle_deg":
                    90.0,

                "elongation_ratio":
                    0.0,

                "rectangle_ratio":
                    0.0,

                "mask_area_px":
                    0,

                "center_x":
                    0.0,

                "center_y":
                    0.0
            }

        mask_area = int(
            cv2.countNonZero(
                binary
            )
        )

        if mask_area < self.MIN_MASK_PIXELS:

            return {
                "view_class":
                    "NON_SIDE_VIEW",

                "confidence":
                    0.0,

                "measurement_valid":
                    False,

                "orientation_deg":
                    0.0,

                "horizontal_angle_deg":
                    90.0,

                "elongation_ratio":
                    0.0,

                "rectangle_ratio":
                    0.0,

                "mask_area_px":
                    mask_area,

                "center_x":
                    0.0,

                "center_y":
                    0.0
            }

        # ---------------------------------------------------------------------
        # Contour
        # ---------------------------------------------------------------------

        contour = (
            self.get_largest_contour(
                binary
            )
        )

        if contour is None:

            return {
                "view_class":
                    "NON_SIDE_VIEW",

                "confidence":
                    0.0,

                "measurement_valid":
                    False,

                "orientation_deg":
                    0.0,

                "horizontal_angle_deg":
                    90.0,

                "elongation_ratio":
                    0.0,

                "rectangle_ratio":
                    0.0,

                "mask_area_px":
                    mask_area,

                "center_x":
                    0.0,

                "center_y":
                    0.0
            }

        # ---------------------------------------------------------------------
        # PCA
        # ---------------------------------------------------------------------

        pca = self.calculate_pca(
            binary
        )

        if pca is None:

            return {
                "view_class":
                    "NON_SIDE_VIEW",

                "confidence":
                    0.0,

                "measurement_valid":
                    False,

                "orientation_deg":
                    0.0,

                "horizontal_angle_deg":
                    90.0,

                "elongation_ratio":
                    0.0,

                "rectangle_ratio":
                    0.0,

                "mask_area_px":
                    mask_area,

                "center_x":
                    0.0,

                "center_y":
                    0.0
            }

        # ---------------------------------------------------------------------
        # Rotated rectangle
        # ---------------------------------------------------------------------

        rectangle = (
            self.calculate_rectangle(
                contour
            )
        )

        if rectangle is None:
            rectangle_ratio = 0.0
        else:
            rectangle_ratio = (
                rectangle[
                    "rectangle_ratio"
                ]
            )

        # ---------------------------------------------------------------------
        # Center
        # ---------------------------------------------------------------------

        center_x, center_y = (
            self.calculate_center(
                contour
            )
        )

        angle = (
            pca[
                "horizontal_angle_deg"
            ]
        )

        elongation = (
            pca[
                "elongation_ratio"
            ]
        )

        # ---------------------------------------------------------------------
        # VIEW CLASSIFICATION
        # ---------------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # Angle is the PRIMARY signal.
        #
        # This fixes the previous problem where a genuine side-view cow
        # was rejected because its legs reduced the elongation ratio.
        #
        # ---------------------------------------------------------------------

        if (
            angle <= self.GOOD_ANGLE_DEG
            and
            elongation >= self.GOOD_ELONGATION
        ):

            view_class = (
                "SIDE_VIEW_GOOD"
            )

            measurement_valid = True

        elif (
            angle <= self.ACCEPTABLE_ANGLE_DEG
            and
            elongation >= self.ACCEPTABLE_ELONGATION
        ):

            view_class = (
                "SIDE_VIEW_ACCEPTABLE"
            )

            measurement_valid = True

        else:

            view_class = (
                "NON_SIDE_VIEW"
            )

            measurement_valid = False

        # ---------------------------------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------------------------------

        # Angle confidence:
        # 0 degrees = strongest lateral view.
        angle_confidence = max(
            0.0,
            1.0 -
            (
                angle /
                90.0
            )
        )

        # Shape confidence.
        shape_confidence = min(
            1.0,
            max(
                0.0,
                (
                    elongation -
                    1.0
                ) /
                1.5
            )
        )

        confidence = (
            0.75 *
            angle_confidence
            +
            0.25 *
            shape_confidence
        )

        # Valid side view gets confidence boost according to classification.
        if view_class == "SIDE_VIEW_GOOD":

            confidence = max(
                confidence,
                0.75
            )

        elif (
            view_class ==
            "SIDE_VIEW_ACCEPTABLE"
        ):

            confidence = max(
                confidence,
                0.55
            )

        confidence = min(
            1.0,
            confidence
        )

        # ---------------------------------------------------------------------
        # RESULT
        # ---------------------------------------------------------------------

        return {

            "view_class":
                view_class,

            "confidence":
                float(confidence),

            "measurement_valid":
                bool(measurement_valid),

            "orientation_deg":
                float(
                    pca[
                        "orientation_deg"
                    ]
                ),

            "horizontal_angle_deg":
                float(angle),

            "elongation_ratio":
                float(elongation),

            "rectangle_ratio":
                float(
                    rectangle_ratio
                ),

            "mask_area_px":
                int(mask_area),

            "center_x":
                float(center_x),

            "center_y":
                float(center_y)
        }

    # =========================================================================
    # ALIAS
    # =========================================================================

    def analyze_mask(
        self,
        mask,
        frame_width=None,
        frame_height=None
    ):
        """
        Compatibility alias.
        """

        return self.analyze(
            mask=mask,
            frame_width=frame_width,
            frame_height=frame_height
        )


# =============================================================================
# SYNTHETIC TEST
# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("COW PLF - VIEW ANGLE ANALYZER TEST")
    print("=" * 80)

    analyzer = (
        CowViewAngleAnalyzer()
    )

    # -------------------------------------------------------------------------
    # Side-like cow silhouette
    # -------------------------------------------------------------------------

    side = np.zeros(
        (600, 1000),
        dtype=np.uint8
    )

    cv2.ellipse(
        side,
        (500, 300),
        (260, 100),
        0,
        0,
        360,
        255,
        -1
    )

    result = analyzer.analyze(
        side,
        frame_width=1000,
        frame_height=600
    )

    print()
    print("SIDE-LIKE SILHOUETTE")
    print("-" * 80)

    for key, value in result.items():
        print(
            f"{key:30s}: {value}"
        )

    # -------------------------------------------------------------------------
    # Front-like cow silhouette
    # -------------------------------------------------------------------------

    front = np.zeros(
        (600, 1000),
        dtype=np.uint8
    )

    cv2.ellipse(
        front,
        (500, 300),
        (110, 220),
        0,
        0,
        360,
        255,
        -1
    )

    result = analyzer.analyze(
        front,
        frame_width=1000,
        frame_height=600
    )

    print()
    print("FRONT-LIKE SILHOUETTE")
    print("-" * 80)

    for key, value in result.items():
        print(
            f"{key:30s}: {value}"
        )

    print()
    print("=" * 80)
    print("VIEW ANGLE TEST COMPLETE")
    print("=" * 80)