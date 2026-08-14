"""
===============================================================================
CONTACTLESS COW MORPHOMETRICS
===============================================================================

Purpose
-------
Convert an instance-segmentation mask into measurable body-shape features.

Current features
----------------
    - Mask area
    - Bounding-box width/height
    - Body length
    - Body width
    - Length/width ratio
    - Mask perimeter
    - Convex-hull area
    - Solidity
    - Extent
    - Circularity / compactness
    - Minimum-area-rectangle dimensions
    - Orientation angle

IMPORTANT
---------
These measurements are image-space measurements.

They are NOT centimetres or kilograms unless the camera is calibrated.

For true volumetric morphometrics later, we will add:
    - camera calibration
    - perspective correction
    - depth/stereo information
    - multi-view measurements

For BCS:
    These features become inputs to a trained/validated BCS model.
===============================================================================
"""

import cv2
import math
import numpy as np


class CowMorphometrics:
    """
    Extract body-shape measurements from one cow segmentation mask.
    """

    def __init__(
        self,
        min_mask_pixels=100
    ):

        self.min_mask_pixels = int(
            min_mask_pixels
        )

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
        Convert a mask into uint8 binary format.

        Accepted:
            bool array
            float probability mask
            uint8 mask
            0/1 mask
        """

        if mask is None:
            return None

        mask = np.asarray(mask)

        if mask.size == 0:
            return None

        # ---------------------------------------------------------
        # Probability mask
        # ---------------------------------------------------------

        if mask.dtype.kind == "f":

            binary = (
                mask > 0.5
            ).astype(
                np.uint8
            ) * 255

        else:

            binary = (
                mask > 0
            ).astype(
                np.uint8
            ) * 255

        # ---------------------------------------------------------
        # Resize if requested
        # ---------------------------------------------------------

        if (
            width is not None
            and
            height is not None
            and
            (
                binary.shape[1] != width
                or
                binary.shape[0] != height
            )
        ):

            binary = cv2.resize(
                binary,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        return binary

    # =========================================================================
    # LARGEST COMPONENT
    # =========================================================================

    @staticmethod
    def keep_largest_component(
        mask
    ):
        """
        Removes tiny disconnected components.

        Useful when a segmentation mask contains small isolated pixels.
        """

        if mask is None:
            return None

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        if num_labels <= 1:
            return mask

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        cleaned = np.zeros_like(
            mask
        )

        cleaned[
            labels == largest_label
        ] = 255

        return cleaned

    # =========================================================================
    # CONTOUR
    # =========================================================================

    @staticmethod
    def get_largest_contour(
        mask
    ):
        """
        Returns the largest external contour.
        """

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
    # MINIMUM AREA RECTANGLE
    # =========================================================================

    @staticmethod
    def rotated_rectangle_features(
        contour
    ):
        """
        Minimum-area bounding rectangle.

        Returns:
            rectangle width
            rectangle height
            long dimension
            short dimension
            angle
        """

        rect = cv2.minAreaRect(
            contour
        )

        (
            _center,
            (w, h),
            angle
        ) = rect

        if w <= 0 or h <= 0:

            return {
                "rotated_width_px": 0.0,
                "rotated_height_px": 0.0,
                "body_length_px": 0.0,
                "body_width_px": 0.0,
                "orientation_deg": 0.0,
            }

        if w >= h:

            length = float(w)

            width = float(h)

        else:

            length = float(h)

            width = float(w)

        return {
            "rotated_width_px": float(w),
            "rotated_height_px": float(h),
            "body_length_px": length,
            "body_width_px": width,
            "orientation_deg": float(angle),
        }

    # =========================================================================
    # PCA BODY AXES
    # =========================================================================

    @staticmethod
    def pca_body_axes(
        mask
    ):
        """
        Estimate body principal axes using PCA on mask pixels.

        This is often more stable than a normal bounding box because it follows
        the overall silhouette orientation.
        """

        ys, xs = np.where(
            mask > 0
        )

        if len(xs) < 10:

            return {
                "pca_length_px": 0.0,
                "pca_width_px": 0.0,
                "pca_angle_deg": 0.0
            }

        points = np.column_stack(
            (
                xs.astype(
                    np.float32
                ),
                ys.astype(
                    np.float32
                )
            )
        )

        mean = np.mean(
            points,
            axis=0
        )

        centered = (
            points - mean
        )

        covariance = np.cov(
            centered,
            rowvar=False
        )

        eigenvalues, eigenvectors = (
            np.linalg.eigh(
                covariance
            )
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

        # Standard deviations projected along axes.
        # Multiplication by 4 gives an approximate visible body span.
        length = (
            4.0
            *
            math.sqrt(
                max(
                    0.0,
                    float(
                        eigenvalues[0]
                    )
                )
            )
        )

        width = (
            4.0
            *
            math.sqrt(
                max(
                    0.0,
                    float(
                        eigenvalues[1]
                    )
                )
            )
        )

        vector = eigenvectors[
            :,
            0
        ]

        angle = math.degrees(
            math.atan2(
                float(vector[1]),
                float(vector[0])
            )
        )

        return {
            "pca_length_px": float(
                length
            ),
            "pca_width_px": float(
                width
            ),
            "pca_angle_deg": float(
                angle
            )
        }

    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================

    def extract(
        self,
        mask,
        bbox=None,
        frame_width=None,
        frame_height=None,
        clean_mask=True
    ):
        """
        Main morphometric extraction function.

        Parameters
        ----------
        mask:
            YOLO segmentation mask.

        bbox:
            Optional tuple:
                (x1, y1, x2, y2)

        Returns
        -------
        dict
            Morphometric measurements.
        """

        binary = self.prepare_mask(
            mask,
            width=frame_width,
            height=frame_height
        )

        if binary is None:

            return None

        # ---------------------------------------------------------
        # Clean segmentation
        # ---------------------------------------------------------

        if clean_mask:

            binary = (
                self.keep_largest_component(
                    binary
                )
            )

        mask_pixels = int(
            cv2.countNonZero(
                binary
            )
        )

        if mask_pixels < (
            self.min_mask_pixels
        ):

            return None

        # ---------------------------------------------------------
        # Contour
        # ---------------------------------------------------------

        contour = (
            self.get_largest_contour(
                binary
            )
        )

        if contour is None:

            return None

        # ---------------------------------------------------------
        # Basic area / perimeter
        # ---------------------------------------------------------

        area = float(
            cv2.contourArea(
                contour
            )
        )

        perimeter = float(
            cv2.arcLength(
                contour,
                True
            )
        )

        if area <= 0:

            return None

        # ---------------------------------------------------------
        # Standard bounding box
        # ---------------------------------------------------------

        bx, by, bw, bh = (
            cv2.boundingRect(
                contour
            )
        )

        bbox_area = float(
            bw * bh
        )

        # ---------------------------------------------------------
        # Rotated rectangle
        # ---------------------------------------------------------

        rotated = (
            self.rotated_rectangle_features(
                contour
            )
        )

        body_length = (
            rotated[
                "body_length_px"
            ]
        )

        body_width = (
            rotated[
                "body_width_px"
            ]
        )

        # ---------------------------------------------------------
        # Length / width ratio
        # ---------------------------------------------------------

        if body_width > 1e-6:

            length_width_ratio = (
                body_length
                /
                body_width
            )

        else:

            length_width_ratio = 0.0

        # ---------------------------------------------------------
        # Convex hull
        # ---------------------------------------------------------

        hull = cv2.convexHull(
            contour
        )

        hull_area = float(
            cv2.contourArea(
                hull
            )
        )

        if hull_area > 1e-6:

            solidity = (
                area
                /
                hull_area
            )

        else:

            solidity = 0.0

        # ---------------------------------------------------------
        # Extent
        # ---------------------------------------------------------

        if bbox_area > 1e-6:

            extent = (
                area
                /
                bbox_area
            )

        else:

            extent = 0.0

        # ---------------------------------------------------------
        # Circularity / compactness
        #
        # 4*pi*A / P^2
        #
        # Higher = more compact
        # Lower = elongated / irregular
        # ---------------------------------------------------------

        if perimeter > 1e-6:

            circularity = (
                4.0
                *
                math.pi
                *
                area
                /
                (
                    perimeter
                    *
                    perimeter
                )
            )

        else:

            circularity = 0.0

        # ---------------------------------------------------------
        # Convexity
        # ---------------------------------------------------------

        hull_perimeter = float(
            cv2.arcLength(
                hull,
                True
            )
        )

        if hull_perimeter > 1e-6:

            convexity = (
                hull_perimeter
                /
                perimeter
            )

        else:

            convexity = 0.0

        # ---------------------------------------------------------
        # PCA
        # ---------------------------------------------------------

        pca = (
            self.pca_body_axes(
                binary
            )
        )

        # ---------------------------------------------------------
        # BBox supplied by detector
        # ---------------------------------------------------------

        detector_bbox_width = 0.0

        detector_bbox_height = 0.0

        if bbox is not None:

            x1, y1, x2, y2 = map(
                float,
                bbox
            )

            detector_bbox_width = max(
                0.0,
                x2 - x1
            )

            detector_bbox_height = max(
                0.0,
                y2 - y1
            )

        else:

            detector_bbox_width = float(
                bw
            )

            detector_bbox_height = float(
                bh
            )

        if (
            detector_bbox_width > 0
            and
            detector_bbox_height > 0
        ):

            detector_bbox_area = (
                detector_bbox_width
                *
                detector_bbox_height
            )

            mask_to_detection_ratio = (
                area
                /
                detector_bbox_area
            )

        else:

            detector_bbox_area = 0.0

            mask_to_detection_ratio = 0.0

        # ---------------------------------------------------------
        # Center
        # ---------------------------------------------------------

        moments = cv2.moments(
            contour
        )

        if (
            moments["m00"]
            != 0
        ):

            centroid_x = (
                moments["m10"]
                /
                moments["m00"]
            )

            centroid_y = (
                moments["m01"]
                /
                moments["m00"]
            )

        else:

            centroid_x = float(
                bx + bw / 2
            )

            centroid_y = float(
                by + bh / 2
            )

        # ---------------------------------------------------------
        # Normalized image-space features
        # ---------------------------------------------------------

        if (
            frame_width
            and
            frame_width > 0
        ):

            normalized_area = (
                area
                /
                (
                    frame_width
                    *
                    frame_width
                )
            )

        else:

            normalized_area = 0.0

        # ---------------------------------------------------------
        # Result
        # ---------------------------------------------------------

        features = {

            # -----------------------------------------------------
            # BASIC SEGMENTATION
            # -----------------------------------------------------

            "mask_area_px":
                float(mask_pixels),

            "contour_area_px":
                area,

            "perimeter_px":
                perimeter,

            # -----------------------------------------------------
            # BOUNDING BOX
            # -----------------------------------------------------

            "bbox_x":
                int(bx),

            "bbox_y":
                int(by),

            "bbox_width_px":
                float(bw),

            "bbox_height_px":
                float(bh),

            "bbox_area_px":
                bbox_area,

            # -----------------------------------------------------
            # BODY SHAPE
            # -----------------------------------------------------

            "body_length_px":
                body_length,

            "body_width_px":
                body_width,

            "length_width_ratio":
                float(
                    length_width_ratio
                ),

            "orientation_deg":
                rotated[
                    "orientation_deg"
                ],

            # -----------------------------------------------------
            # PCA
            # -----------------------------------------------------

            "pca_length_px":
                pca[
                    "pca_length_px"
                ],

            "pca_width_px":
                pca[
                    "pca_width_px"
                ],

            "pca_angle_deg":
                pca[
                    "pca_angle_deg"
                ],

            "pca_length_width_ratio":
                (
                    pca[
                        "pca_length_px"
                    ]
                    /
                    max(
                        1e-6,
                        pca[
                            "pca_width_px"
                        ]
                    )
                ),

            # -----------------------------------------------------
            # SHAPE QUALITY
            # -----------------------------------------------------

            "convex_hull_area_px":
                hull_area,

            "solidity":
                float(
                    solidity
                ),

            "extent":
                float(
                    extent
                ),

            "circularity":
                float(
                    circularity
                ),

            "convexity":
                float(
                    convexity
                ),

            # -----------------------------------------------------
            # DETECTOR RELATION
            # -----------------------------------------------------

            "detector_bbox_width_px":
                float(
                    detector_bbox_width
                ),

            "detector_bbox_height_px":
                float(
                    detector_bbox_height
                ),

            "detector_bbox_area_px":
                float(
                    detector_bbox_area
                ),

            "mask_to_bbox_ratio":
                float(
                    mask_to_detection_ratio
                ),

            # -----------------------------------------------------
            # CENTROID
            # -----------------------------------------------------

            "centroid_x":
                float(
                    centroid_x
                ),

            "centroid_y":
                float(
                    centroid_y
                ),

            # -----------------------------------------------------
            # NORMALIZED AREA
            # -----------------------------------------------------

            "normalized_area":
                float(
                    normalized_area
                )
        }

        return features

    # =========================================================================
    # BCS FEATURE VECTOR
    # =========================================================================

    @staticmethod
    def bcs_feature_vector(
        features
    ):
        """
        Prepare a feature vector for a future trained BCS model.

        IMPORTANT:
            This is NOT a BCS score.
            It is the feature vector that can feed a trained model.
        """

        if features is None:

            return None

        return np.array(

            [

                features.get(
                    "body_length_px",
                    0.0
                ),

                features.get(
                    "body_width_px",
                    0.0
                ),

                features.get(
                    "length_width_ratio",
                    0.0
                ),

                features.get(
                    "mask_area_px",
                    0.0
                ),

                features.get(
                    "solidity",
                    0.0
                ),

                features.get(
                    "extent",
                    0.0
                ),

                features.get(
                    "circularity",
                    0.0
                ),

                features.get(
                    "mask_to_bbox_ratio",
                    0.0
                ),

                features.get(
                    "pca_length_px",
                    0.0
                ),

                features.get(
                    "pca_width_px",
                    0.0
                )
            ],

            dtype=np.float32
        )


# =============================================================================
# TEST WITH SYNTHETIC COW-LIKE SILHOUETTE
# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("CONTACTLESS COW MORPHOMETRICS TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Create a simple synthetic silhouette.
    #
    # This only validates the mathematics/code.
    # It is NOT a cow BCS example.
    # -------------------------------------------------------------------------

    canvas = np.zeros(
        (600, 1000),
        dtype=np.uint8
    )

    cv2.ellipse(
        canvas,
        (500, 300),
        (260, 110),
        0,
        0,
        360,
        255,
        -1
    )

    # -------------------------------------------------------------------------
    # Extract
    # -------------------------------------------------------------------------

    extractor = CowMorphometrics()

    features = extractor.extract(
        canvas,
        frame_width=1000,
        frame_height=600
    )

    if features is None:

        print(
            "[ERROR] Morphometric extraction failed."
        )

    else:

        print()
        print("FEATURES")
        print("-" * 80)

        important = [

            "mask_area_px",

            "body_length_px",

            "body_width_px",

            "length_width_ratio",

            "perimeter_px",

            "convex_hull_area_px",

            "solidity",

            "extent",

            "circularity",

            "convexity",

            "pca_length_px",

            "pca_width_px",

            "pca_length_width_ratio",

            "orientation_deg",

            "centroid_x",

            "centroid_y",
        ]

        for name in important:

            print(
                f"{name:30s}: "
                f"{features[name]:.4f}"
            )

        vector = (
            extractor.bcs_feature_vector(
                features
            )
        )

        print()
        print("BCS MODEL FEATURE VECTOR")
        print("-" * 80)

        print(vector)

        print()
        print("=" * 80)
        print("MORPHOMETRICS TEST COMPLETE")
        print("=" * 80)