from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# BASIC MASK HELPERS
# ============================================================

def normalize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Return a clean uint8 binary mask."""
    if binary_mask is None:
        raise ValueError("binary_mask is None")

    mask = np.asarray(binary_mask)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, got shape {mask.shape}"
        )

    return (mask > 0).astype(np.uint8)


def get_mask_bbox(binary_mask: np.ndarray):
    """Return x1, y1, x2, y2."""
    mask = normalize_mask(binary_mask)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )


def get_mask_contour(binary_mask: np.ndarray):
    """Return the largest external contour."""
    mask = normalize_mask(binary_mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None

    return max(
        contours,
        key=cv2.contourArea,
    )


# ============================================================
# COLUMN PROFILE
# ============================================================

def _column_profiles(binary_mask: np.ndarray):
    """
    Extract top, bottom and depth profiles across the body.

    The profile is normalized to x = 0..1 so that pixel scale
    does not dominate the resulting features.
    """
    mask = normalize_mask(binary_mask)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())

    top = []
    bottom = []
    depth = []

    for x in range(x_min, x_max + 1):

        column = np.where(mask[:, x] > 0)[0]

        if len(column) == 0:
            continue

        t = float(column.min())
        b = float(column.max())
        d = b - t + 1.0

        top.append(t)
        bottom.append(b)
        depth.append(d)

    if len(depth) < 10:
        return None

    top = np.asarray(top, dtype=np.float32)
    bottom = np.asarray(bottom, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)

    # Robust body-depth scale.
    depth_scale = float(
        np.percentile(depth, 95)
    )

    if depth_scale <= 0:
        depth_scale = 1.0

    depth_norm = depth / depth_scale

    return {
        "top": top,
        "bottom": bottom,
        "depth": depth,
        "depth_norm": depth_norm,
    }


# ============================================================
# ORIENTATION-INVARIANT REGIONAL FEATURES
# ============================================================

def _regional_features(profile: dict) -> dict:
    """
    Calculate body-region depth descriptors.

    We deliberately make the two ends orientation-invariant:
    the animal can face left or right and the feature set remains
    stable.
    """
    depth_norm = profile["depth_norm"]

    n = len(depth_norm)

    if n < 10:
        return {
            "bcs_front_depth_norm": 0.0,
            "bcs_middle_depth_norm": 0.0,
            "bcs_rear_depth_norm": 0.0,
            "bcs_end_min_depth_norm": 0.0,
            "bcs_end_max_depth_norm": 0.0,
            "bcs_body_depth_cv": 0.0,
            "bcs_mid_to_end_ratio": 0.0,
            "bcs_depth_profile_slope": 0.0,
        }

    # Ignore the extreme silhouette ends because those are often
    # dominated by head/legs/tail geometry.
    lo = max(0, int(0.08 * n))
    hi = min(n, int(0.92 * n))

    d = depth_norm[lo:hi]

    if len(d) < 10:
        d = depth_norm

    m = len(d)

    q1_end = max(1, int(0.20 * m))
    q2 = int(0.40 * m)
    q3 = int(0.60 * m)
    q4 = max(q3 + 1, int(0.80 * m))

    end_a = float(np.median(d[:q1_end]))
    middle = float(np.median(d[q2:q3]))
    end_b = float(np.median(d[q4:]))

    end_min = min(end_a, end_b)
    end_max = max(end_a, end_b)

    d_mean = float(np.mean(d))
    d_std = float(np.std(d))

    depth_cv = (
        d_std / d_mean
        if d_mean > 1e-8
        else 0.0
    )

    end_reference = max(
        end_min,
        1e-6,
    )

    mid_to_end = middle / end_reference

    # Broad depth trend.
    first_half = float(
        np.mean(d[: max(1, m // 2)])
    )

    second_half = float(
        np.mean(d[m // 2:])
    )

    slope = (
        abs(first_half - second_half)
        / max(d_mean, 1e-6)
    )

    return {
        "bcs_front_depth_norm": end_a,
        "bcs_middle_depth_norm": middle,
        "bcs_rear_depth_norm": end_b,
        "bcs_end_min_depth_norm": end_min,
        "bcs_end_max_depth_norm": end_max,
        "bcs_body_depth_cv": depth_cv,
        "bcs_mid_to_end_ratio": mid_to_end,
        "bcs_depth_profile_slope": slope,
    }


# ============================================================
# CONTOUR FEATURES
# ============================================================

def _contour_features(profile: dict) -> dict:
    """
    Shape/contour descriptors normalized by body depth.
    """
    top = profile["top"]
    bottom = profile["bottom"]
    depth = profile["depth"]

    scale = max(
        float(np.percentile(depth, 95)),
        1.0,
    )

    top_norm = top / scale
    bottom_norm = bottom / scale

    # First differences.
    top_diff = np.diff(top_norm)
    bottom_diff = np.diff(bottom_norm)

    # Second differences.
    top_curve = np.diff(top_diff)
    bottom_curve = np.diff(bottom_diff)

    return {
        "bcs_top_contour_std_norm": float(
            np.std(top_norm)
        ),
        "bcs_bottom_contour_std_norm": float(
            np.std(bottom_norm)
        ),
        "bcs_top_contour_smoothness": float(
            np.std(top_curve)
            if len(top_curve) > 0
            else 0.0
        ),
        "bcs_bottom_contour_smoothness": float(
            np.std(bottom_curve)
            if len(bottom_curve) > 0
            else 0.0
        ),
        "bcs_depth_profile_std_norm": float(
            np.std(depth / scale)
        ),
    }


# ============================================================
# GENERAL SHAPE FEATURES
# ============================================================

def calculate_shape_features(binary_mask: np.ndarray) -> dict:

    mask = normalize_mask(binary_mask)

    area = float(
        np.count_nonzero(mask)
    )

    if area <= 0:

        return {
            "bcs_mask_area_px": 0.0,
            "bcs_bbox_width_px": 0.0,
            "bcs_bbox_height_px": 0.0,
            "bcs_body_length_px": 0.0,
            "bcs_body_depth_px": 0.0,
            "bcs_elongation_ratio": 0.0,
            "bcs_compactness": 0.0,
            "bcs_area_to_bbox_ratio": 0.0,
        }

    bbox = get_mask_bbox(mask)
    contour = get_mask_contour(mask)

    if bbox is None:

        return {
            "bcs_mask_area_px": 0.0,
            "bcs_bbox_width_px": 0.0,
            "bcs_bbox_height_px": 0.0,
            "bcs_body_length_px": 0.0,
            "bcs_body_depth_px": 0.0,
            "bcs_elongation_ratio": 0.0,
            "bcs_compactness": 0.0,
            "bcs_area_to_bbox_ratio": 0.0,
        }

    x1, y1, x2, y2 = bbox

    bbox_width = float(
        x2 - x1 + 1
    )

    bbox_height = float(
        y2 - y1 + 1
    )

    body_length = max(
        bbox_width,
        bbox_height,
    )

    body_depth = min(
        bbox_width,
        bbox_height,
    )

    elongation = (
        body_length / body_depth
        if body_depth > 0
        else 0.0
    )

    perimeter = (
        float(
            cv2.arcLength(
                contour,
                True,
            )
        )
        if contour is not None
        else 0.0
    )

    compactness = (
        (4.0 * np.pi * area)
        / (perimeter ** 2)
        if perimeter > 0
        else 0.0
    )

    bbox_area = (
        bbox_width
        * bbox_height
    )

    area_to_bbox = (
        area / bbox_area
        if bbox_area > 0
        else 0.0
    )

    return {
        "bcs_mask_area_px": area,
        "bcs_bbox_width_px": bbox_width,
        "bcs_bbox_height_px": bbox_height,
        "bcs_body_length_px": body_length,
        "bcs_body_depth_px": body_depth,
        "bcs_elongation_ratio": elongation,
        "bcs_compactness": compactness,
        "bcs_area_to_bbox_ratio": area_to_bbox,
    }


# ============================================================
# MAIN FEATURE EXTRACTOR
# ============================================================

def extract_bcs_features(
    binary_mask: np.ndarray,
) -> dict:
    """
    Extract image-derived BCS-supporting silhouette features.

    These features are NOT themselves a veterinary BCS.
    """
    mask = normalize_mask(
        binary_mask
    )

    features = {}

    # General geometry.
    features.update(
        calculate_shape_features(
            mask
        )
    )

    profile = _column_profiles(
        mask
    )

    if profile is None:

        features.update(
            {
                "bcs_top_contour_std_norm": 0.0,
                "bcs_bottom_contour_std_norm": 0.0,
                "bcs_top_contour_smoothness": 0.0,
                "bcs_bottom_contour_smoothness": 0.0,
                "bcs_depth_profile_std_norm": 0.0,
                "bcs_front_depth_norm": 0.0,
                "bcs_middle_depth_norm": 0.0,
                "bcs_rear_depth_norm": 0.0,
                "bcs_end_min_depth_norm": 0.0,
                "bcs_end_max_depth_norm": 0.0,
                "bcs_body_depth_cv": 0.0,
                "bcs_mid_to_end_ratio": 0.0,
                "bcs_depth_profile_slope": 0.0,
            }
        )

        return features

    features.update(
        _contour_features(
            profile
        )
    )

    features.update(
        _regional_features(
            profile
        )
    )

    return features