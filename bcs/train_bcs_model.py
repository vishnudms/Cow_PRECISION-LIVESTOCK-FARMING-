"""
COW PLF - BCS MODEL TRAINING PIPELINE

Trains and evaluates BCS regression models using:
    - BCS vision features
    - Reference BCS labels

IMPORTANT:
    This script will NOT train if reference BCS labels are missing.

Input:
    D:\cow\output\bcs_dataset\bcs_features.csv
    D:\cow\bcs\labels\bcs_labeling_dataset.csv

Output:
    D:\cow\output\bcs_model\
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"D:\cow")

FEATURES_PATH = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
    / "bcs_features.csv"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "bcs"
    / "labels"
    / "bcs_labeling_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "bcs_model"
)

MODEL_PATH = OUTPUT_DIR / "best_bcs_model.joblib"
METRICS_PATH = OUTPUT_DIR / "bcs_model_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "bcs_cv_predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_COLUMN = "bcs_score"

ID_COLUMNS = [
    "cow_id",
    "image_name",
]

# Do NOT use actual_weight_kg as a predictor.
#
# Weight is useful for other livestock models, but using it here could make
# the BCS model artificially dependent on weight and hide the actual visual
# performance of the BCS system.
EXCLUDED_COLUMNS = [
    "actual_weight_kg",
]

RANDOM_STATE = 42

# Five-fold CV is appropriate for this small dataset.
N_SPLITS = 5


# =============================================================================
# HEADER
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    print_header("COW PLF - BCS MODEL TRAINING")

    print("[LOAD] BCS feature dataset")
    print(f"       {FEATURES_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"BCS feature dataset not found:\n{FEATURES_PATH}"
        )

    print("[LOAD] BCS reference labels")
    print(f"       {LABELS_PATH}")

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"BCS label dataset not found:\n{LABELS_PATH}"
        )

    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)

    print(f"[OK] Feature rows : {len(features)}")
    print(f"[OK] Label rows   : {len(labels)}")

    return features, labels


# =============================================================================
# VALIDATE LABELS
# =============================================================================

def validate_labels(labels):
    print_header("REFERENCE BCS LABEL VALIDATION")

    if TARGET_COLUMN not in labels.columns:
        raise ValueError(
            f"Missing required target column: {TARGET_COLUMN}"
        )

    scores = pd.to_numeric(
        labels[TARGET_COLUMN],
        errors="coerce",
    )

    labeled_mask = scores.notna()

    labeled_count = int(labeled_mask.sum())
    unlabeled_count = int((~labeled_mask).sum())

    print(f"Labeled   : {labeled_count}")
    print(f"Unlabeled : {unlabeled_count}")

    # -------------------------------------------------------------------------
    # HARD SAFETY CHECK
    # -------------------------------------------------------------------------

    if labeled_count == 0:
        raise RuntimeError(
            "\nNO REFERENCE BCS LABELS FOUND.\n\n"
            "The model cannot be trained yet.\n"
            "Fill the bcs_score column in:\n"
            f"{LABELS_PATH}\n"
        )

    if unlabeled_count > 0:
        raise RuntimeError(
            f"\nINCOMPLETE BCS LABELS.\n\n"
            f"Labeled   : {labeled_count}\n"
            f"Unlabeled : {unlabeled_count}\n\n"
            "For this first controlled training run, all cows must have "
            "reference BCS labels."
        )

    # -------------------------------------------------------------------------
    # RANGE
    # -------------------------------------------------------------------------

    if scores.min() < 1.0 or scores.max() > 5.0:
        raise ValueError(
            "BCS scores must be between 1.00 and 5.00."
        )

    # -------------------------------------------------------------------------
    # QUARTER-SCORE CHECK
    # -------------------------------------------------------------------------

    scaled = scores * 4

    if not np.allclose(
        scaled,
        np.round(scaled),
        atol=1e-6,
    ):
        raise ValueError(
            "BCS scores must use 0.25 increments."
        )

    print()
    print("[PASS] All 71 cows have reference BCS labels.")
    print(
        f"BCS range : {scores.min():.2f} - {scores.max():.2f}"
    )

    print()
    print("BCS distribution:")
    print(scores.value_counts().sort_index())

    return scores


# =============================================================================
# MERGE
# =============================================================================

def merge_dataset(features, labels):
    print_header("MERGING FEATURES + REFERENCE BCS")

    if "cow_id" not in features.columns:
        raise ValueError("Feature dataset missing cow_id.")

    if "cow_id" not in labels.columns:
        raise ValueError("Label dataset missing cow_id.")

    # Only bring target and metadata from labels.
    label_columns = [
        "cow_id",
        "bcs_score",
        "bcs_source",
        "assessor_id",
        "assessment_notes",
    ]

    label_columns = [
        c for c in label_columns
        if c in labels.columns
    ]

    dataset = features.merge(
        labels[label_columns],
        on="cow_id",
        how="inner",
        validate="one_to_one",
    )

    print(f"Feature rows : {len(features)}")
    print(f"Label rows   : {len(labels)}")
    print(f"Merged rows  : {len(dataset)}")

    if len(dataset) != len(features):
        raise RuntimeError(
            "Merge lost rows. Check cow_id consistency."
        )

    if dataset[TARGET_COLUMN].isna().any():
        raise RuntimeError(
            "Merged dataset contains missing BCS labels."
        )

    print("[PASS] One-to-one feature/label merge.")

    return dataset


# =============================================================================
# SELECT FEATURES
# =============================================================================

def select_features(dataset):
    print_header("FEATURE SELECTION")

    excluded = set(
        ID_COLUMNS
        + EXCLUDED_COLUMNS
        + [
            TARGET_COLUMN,
            "bcs_source",
            "assessor_id",
            "assessment_notes",
            "label_status",
        ]
    )

    feature_columns = []

    for column in dataset.columns:

        if column in excluded:
            continue

        if pd.api.types.is_numeric_dtype(dataset[column]):
            feature_columns.append(column)

    if not feature_columns:
        raise RuntimeError(
            "No numeric BCS features available."
        )

    print(f"Numeric features selected : {len(feature_columns)}")

    for column in feature_columns:
        print(f"  - {column}")

    return feature_columns


# =============================================================================
# MODELS
# =============================================================================

def build_models():
    """
    Multiple models are evaluated.

    Tree models are particularly useful for this type of nonlinear
    morphometric feature set.
    """

    models = {

        "Ridge": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(alpha=10.0),
                ),
            ]
        ),

        "RandomForest": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "ExtraTrees": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_depth=6,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "GradientBoosting": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=150,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=3,
                        loss="huber",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "HistGradientBoosting": Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=150,
                        learning_rate=0.04,
                        max_leaf_nodes=7,
                        min_samples_leaf=5,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_models(X, y, cow_ids):
    print_header("CROSS-VALIDATED MODEL EVALUATION")

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    results = []
    prediction_tables = {}

    for model_name, model in models.items():

        print()
        print("-" * 80)
        print(f"[MODEL] {model_name}")

        predictions = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            n_jobs=None,
        )

        predictions = np.clip(
            predictions,
            1.0,
            5.0,
        )

        mae = mean_absolute_error(
            y,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y,
                predictions,
            )
        )

        r2 = r2_score(
            y,
            predictions,
        )

        results.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        prediction_tables[model_name] = predictions

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R²   : {r2:.4f}")

    results_df = pd.DataFrame(results)

    # Primary selection criterion:
    # lowest cross-validated MAE.
    results_df = results_df.sort_values(
        by=["MAE", "RMSE"],
        ascending=[True, True],
    ).reset_index(drop=True)

    print_header("MODEL RANKING")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best_model_name = results_df.iloc[0]["model"]

    print()
    print(f"[BEST] {best_model_name}")

    best_predictions = prediction_tables[
        best_model_name
    ]

    prediction_df = pd.DataFrame(
        {
            "cow_id": cow_ids,
            "actual_bcs": y,
            "predicted_bcs": best_predictions,
            "absolute_error": np.abs(
                y - best_predictions
            ),
        }
    )

    return (
        results_df,
        best_model_name,
        best_predictions,
        prediction_df,
    )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def calculate_feature_importance(
    best_model,
    X,
    feature_columns,
):
    print_header("FEATURE IMPORTANCE")

    # Fit model on complete labeled dataset.
    best_model.fit(
        X,
        pd.read_csv(LABELS_PATH)[TARGET_COLUMN],
    )

    estimator = best_model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        print(
            "[INFO] Best model does not expose tree feature importance."
        )
        return None

    importances = estimator.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    print(
        importance_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return importance_df


# =============================================================================
# SAVE
# =============================================================================

def save_outputs(
    dataset,
    feature_columns,
    results_df,
    best_model_name,
    prediction_df,
    importance_df,
    X,
    y,
):
    print_header("SAVING BCS MODEL")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = build_models()

    best_model = models[best_model_name]

    print("[TRAIN] Fitting best model on all labeled cows...")

    best_model.fit(
        X,
        y,
    )

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    print(f"[SAVED] Model       : {MODEL_PATH}")

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print(
        f"[SAVED] Predictions : {PREDICTIONS_PATH}"
    )

    if importance_df is not None:
        importance_df.to_csv(
            FEATURE_IMPORTANCE_PATH,
            index=False,
        )

        print(
            f"[SAVED] Importance  : "
            f"{FEATURE_IMPORTANCE_PATH}"
        )

    best_row = results_df.iloc[0]

    metrics = {
        "best_model": best_model_name,
        "sample_count": int(len(dataset)),
        "feature_count": int(len(feature_columns)),
        "cross_validation_folds": N_SPLITS,
        "mae": float(best_row["MAE"]),
        "rmse": float(best_row["RMSE"]),
        "r2": float(best_row["R2"]),
        "feature_columns": feature_columns,
        "excluded_columns": EXCLUDED_COLUMNS,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(
        f"[SAVED] Metrics     : {METRICS_PATH}"
    )

    return best_model


# =============================================================================
# MAIN
# =============================================================================

def main():

    features, labels = load_data()

    scores = validate_labels(labels)

    dataset = merge_dataset(
        features,
        labels,
    )

    feature_columns = select_features(
        dataset,
    )

    X = dataset[feature_columns].copy()

    y = pd.to_numeric(
        dataset[TARGET_COLUMN],
        errors="raise",
    ).to_numpy()

    cow_ids = dataset["cow_id"].to_numpy()

    results_df, best_model_name, predictions, prediction_df = (
        evaluate_models(
            X,
            y,
            cow_ids,
        )
    )

    # Build best model for feature importance.
    models = build_models()
    best_model = models[best_model_name]

    importance_df = None

    if hasattr(
        best_model.named_steps["model"],
        "feature_importances_",
    ):
        # Fit temporarily to obtain importance.
        best_model.fit(X, y)

        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": (
                    best_model
                    .named_steps["model"]
                    .feature_importances_
                ),
            }
        )

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

    save_outputs(
        dataset=dataset,
        feature_columns=feature_columns,
        results_df=results_df,
        best_model_name=best_model_name,
        prediction_df=prediction_df,
        importance_df=importance_df,
        X=X,
        y=y,
    )

    # -------------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------------

    print_header("BCS MODEL TRAINING COMPLETE")

    best = results_df.iloc[0]

    print(f"Samples              : {len(dataset)}")
    print(f"Features             : {len(feature_columns)}")
    print(f"Best model           : {best_model_name}")
    print(f"Cross-validation     : {N_SPLITS}-fold")
    print()
    print(f"Cross-val MAE        : {best['MAE']:.4f}")
    print(f"Cross-val RMSE       : {best['RMSE']:.4f}")
    print(f"Cross-val R²         : {best['R2']:.4f}")
    print()
    print(f"Model                : {MODEL_PATH}")
    print(f"Metrics              : {METRICS_PATH}")
    print(f"Predictions          : {PREDICTIONS_PATH}")
    print(f"Feature importance   : {FEATURE_IMPORTANCE_PATH}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()