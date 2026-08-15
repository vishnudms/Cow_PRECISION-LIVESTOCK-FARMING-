from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

TRAIN_CSV = (
    PROJECT_ROOT
    / "output"
    / "behavior"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "output"
    / "behavior"
    / "validation.csv"
)

TEST_CSV = (
    PROJECT_ROOT
    / "output"
    / "behavior"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "behavior"
    / "model"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "best_behavior_model.pt"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "behavior_model_metrics.json"
)

TEST_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "test_predictions.csv"
)


# ============================================================
# MODEL CONFIG
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

RANDOM_STATE = 42
NUM_WORKERS = 0

BEHAVIORS = [
    "standing",
    "lying",
    "foraging",
    "drinking",
    "rumination",
]

NUM_CLASSES = len(BEHAVIORS)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class BehaviorDataset(Dataset):

    def __init__(
        self,
        csv_path: Path,
        transform=None,
    ):
        self.df = pd.read_csv(
            csv_path
        ).reset_index(drop=True)

        self.transform = transform

        required = [
            "image_path",
            "x",
            "y",
            "width",
            "height",
            *BEHAVIORS,
        ]

        missing = [
            column
            for column in required
            if column not in self.df.columns
        ]

        if missing:
            raise RuntimeError(
                "Missing dataset columns:\n"
                + "\n".join(missing)
            )

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        numeric_columns = [
            "x",
            "y",
            "width",
            "height",
            *BEHAVIORS,
        ]

        for column in numeric_columns:
            self.df[column] = pd.to_numeric(
                self.df[column],
                errors="coerce",
            )

        # ----------------------------------------------------
        # Remove unusable rows
        # ----------------------------------------------------

        before = len(self.df)

        self.df = self.df.dropna(
            subset=[
                "image_path",
                "x",
                "y",
                "width",
                "height",
            ]
        ).copy()

        self.df = self.df[
            (self.df["width"] > 0)
            & (self.df["height"] > 0)
        ].copy()

        self.df = self.df.reset_index(
            drop=True
        )

        removed = before - len(self.df)

        if removed > 0:
            print(
                f"[DATASET] Removed {removed} invalid rows from {csv_path.name}"
            )

        # ----------------------------------------------------
        # Verify image files
        # ----------------------------------------------------

        missing_files = [
            path
            for path in self.df["image_path"]
            if not Path(path).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Missing behavior image files. "
                "First examples:\n"
                + "\n".join(
                    missing_files[:10]
                )
            )

    def __len__(self):
        return len(self.df)

    # ========================================================
    # SAFE CROP
    # ========================================================

    @staticmethod
    def safe_crop(
        image: Image.Image,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> Image.Image:

        image_width, image_height = image.size

        # ----------------------------------------------------
        # Validate original annotation
        # ----------------------------------------------------

        if not np.isfinite(x):
            x = 0.0

        if not np.isfinite(y):
            y = 0.0

        if not np.isfinite(width):
            width = 1.0

        if not np.isfinite(height):
            height = 1.0

        width = max(
            float(width),
            1.0,
        )

        height = max(
            float(height),
            1.0,
        )

        # ----------------------------------------------------
        # Original rectangle
        # ----------------------------------------------------

        left = float(x)
        top = float(y)
        right = float(
            x + width
        )
        bottom = float(
            y + height
        )

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        pad_x = max(
            2.0,
            width * 0.08,
        )

        pad_y = max(
            2.0,
            height * 0.08,
        )

        left -= pad_x
        top -= pad_y
        right += pad_x
        bottom += pad_y

        # ----------------------------------------------------
        # Clip against the actual image
        # ----------------------------------------------------

        left = max(
            0.0,
            min(
                left,
                max(
                    0.0,
                    image_width - 1,
                ),
            ),
        )

        top = max(
            0.0,
            min(
                top,
                max(
                    0.0,
                    image_height - 1,
                ),
            ),
        )

        right = max(
            1.0,
            min(
                right,
                float(image_width),
            ),
        )

        bottom = max(
            1.0,
            min(
                bottom,
                float(image_height),
            ),
        )

        # ----------------------------------------------------
        # Convert to integer coordinates
        # ----------------------------------------------------

        x1 = int(
            np.floor(left)
        )

        y1 = int(
            np.floor(top)
        )

        x2 = int(
            np.ceil(right)
        )

        y2 = int(
            np.ceil(bottom)
        )

        # ----------------------------------------------------
        # Final boundary clamp
        # ----------------------------------------------------

        x1 = max(
            0,
            min(
                x1,
                image_width - 1,
            ),
        )

        y1 = max(
            0,
            min(
                y1,
                image_height - 1,
            ),
        )

        x2 = max(
            x1 + 1,
            min(
                x2,
                image_width,
            ),
        )

        y2 = max(
            y1 + 1,
            min(
                y2,
                image_height,
            ),
        )

        # ----------------------------------------------------
        # Final safety check
        # ----------------------------------------------------

        if (
            x2 <= x1
            or y2 <= y1
        ):
            # Last-resort valid 2x2 crop.
            x1 = min(
                max(0, x1),
                max(0, image_width - 2),
            )

            y1 = min(
                max(0, y1),
                max(0, image_height - 2),
            )

            x2 = min(
                image_width,
                x1 + 2,
            )

            y2 = min(
                image_height,
                y1 + 2,
            )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            raise RuntimeError(
                "Unable to construct a valid crop: "
                f"image={image_width}x{image_height}, "
                f"box={x1},{y1},{x2},{y2}"
            )

        crop = image.crop(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )

        return crop

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        row = self.df.iloc[
            index
        ]

        image_path = Path(
            row["image_path"]
        )

        try:
            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

        except Exception as exc:
            raise RuntimeError(
                f"Unable to load image:\n"
                f"{image_path}\n"
                f"{exc}"
            ) from exc

        # ----------------------------------------------------
        # Read annotation
        # ----------------------------------------------------

        x = float(row["x"])
        y = float(row["y"])
        width = float(row["width"])
        height = float(row["height"])

        # ----------------------------------------------------
        # Safe cow crop
        # ----------------------------------------------------

        crop = self.safe_crop(
            image=image,
            x=x,
            y=y,
            width=width,
            height=height,
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        if self.transform is not None:
            crop = self.transform(
                crop
            )

        # ----------------------------------------------------
        # Multi-label target
        # ----------------------------------------------------

        labels = torch.tensor(
            [
                float(row[behavior])
                for behavior in BEHAVIORS
            ],
            dtype=torch.float32,
        )

        sequence_id = str(
            row.get(
                "sequence_id",
                "",
            )
        )

        image_name = str(
            row.get(
                "image_name",
                image_path.name,
            )
        )

        return (
            crop,
            labels,
            sequence_id,
            image_name,
            int(index),
        )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=5
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


eval_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# MODEL
# ============================================================

def build_model():

    try:
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        pretrained = True

    except Exception as exc:

        print(
            "[WARNING] Could not load pretrained ResNet18:"
        )

        print(
            exc
        )

        print(
            "[INFO] Using randomly initialized model."
        )

        model = models.resnet18(
            weights=None
        )

        pretrained = False

    in_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(
        nn.Dropout(
            p=0.25
        ),
        nn.Linear(
            in_features,
            NUM_CLASSES,
        ),
    )

    print(
        f"[MODEL] ResNet18 | pretrained={pretrained}"
    )

    return model


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print(
            "[DEVICE] CUDA"
        )

        print(
            "[GPU]",
            torch.cuda.get_device_name(0),
        )

        return device

    print(
        "[DEVICE] CPU"
    )

    return torch.device(
        "cpu"
    )


# ============================================================
# POSITIVE CLASS WEIGHTS
# ============================================================

def calculate_pos_weights(
    dataset: BehaviorDataset,
):

    positive_counts = (
        dataset.df[
            BEHAVIORS
        ]
        .sum()
        .astype(float)
    )

    negative_counts = (
        len(dataset.df)
        - positive_counts
    )

    pos_weight = (
        negative_counts
        / positive_counts.replace(
            0,
            1,
        )
    )

    pos_weight = pos_weight.clip(
        lower=1.0,
        upper=20.0,
    )

    return torch.tensor(
        pos_weight.values,
        dtype=torch.float32,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    for (
        images,
        labels,
        _,
        _,
        _,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

    return (
        total_loss
        / max(
            1,
            len(loader.dataset),
        )
    )


# ============================================================
# PREDICT DATASET
# ============================================================

@torch.no_grad()
def predict_dataset(
    model,
    loader,
    device,
):

    model.eval()

    all_probabilities = []
    all_labels = []

    all_sequences = []
    all_images = []
    all_indices = []

    for (
        images,
        labels,
        sequences,
        image_names,
        indices,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_labels.append(
            labels.cpu().numpy()
        )

        all_sequences.extend(
            list(sequences)
        )

        all_images.extend(
            list(image_names)
        )

        if torch.is_tensor(indices):
            all_indices.extend(
                indices.cpu().numpy().tolist()
            )
        else:
            all_indices.extend(
                list(indices)
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    return (
        probabilities,
        labels,
        all_indices,
        all_sequences,
        all_images,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold=0.50,
):

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    per_class = {}

    for index, behavior in enumerate(
        BEHAVIORS
    ):

        y_true = labels[
            :,
            index,
        ]

        y_prob = probabilities[
            :,
            index,
        ]

        y_pred = predictions[
            :,
            index,
        ]

        try:
            ap = average_precision_score(
                y_true,
                y_prob,
            )

        except Exception:
            ap = float("nan")

        per_class[
            behavior
        ] = {
            "average_precision": float(
                ap
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "positive_count": int(
                y_true.sum()
            ),
        }

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    valid_ap = [
        value["average_precision"]
        for value in per_class.values()
        if np.isfinite(
            value["average_precision"]
        )
    ]

    macro_ap = (
        float(np.mean(valid_ap))
        if valid_ap
        else 0.0
    )

    return {
        "threshold": float(
            threshold
        ),
        "macro_f1": float(
            macro_f1
        ),
        "micro_f1": float(
            micro_f1
        ),
        "macro_average_precision": float(
            macro_ap
        ),
        "per_behavior": per_class,
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    labels,
    probabilities,
):

    best_threshold = 0.50
    best_score = -1.0

    candidates = np.arange(
        0.20,
        0.81,
        0.05,
    )

    for threshold in candidates:

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        score = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        if score > best_score:

            best_score = float(
                score
            )

            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_score,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(
        RANDOM_STATE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("COW PLF - BEHAVIOR RECOGNITION MODEL")
    print("=" * 80)

    print()
    print(
        "TASK: Multi-label cattle behavior recognition"
    )

    print()
    print(
        "Behaviors:"
    )

    for behavior in BEHAVIORS:
        print(
            f"  - {behavior}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print()
    print(
        "[DATA] Loading datasets..."
    )

    train_dataset = BehaviorDataset(
        TRAIN_CSV,
        transform=train_transform,
    )

    val_dataset = BehaviorDataset(
        VAL_CSV,
        transform=eval_transform,
    )

    test_dataset = BehaviorDataset(
        TEST_CSV,
        transform=eval_transform,
    )

    print(
        f"Train annotations : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation        : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test              : "
        f"{len(test_dataset)}"
    )

    print()
    print(
        f"Train sequences : "
        f"{train_dataset.df['sequence_id'].nunique()}"
    )

    print(
        f"Val sequences   : "
        f"{val_dataset.df['sequence_id'].nunique()}"
    )

    print(
        f"Test sequences  : "
        f"{test_dataset.df['sequence_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": (
            device.type == "cuda"
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    pos_weight = calculate_pos_weights(
        train_dataset
    ).to(
        device
    )

    print()
    print(
        "[LOSS] Positive-class weights:"
    )

    for behavior, weight in zip(
        BEHAVIORS,
        pos_weight.cpu().numpy(),
    ):

        print(
            f"  {behavior:12s}: "
            f"{weight:.3f}"
        )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.50

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print("=" * 80)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 80)

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        (
            val_prob,
            val_labels,
            _,
            _,
            _,
        ) = predict_dataset(
            model,
            val_loader,
            device,
        )

        val_metrics = calculate_metrics(
            val_labels,
            val_prob,
            threshold=0.50,
        )

        val_threshold, threshold_f1 = (
            find_best_threshold(
                val_labels,
                val_prob,
            )
        )

        scheduler.step(
            val_metrics[
                "macro_f1"
            ]
        )

        print(
            f"Train loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val macro F1     : "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val micro F1     : "
            f"{val_metrics['micro_f1']:.4f}"
        )

        print(
            f"Val macro AP     : "
            f"{val_metrics['macro_average_precision']:.4f}"
        )

        print(
            f"Best threshold   : "
            f"{val_threshold:.2f}"
        )

        print(
            f"Thresholded F1   : "
            f"{threshold_f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                "val_macro_f1": float(
                    val_metrics[
                        "macro_f1"
                    ]
                ),
                "val_micro_f1": float(
                    val_metrics[
                        "micro_f1"
                    ]
                ),
                "val_macro_ap": float(
                    val_metrics[
                        "macro_average_precision"
                    ]
                ),
                "threshold": float(
                    val_threshold
                ),
                "threshold_f1": float(
                    threshold_f1
                ),
            }
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_metrics["macro_f1"] > best_val_f1:

            best_val_f1 = float(
                val_metrics[
                    "macro_f1"
                ]
            )

            best_epoch = epoch

            best_threshold = float(
                val_threshold
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "behaviors":
                        BEHAVIORS,

                    "image_size":
                        IMAGE_SIZE,

                    "threshold":
                        best_threshold,

                    "best_validation_macro_f1":
                        best_val_f1,

                    "epoch":
                        best_epoch,
                },
                MODEL_PATH,
            )

            print(
                "[SAVE] New best model saved."
            )

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    best_threshold = float(
        checkpoint.get(
            "threshold",
            0.50,
        )
    )

    # --------------------------------------------------------
    # Final held-out test
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL HELD-OUT TEST")
    print("=" * 80)

    (
        test_prob,
        test_labels,
        test_indices,
        test_sequences,
        test_images,
    ) = predict_dataset(
        model,
        test_loader,
        device,
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_prob,
        threshold=best_threshold,
    )

    test_predictions = (
        test_prob
        >= best_threshold
    ).astype(int)

    print(
        f"Threshold          : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Macro F1           : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Micro F1           : "
        f"{test_metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro Average Prec.: "
        f"{test_metrics['macro_average_precision']:.4f}"
    )

    # --------------------------------------------------------
    # Per behavior
    # --------------------------------------------------------

    print()
    print(
        "PER-BEHAVIOR RESULTS"
    )

    print(
        "-" * 80
    )

    for behavior in BEHAVIORS:

        result = test_metrics[
            "per_behavior"
        ][behavior]

        print(
            f"{behavior:12s} "
            f"AP={result['average_precision']:.4f} "
            f"P={result['precision']:.4f} "
            f"R={result['recall']:.4f} "
            f"F1={result['f1']:.4f} "
            f"N={result['positive_count']}"
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print(
        "CLASSIFICATION REPORT"
    )

    print(
        classification_report(
            test_labels,
            test_predictions,
            target_names=BEHAVIORS,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "image_name":
                test_images,

            "sequence_id":
                test_sequences,
        }
    )

    for index, behavior in enumerate(
        BEHAVIORS
    ):

        prediction_df[
            f"{behavior}_prob"
        ] = test_prob[
            :,
            index,
        ]

        prediction_df[
            f"{behavior}_true"
        ] = test_labels[
            :,
            index,
        ].astype(int)

        prediction_df[
            f"{behavior}_pred"
        ] = test_predictions[
            :,
            index,
        ]

    prediction_df.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "task":
            "multi_label_cattle_behavior",

        "behaviors":
            BEHAVIORS,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            float(
                best_val_f1
            ),

        "test_macro_f1":
            float(
                test_metrics[
                    "macro_f1"
                ]
            ),

        "test_micro_f1":
            float(
                test_metrics[
                    "micro_f1"
                ]
            ),

        "test_macro_average_precision":
            float(
                test_metrics[
                    "macro_average_precision"
                ]
            ),

        "threshold":
            best_threshold,

        "train_annotations":
            len(train_dataset),

        "validation_annotations":
            len(val_dataset),

        "test_annotations":
            len(test_dataset),

        "train_sequences":
            int(
                train_dataset.df[
                    "sequence_id"
                ].nunique()
            ),

        "validation_sequences":
            int(
                val_dataset.df[
                    "sequence_id"
                ].nunique()
            ),

        "test_sequences":
            int(
                test_dataset.df[
                    "sequence_id"
                ].nunique()
            ),

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "history":
            history,

        "per_behavior":
            test_metrics[
                "per_behavior"
            ],
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "BEHAVIOR MODEL TRAINING COMPLETE"
    )
    print("=" * 80)

    print(
        f"Best epoch    : {best_epoch}"
    )

    print(
        f"Test macro F1 : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test macro AP : "
        f"{test_metrics['macro_average_precision']:.4f}"
    )

    print()
    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        f"Metrics     : {METRICS_PATH}"
    )

    print(
        f"Predictions : "
        f"{TEST_PREDICTIONS_PATH}"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()