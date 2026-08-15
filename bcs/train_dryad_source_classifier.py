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
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

TRAIN_CSV = (
    PROJECT_ROOT
    / "output"
    / "bcs_training"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "output"
    / "bcs_training"
    / "validation.csv"
)

TEST_CSV = (
    PROJECT_ROOT
    / "output"
    / "bcs_training"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "bcs_training"
    / "source_classifier"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "best_source_classifier.pt"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "source_classifier_metrics.json"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

RANDOM_STATE = 42

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 6

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

MAX_IMAGES_PER_COW = 20

NUM_WORKERS = 0


# Source classes in the Dryad archive.
SOURCE_CLASSES = [2, 3, 4, 5, 6, 7]

CLASS_TO_INDEX = {
    cls: idx
    for idx, cls in enumerate(SOURCE_CLASSES)
}

INDEX_TO_CLASS = {
    idx: cls
    for cls, idx in CLASS_TO_INDEX.items()
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# ============================================================
# TIFF LOADING
# ============================================================

def load_tiff_as_rgb(path: str) -> Image.Image:
    """
    Load a TIFF robustly, including grayscale/depth-like data.

    The image is converted into a normalized 8-bit grayscale image
    and then replicated into 3 channels so a standard ImageNet
    backbone can be used.
    """

    with Image.open(path) as img:

        arr = np.asarray(img)

    # --------------------------------------------------------
    # Handle unusual TIFF dimensions
    # --------------------------------------------------------

    if arr.ndim == 3:

        # If multiple channels exist, use the first channel.
        arr = arr[..., 0]

    if arr.ndim != 2:

        raise ValueError(
            f"Unsupported TIFF shape {arr.shape}: {path}"
        )

    # --------------------------------------------------------
    # Convert to float
    # --------------------------------------------------------

    arr = arr.astype(
        np.float32,
        copy=False,
    )

    # --------------------------------------------------------
    # Robust per-image normalization
    # --------------------------------------------------------

    finite = np.isfinite(arr)

    if not finite.any():

        arr = np.zeros_like(
            arr,
            dtype=np.float32,
        )

    else:

        values = arr[finite]

        low = np.percentile(
            values,
            1,
        )

        high = np.percentile(
            values,
            99,
        )

        if high <= low:

            low = float(
                values.min()
            )

            high = float(
                values.max()
            )

        if high > low:

            arr = (
                arr - low
            ) / (
                high - low
            )

        else:

            arr = np.zeros_like(
                arr,
                dtype=np.float32,
            )

        arr = np.clip(
            arr,
            0.0,
            1.0,
        )

    arr = (
        arr * 255.0
    ).astype(
        np.uint8
    )

    gray = Image.fromarray(
        arr,
        mode="L",
    )

    return gray.convert(
        "RGB"
    )


# ============================================================
# DATASET
# ============================================================

class DryadDataset(Dataset):

    def __init__(
        self,
        csv_path: Path,
        transform=None,
        max_images_per_cow: int | None = None,
        seed: int = 42,
    ):

        self.df = pd.read_csv(
            csv_path
        )

        self.transform = transform

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        required = [
            "image_path",
            "source_class",
            "cow_group",
        ]

        missing = [
            c
            for c in required
            if c not in self.df.columns
        ]

        if missing:

            raise RuntimeError(
                f"Missing dataset columns: {missing}"
            )

        # ----------------------------------------------------
        # Keep only known classes
        # ----------------------------------------------------

        self.df = self.df[
            self.df["source_class"]
            .astype(int)
            .isin(SOURCE_CLASSES)
        ].copy()

        # ----------------------------------------------------
        # Deterministic per-cow sampling
        #
        # This reduces dominance by cows with hundreds of
        # nearly identical frames.
        # ----------------------------------------------------

        if max_images_per_cow is not None:

            sampled = []

            rng = random.Random(
                seed
            )

            for cow_group, group in self.df.groupby(
                "cow_group"
            ):

                group = group.sample(
                    frac=1.0,
                    random_state=seed,
                )

                if len(group) > max_images_per_cow:

                    indices = np.linspace(
                        0,
                        len(group) - 1,
                        max_images_per_cow,
                        dtype=int,
                    )

                    group = group.iloc[
                        sorted(
                            set(indices)
                        )
                    ]

                sampled.append(
                    group
                )

            self.df = pd.concat(
                sampled,
                ignore_index=True,
            )

        self.df = self.df.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Verify files
        # ----------------------------------------------------

        missing_files = [
            path
            for path in self.df["image_path"]
            if not Path(path).exists()
        ]

        if missing_files:

            raise FileNotFoundError(
                "Missing image files. First missing:\n"
                + "\n".join(
                    missing_files[:10]
                )
            )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[
            index
        ]

        path = row["image_path"]

        source_class = int(
            row["source_class"]
        )

        label = CLASS_TO_INDEX[
            source_class
        ]

        try:

            image = load_tiff_as_rgb(
                path
            )

        except Exception as exc:

            raise RuntimeError(
                f"Could not load:\n{path}\n{exc}"
            )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return (
            image,
            label,
            source_class,
            row["cow_group"],
            path,
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
        transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
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
            "[WARNING] Could not load pretrained ResNet18 weights."
        )

        print(
            f"Reason: {exc}"
        )

        print(
            "[INFO] Falling back to randomly initialized ResNet18."
        )

        model = models.resnet18(
            weights=None
        )

        pretrained = False

    in_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        in_features,
        len(SOURCE_CLASSES),
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
            f"[DEVICE] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        return device

    print(
        "[DEVICE] CPU"
    )

    return torch.device(
        "cpu"
    )


# ============================================================
# TRAINING
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

    all_true = []
    all_pred = []

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):

        images, labels, _, _, _ = batch

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

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        all_true.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    mean_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_true,
        all_pred,
    )

    macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0,
    )

    return (
        mean_loss,
        accuracy,
        macro_f1,
    )


# ============================================================
# IMAGE-LEVEL EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    records = []

    for batch in loader:

        (
            images,
            labels,
            source_classes,
            cow_groups,
            paths,
        ) = batch

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        confidences = (
            probabilities.max(
                dim=1
            ).values
        )

        for i in range(
            len(paths)
        ):

            true_source = int(
                source_classes[i]
            )

            pred_index = int(
                predictions[i]
                .cpu()
                .item()
            )

            pred_source = INDEX_TO_CLASS[
                pred_index
            ]

            records.append(
                {
                    "image_path": paths[i],
                    "cow_group": cow_groups[i],
                    "true_source_class": true_source,
                    "pred_source_class": pred_source,
                    "confidence": float(
                        confidences[i]
                        .cpu()
                        .item()
                    ),
                }
            )

    predictions_df = pd.DataFrame(
        records
    )

    image_accuracy = accuracy_score(
        predictions_df[
            "true_source_class"
        ],
        predictions_df[
            "pred_source_class"
        ],
    )

    image_macro_f1 = f1_score(
        predictions_df[
            "true_source_class"
        ],
        predictions_df[
            "pred_source_class"
        ],
        labels=SOURCE_CLASSES,
        average="macro",
        zero_division=0,
    )

    mean_loss = (
        total_loss
        / len(loader.dataset)
    )

    return (
        mean_loss,
        image_accuracy,
        image_macro_f1,
        predictions_df,
    )


# ============================================================
# COW-LEVEL EVALUATION
# ============================================================

def aggregate_by_cow(
    predictions_df,
):

    rows = []

    for cow_group, group in predictions_df.groupby(
        "cow_group"
    ):

        true_class = int(
            group[
                "true_source_class"
            ].mode().iloc[0]
        )

        # Mean probability is unavailable in this compact
        # prediction table, so use majority vote.
        pred_class = int(
            group[
                "pred_source_class"
            ].mode().iloc[0]
        )

        correct = (
            true_class
            == pred_class
        )

        rows.append(
            {
                "cow_group": cow_group,
                "true_source_class": true_class,
                "pred_source_class": pred_class,
                "correct": bool(correct),
                "image_count": len(group),
            }
        )

    return pd.DataFrame(
        rows
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

    print()
    print("=" * 80)
    print("COW PLF - DRYAD SOURCE-CLASS BCS MODEL")
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This experiment predicts the Dryad source classes 2-7."
    )

    print(
        "Those classes are NOT yet being claimed as your 1-5 BCS scale."
    )

    print()

    device = get_device()

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print()
    print("[DATA] Loading datasets...")

    train_dataset = DryadDataset(
        TRAIN_CSV,
        transform=train_transform,
        max_images_per_cow=MAX_IMAGES_PER_COW,
        seed=RANDOM_STATE,
    )

    val_dataset = DryadDataset(
        VAL_CSV,
        transform=eval_transform,
        max_images_per_cow=MAX_IMAGES_PER_COW,
        seed=RANDOM_STATE,
    )

    test_dataset = DryadDataset(
        TEST_CSV,
        transform=eval_transform,
        max_images_per_cow=MAX_IMAGES_PER_COW,
        seed=RANDOM_STATE,
    )

    print(
        f"Train images      : {len(train_dataset)}"
    )

    print(
        f"Validation images : {len(val_dataset)}"
    )

    print(
        f"Test images       : {len(test_dataset)}"
    )

    print()
    print(
        f"Train cow groups  : "
        f"{train_dataset.df['cow_group'].nunique()}"
    )

    print(
        f"Val cow groups    : "
        f"{val_dataset.df['cow_group'].nunique()}"
    )

    print(
        f"Test cow groups   : "
        f"{test_dataset.df['cow_group'].nunique()}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
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
    # Loss
    # --------------------------------------------------------

    # Class weighting at the cow-group level.
    class_counts = (
        train_dataset.df[
            "source_class"
        ]
        .value_counts()
        .reindex(
            SOURCE_CLASSES,
            fill_value=0,
        )
    )

    weights = (
        class_counts.sum()
        / class_counts.replace(
            0,
            1,
        )
    )

    weights = (
        weights
        / weights.mean()
    )

    class_weights = torch.tensor(
        weights.values,
        dtype=torch.float32,
        device=device,
    )

    print()
    print(
        "[LOSS] Class weights:"
    )

    print(
        {
            cls: round(
                float(weights.loc[cls]),
                3,
            )
            for cls in SOURCE_CLASSES
        }
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

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

        train_loss, train_acc, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        val_loss, val_acc, val_f1, _ = (
            evaluate(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        scheduler.step(
            val_f1
        )

        print(
            f"Train loss : {train_loss:.4f}"
        )

        print(
            f"Train acc  : {train_acc:.4f}"
        )

        print(
            f"Train F1   : {train_f1:.4f}"
        )

        print(
            f"Val loss   : {val_loss:.4f}"
        )

        print(
            f"Val acc    : {val_acc:.4f}"
        )

        print(
            f"Val F1     : {val_f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "source_classes": SOURCE_CLASSES,
                    "class_to_index": CLASS_TO_INDEX,
                    "image_size": IMAGE_SIZE,
                },
                MODEL_FILE,
            )

            print(
                "[SAVE] New best model saved."
            )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL HELD-OUT TEST")
    print("=" * 80)

    (
        test_loss,
        test_image_accuracy,
        test_image_f1,
        predictions,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    cow_predictions = aggregate_by_cow(
        predictions
    )

    cow_accuracy = accuracy_score(
        cow_predictions[
            "true_source_class"
        ],
        cow_predictions[
            "pred_source_class"
        ],
    )

    cow_f1 = f1_score(
        cow_predictions[
            "true_source_class"
        ],
        cow_predictions[
            "pred_source_class"
        ],
        labels=SOURCE_CLASSES,
        average="macro",
        zero_division=0,
    )

    print(
        f"Test loss          : {test_loss:.4f}"
    )

    print(
        f"Image accuracy     : {test_image_accuracy:.4f}"
    )

    print(
        f"Image macro F1     : {test_image_f1:.4f}"
    )

    print(
        f"Cow-group accuracy : {cow_accuracy:.4f}"
    )

    print(
        f"Cow-group macro F1 : {cow_f1:.4f}"
    )

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 80)

    print(
        classification_report(
            predictions[
                "true_source_class"
            ],
            predictions[
                "pred_source_class"
            ],
            labels=SOURCE_CLASSES,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        predictions[
            "true_source_class"
        ],
        predictions[
            "pred_source_class"
        ],
        labels=SOURCE_CLASSES,
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        pd.DataFrame(
            cm,
            index=[
                f"true_{c}"
                for c in SOURCE_CLASSES
            ],
            columns=[
                f"pred_{c}"
                for c in SOURCE_CLASSES
            ],
        ).to_string()
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    cow_predictions.to_csv(
        OUTPUT_DIR
        / "test_cow_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_image_accuracy": test_image_accuracy,
        "test_image_macro_f1": test_image_f1,
        "test_cow_group_accuracy": cow_accuracy,
        "test_cow_group_macro_f1": cow_f1,
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "test_images": len(test_dataset),
        "train_cow_groups": int(
            train_dataset.df[
                "cow_group"
            ].nunique()
        ),
        "validation_cow_groups": int(
            val_dataset.df[
                "cow_group"
            ].nunique()
        ),
        "test_cow_groups": int(
            test_dataset.df[
                "cow_group"
            ].nunique()
        ),
        "source_classes": SOURCE_CLASSES,
        "max_images_per_cow": MAX_IMAGES_PER_COW,
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "history": history,
    }

    with open(
        METRICS_FILE,
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
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Best epoch        : {best_epoch}"
    )

    print(
        f"Best validation F1: {best_val_f1:.4f}"
    )

    print(
        f"Test image acc    : {test_image_accuracy:.4f}"
    )

    print(
        f"Test cow-group acc: {cow_accuracy:.4f}"
    )

    print()
    print(
        f"Model   : {MODEL_FILE}"
    )

    print(
        f"Metrics : {METRICS_FILE}"
    )

    print(
        f"Predictions: {PREDICTIONS_FILE}"
    )

    print()
    print(
        "IMPORTANT: These metrics evaluate the Dryad source classes."
    )

    print(
        "They are not yet validation of your 1-5 BCS system."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()