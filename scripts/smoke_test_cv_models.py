from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_NAMES, IMAGE_SIZE

NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 16
EPOCHS = 2
SEED = 42

tf.keras.utils.set_random_seed(SEED)

def resolve_path(value):
    p = Path(str(value))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())

def load_manifest():
    path = PROJECT_ROOT / "experiments" / "five_fold_cv" / "fold_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Fold manifest not found: {path}")

    df = pd.read_csv(path)
    required = {"path", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest needs columns {required}; found {list(df.columns)}")

    fold_col = next(
        (c for c in ["fold", "cv_fold", "validation_fold", "fold_id"]
         if c in df.columns),
        None
    )
    if fold_col is None:
        raise ValueError(
            "Could not find the fold assignment column. "
            f"Available columns: {list(df.columns)}"
        )

    # Fold 1 is the validation set; all other folds form training.
    fold_values = pd.to_numeric(df[fold_col], errors="raise")
    val_df = df.loc[fold_values == 1].copy()
    train_df = df.loc[fold_values != 1].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Unexpected fold values in {fold_col}: "
            f"{sorted(fold_values.unique().tolist())}"
        )

    for frame in (train_df, val_df):
        frame["resolved_path"] = frame["path"].map(resolve_path)
        missing = [p for p in frame["resolved_path"] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image paths are missing. Example: {missing[0]}"
            )

    class_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}

    def encode(series):
        labels = series.astype(str).str.strip()
        if set(labels).issubset(class_to_index):
            return labels.map(class_to_index).to_numpy(dtype=np.int32)

        numeric = pd.to_numeric(labels, errors="raise").to_numpy(dtype=np.int32)
        if np.any(numeric < 0) or np.any(numeric >= NUM_CLASSES):
            raise ValueError(
                f"Numeric labels must be in [0, {NUM_CLASSES - 1}]."
            )
        return numeric

    train_df["encoded_label"] = encode(train_df["label"])
    val_df["encoded_label"] = encode(val_df["label"])

    print(f"Fold column: {fold_col}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Classes: {CLASS_NAMES}")

    return train_df, val_df

def make_dataset(frame, training=False):
    paths = frame["resolved_path"].to_numpy()
    labels = frame["encoded_label"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(frame), 2000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            image, IMAGE_SIZE, method="bilinear", antialias=True
        )
        # EfficientNetB0 includes its own rescaling/preprocessing layers.
        image = tf.cast(image, tf.float32)
        return image, label

    ds = ds.map(load_image, num_parallel_calls=2)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(1)

def build_fresh_backbone():
    return tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

def add_classifier(features, name):
    x = tf.keras.layers.Dropout(0.35)(features)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="predictions"
    )(x)
    return outputs

def build_model(model_type):
    if model_type == "efficientnet_b0":
        # Reuse the project's official stage-1 builder.
        from src.models.efficientnet import build_efficientnet_b0_stage1
        model = build_efficientnet_b0_stage1(
            num_classes=NUM_CLASSES,
            learning_rate=1e-3
        )
        return model

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image"
    )
    base = build_fresh_backbone()
    base.trainable = False

    if model_type == "multiscale_fusion":
        extractor = tf.keras.Model(
            base.input,
            [
                base.get_layer("block6d_add").output,
                base.get_layer("top_activation").output
            ],
            name="fresh_multiscale_extractor"
        )
        extractor.trainable = False
        early_map, deep_map = extractor(inputs)
        early = tf.keras.layers.GlobalAveragePooling2D()(early_map)
        deep = tf.keras.layers.GlobalAveragePooling2D()(deep_map)
        fused = tf.keras.layers.Concatenate(name="multiscale_concat")(
            [early, deep]
        )
        outputs = add_classifier(fused, "multiscale")
        model = tf.keras.Model(inputs, outputs, name="multiscale_fusion")

    elif model_type == "spatial_fusion":
        extractor = tf.keras.Model(
            base.input,
            [
                base.get_layer("block3b_add").output,
                base.get_layer("top_activation").output
            ],
            name="fresh_spatial_extractor"
        )
        extractor.trainable = False
        early_map, deep_map = extractor(inputs)

        early = tf.keras.layers.Conv2D(
            64, 1, activation="relu", name="early_projection"
        )(early_map)
        early = tf.keras.layers.AveragePooling2D(
            pool_size=4, name="early_downsample"
        )(early)

        deep = tf.keras.layers.Conv2D(
            128, 1, activation="relu", name="deep_projection"
        )(deep_map)

        fused = tf.keras.layers.Concatenate(name="spatial_concat")(
            [early, deep]
        )
        fused = tf.keras.layers.Conv2D(
            128, 3, padding="same", activation="relu", name="fusion_conv"
        )(fused)
        fused = tf.keras.layers.GlobalAveragePooling2D()(fused)
        outputs = add_classifier(fused, "spatial")
        model = tf.keras.Model(inputs, outputs, name="spatial_fusion")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "efficientnet_b0",
            "multiscale_fusion",
            "spatial_fusion"
        ]
    )
    args = parser.parse_args()

    output_dir = (
        PROJECT_ROOT / "experiments" / "five_fold_cv" /
        "smoke_tests" / args.model / "fold_1"
    )

    # Never overwrite any previous smoke-test results.
    if output_dir.exists():
        raise FileExistsError(
            f"Output folder already exists; refusing to overwrite:\n{output_dir}\n"
            "Choose a new output folder or move the old folder manually."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    train_df, val_df = load_manifest()
    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)

    model = build_model(args.model)
    print(f"\nModel: {args.model}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Output directory: {output_dir}")

    checkpoint_path = output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the saved best checkpoint, not just the final epoch.
    best_model = tf.keras.models.load_model(checkpoint_path)
    val_loss, val_accuracy = best_model.evaluate(val_ds, verbose=1)

    from sklearn.metrics import (
        precision_score, recall_score, f1_score, classification_report
    )

    y_true = val_df["encoded_label"].to_numpy(dtype=np.int32)
    probabilities = best_model.predict(val_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    metrics = {
        "model": args.model,
        "fold": 1,
        "smoke_test": True,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history.history["loss"]),
        "validation_loss": float(val_loss),
        "accuracy": float(val_accuracy),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "class_names": CLASS_NAMES,
        "note": "Smoke test only; not a full CV result or official test result."
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    (output_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    print("\nSMOKE TEST RESULTS")
    for key in [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "validation_loss"
    ]:
        print(f"{key}: {metrics[key]:.4f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nSmoke test complete; this is not the full five-fold experiment.")

if __name__ == "__main__":
    main()
