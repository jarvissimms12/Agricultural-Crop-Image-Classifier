"""
Agricultural crop image classifier: ResNet50 transfer learning, 30 classes.

Pipeline:
  1. Split images 80/10/10 into train/val/test (split-folders, fixed seed)
  2. Phase 1: train a new head on a frozen ImageNet ResNet50
  3. Phase 2: fine-tune the last ResNet block at a low learning rate
  4. Evaluate ONCE on the held-out test split; write metrics and curves

Dataset: https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification
(unzip to ./Agricultural-crops, one folder per class)

    python main.py --epochs-head 15 --epochs-finetune 10

Outputs (results/): metrics.json, per_class_report.txt, training_curves.png,
CropModel.keras
"""
import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 500


def make_generators(input_folder, split_folder):
    if os.path.exists(split_folder):
        shutil.rmtree(split_folder)
    splitfolders.ratio(input_folder, output=split_folder, seed=SEED, ratio=(0.8, 0.1, 0.1))

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
        fill_mode="nearest")
    plain_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    flow = lambda gen, sub, shuffle: gen.flow_from_directory(
        os.path.join(split_folder, sub), target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="categorical", shuffle=shuffle, seed=SEED)
    # test is not shuffled so predictions line up with test.classes
    return flow(train_gen, "train", True), flow(plain_gen, "val", False), flow(plain_gen, "test", False)


def build_model(n_classes):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return model, base


def main(input_folder, out, epochs_head, epochs_finetune):
    tf.random.set_seed(SEED)
    os.makedirs(out, exist_ok=True)
    train, val, test = make_generators(input_folder, "Processed_Data")
    model, base = build_model(train.num_classes)
    stop = callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)

    # Phase 1: head only
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train, validation_data=val, epochs=epochs_head, callbacks=[stop])

    # Phase 2: unfreeze the last ResNet stage (conv5_*) and fine-tune gently
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith("conv5_")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train, validation_data=val, epochs=epochs_finetune, callbacks=[stop])

    # Held-out test set, evaluated once
    test_loss, test_acc = model.evaluate(test)
    probs = model.predict(test)
    y_pred = probs.argmax(axis=1)
    names = list(test.class_indices)
    report = classification_report(test.classes, y_pred, target_names=names, digits=3)
    top3 = float(np.mean([t in np.argsort(p)[-3:] for t, p in zip(test.classes, probs)]))

    hist = {k: h1.history[k] + h2.history[k] for k in ("accuracy", "val_accuracy", "loss", "val_loss")}
    metrics = {
        "n_classes": train.num_classes,
        "images": {"train": train.samples, "val": val.samples, "test": test.samples},
        "test_accuracy": round(float(test_acc), 4),
        "test_top3_accuracy": round(top3, 4),
        "test_loss": round(float(test_loss), 4),
        "final_train_accuracy": round(hist["accuracy"][-1], 4),
        "best_val_accuracy": round(max(hist["val_accuracy"]), 4),
        "epochs_run": len(hist["accuracy"]),
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out, "per_class_report.txt"), "w") as f:
        f.write(report)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("accuracy", "loss")):
        ax.plot(hist[key], label="train"); ax.plot(hist[f"val_{key}"], label="validation")
        ax.axvline(len(h1.history[key]) - 0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{key.title()} (dashed line: fine-tuning starts)"); ax.set_xlabel("epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "training_curves.png"), dpi=120)

    model.save(os.path.join(out, "CropModel.keras"))
    print(json.dumps(metrics, indent=2))
    print(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-folder", default="Agricultural-crops")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs-head", type=int, default=15)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    a = ap.parse_args()
    main(a.input_folder, a.out, a.epochs_head, a.epochs_finetune)
