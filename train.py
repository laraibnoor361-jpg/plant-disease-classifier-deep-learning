"""
LeafCheck - training and testing script
=======================================
Trains a MobileNetV2 (transfer learning) model on a folder of leaf photos,
tests it on photos it has never seen, and saves everything the Streamlit app needs
into the  models/  folder:

    models/plant_disease_model.keras   <- the trained model
    models/class_names.json            <- list of classes (same order as the model output)
    models/history.json                <- accuracy curves + test result (shown in the app)

Dataset layout expected (one sub-folder per class):

    some_folder/
        Apple___Apple_scab/        img1.jpg  img2.jpg ...
        Apple___healthy/           ...
        Tomato___Late_blight/      ...

Example (Kaggle "New Plant Diseases Dataset (Augmented)" - use its  train  folder):

    python train.py --data_dir "path/to/train" --epochs 5 --fine_tune_epochs 3 --max_per_class 400
"""
import argparse
import json
import random
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

layers = keras.layers

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
OUT_DIR = Path("models")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42


# --------------------------------------------------------------------------- data
def find_class_root(path):
    """Accept the dataset root and automatically step into its 'train' folder if there is one."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Folder not found: {path}")
    trains = sorted(p for p in path.rglob("train") if p.is_dir())
    return trains[0] if trains else path


def list_images(data_dir, max_per_class):
    """Return class names + a list of (file path, class index)."""
    data_dir = find_class_root(data_dir)
    class_names = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if len(class_names) < 2:
        raise SystemExit(f"Need at least 2 class folders inside {data_dir}")
    rng = random.Random(SEED)
    items = []
    for idx, name in enumerate(class_names):
        files = [p for p in (data_dir / name).iterdir() if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        items += [(str(p), idx) for p in files]
    rng.shuffle(items)
    print(f"Using {len(items)} images from {len(class_names)} classes in: {data_dir}")
    return class_names, items


def split_items(items, val_frac=0.15, test_frac=0.15):
    n = len(items)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return train, val, test


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return img, label


def make_dataset(items, batch_size, training):
    paths = [p for p, _ in items]
    labels = [y for _, y in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(items), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


# -------------------------------------------------------------------------- model
def build_model(num_classes):
    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = augment(inputs)                       # only active while training
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)   # pixels 0..255 -> -1..1 (what MobileNetV2 expects)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def merge_history(*histories):
    out = {k: [] for k in ("accuracy", "val_accuracy", "loss", "val_loss")}
    for h in histories:
        for k in out:
            out[k] += [float(v) for v in h.history.get(k, [])]
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train the LeafCheck plant disease model")
    ap.add_argument("--data_dir", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--epochs", type=int, default=5, help="epochs with the base model frozen")
    ap.add_argument("--fine_tune_epochs", type=int, default=3, help="extra epochs unfreezing the top layers")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_per_class", type=int, default=400,
                    help="use at most this many images per class (keeps training fast). 0 = use all")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)

    class_names, items = list_images(args.data_dir, args.max_per_class)
    train_items, val_items, test_items = split_items(items)
    print(f"Train: {len(train_items)}   Validation: {len(val_items)}   Test: {len(test_items)}")

    train_ds = make_dataset(train_items, args.batch_size, training=True)
    val_ds = make_dataset(val_items, args.batch_size, training=False)
    test_ds = make_dataset(test_items, args.batch_size, training=False)

    model, base = build_model(len(class_names))

    # Stage 1 - train only the new classification head
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # Stage 2 - fine-tune the top layers of MobileNetV2 with a small learning rate
    h2 = None
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        compile_model(model, 1e-5)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs)

    # Test on images the model never saw during training
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTEST accuracy: {test_acc * 100:.2f}%")

    OUT_DIR.mkdir(exist_ok=True)
    model.save(OUT_DIR / "plant_disease_model.keras")
    (OUT_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    history = merge_history(*(h for h in (h1, h2) if h is not None))
    history["meta"] = {
        "num_classes": len(class_names),
        "train_images": len(train_items),
        "val_images": len(val_items),
        "test_images": len(test_items),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs": args.epochs + max(args.fine_tune_epochs, 0),
        "architecture": "MobileNetV2 (transfer learning)",
    }
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    size_mb = (OUT_DIR / "plant_disease_model.keras").stat().st_size / 1e6
    print(f"Saved model ({size_mb:.1f} MB) + class_names.json + history.json in '{OUT_DIR}/'")
    print("Next: push the models/ folder to GitHub, then reboot the Streamlit app.")


if __name__ == "__main__":
    main()
