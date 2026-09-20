"""
Copies a few leaf photos from your dataset into  sample_images/  so visitors of the
Streamlit app can try the model with one click (no upload needed).

    python make_samples.py --data_dir "path/to/train" --max_classes 12
"""
import argparse
import random
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_class_root(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Folder not found: {path}")
    trains = sorted(p for p in path.rglob("train") if p.is_dir())
    return trains[0] if trains else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--per_class", type=int, default=1)
    ap.add_argument("--max_classes", type=int, default=12)
    args = ap.parse_args()

    root = find_class_root(args.data_dir)
    classes = sorted(d for d in root.iterdir() if d.is_dir())
    rng = random.Random(7)
    rng.shuffle(classes)

    out = Path("sample_images")
    out.mkdir(exist_ok=True)
    saved = 0
    for cls in classes[: args.max_classes]:
        files = [p for p in cls.iterdir() if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(files)
        for i, f in enumerate(files[: args.per_class], start=1):
            img = Image.open(f).convert("RGB").resize((256, 256))
            img.save(out / f"{cls.name}_{i}.jpg", quality=88)
            saved += 1
    print(f"Saved {saved} sample photos in '{out}/'")


if __name__ == "__main__":
    main()
