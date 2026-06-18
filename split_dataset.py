import os
import shutil
import random

# ── APNA PATH YAHAN LIKHO ──────────────────────────────────────────
BASE_DIR = r"D:\University\projects\traffic violation advance\data\helmet_dataset"   # jahan with_helmet or without_helmet folders hain
OUTPUT_DIR = r"D:\University\projects\traffic violation advance\data\helmet_dataset" # same jagah split ho jayega (ya alag path do)
# ──────────────────────────────────────────────────────────────────

CLASSES = ["with_helmet", "without_helmet"]
SPLITS  = {"train": 0.70, "valid": 0.20, "test": 0.10}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)

def split_dataset():
    print("=" * 50)
    print("   Helmet Dataset Split Script")
    print("=" * 50)

    # Step 1 — output folders banao
    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(folder, exist_ok=True)

    total_copied = 0

    for cls in CLASSES:
        src_folder = os.path.join(BASE_DIR, cls)

        if not os.path.exists(src_folder):
            print(f"\n❌ Folder nahi mila: {src_folder}")
            print(f"   Check karo ke path sahi hai!")
            continue

        # sirf images lo
        images = [
            f for f in os.listdir(src_folder)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]

        if not images:
            print(f"\n⚠️  {cls} mein koi image nahi mili!")
            continue

        random.shuffle(images)
        total = len(images)

        n_train = int(total * SPLITS["train"])
        n_valid = int(total * SPLITS["valid"])
        # baqi sab test mein
        n_test  = total - n_train - n_valid

        split_data = {
            "train": images[:n_train],
            "valid": images[n_train:n_train + n_valid],
            "test":  images[n_train + n_valid:]
        }

        print(f"\n📁 Class: {cls}")
        print(f"   Total  : {total}")
        print(f"   Train  : {len(split_data['train'])}")
        print(f"   Valid  : {len(split_data['valid'])}")
        print(f"   Test   : {len(split_data['test'])}")

        for split, files in split_data.items():
            for fname in files:
                src  = os.path.join(src_folder, fname)
                dst  = os.path.join(OUTPUT_DIR, split, cls, fname)
                shutil.copy2(src, dst)
                total_copied += 1

    print("\n" + "=" * 50)
    print(f"✅ Done! Total {total_copied} images copy ho gayi.")
    print(f"\nFolder structure ab aisa hai:")
    print(f"  {OUTPUT_DIR}/")
    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(OUTPUT_DIR, split, cls)
            count  = len(os.listdir(folder)) if os.path.exists(folder) else 0
            print(f"    {split}/{cls}/  →  {count} images")
    print("=" * 50)
    print("\nAb train_models.py chalao! 🚀")

if __name__ == "__main__":
    split_dataset()