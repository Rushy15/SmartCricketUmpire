import os
import random
import shutil

# === CONFIG ===
DATASET_DIR = "datasets/dataset2"  # Change this to your dataset folder name
SPLIT_RATIO = 0.8  # 80% train, 20% validation

# === Setup Paths ===
images_dir = os.path.join(DATASET_DIR, "images")
labels_dir = os.path.join(DATASET_DIR, "labels")

train_img_dir = os.path.join(images_dir, "train")
val_img_dir = os.path.join(images_dir, "val")
train_lbl_dir = os.path.join(labels_dir, "train")
val_lbl_dir = os.path.join(labels_dir, "val")

# === Create folders ===
for folder in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(folder, exist_ok=True)

# === Get all image files ===
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# === Shuffle and split ===
random.shuffle(image_files)
split_index = int(len(image_files) * SPLIT_RATIO)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# === Move files ===
def move_files(files, img_dst, lbl_dst):
    for img_file in files:
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"

        img_src_path = os.path.join(images_dir, img_file)
        lbl_src_path = os.path.join(labels_dir, label_file)

        img_dst_path = os.path.join(img_dst, img_file)
        lbl_dst_path = os.path.join(lbl_dst, label_file)

        if os.path.exists(lbl_src_path):
            shutil.copy(img_src_path, img_dst_path)
            shutil.copy(lbl_src_path, lbl_dst_path)
        else:
            print(f"Label file not found for {img_file}, skipping.")

# Move to respective folders
move_files(train_files, train_img_dir, train_lbl_dir)
move_files(val_files, val_img_dir, val_lbl_dir)

print(f"\n✅ Done! {len(train_files)} train and {len(val_files)} val images processed.")
