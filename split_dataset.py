import os
import shutil
import random

def split_dataset(input_dir, output_base_dir, train_ratio=0.8):
    classes = os.listdir(input_dir)
    for class_name in classes:
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        train_count = int(len(images) * train_ratio)
        train_images = images[:train_count]
        val_images = images[train_count:]

        train_dir = os.path.join(output_base_dir, 'train', class_name)
        val_dir = os.path.join(output_base_dir, 'val', class_name)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(train_dir, img))
        for img in val_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(val_dir, img))

        print(f"[✓] Split {class_name}: {len(train_images)} train / {len(val_images)} val")

if __name__ == "__main__":
    input_dir = 'dataset/full_data'
    output_base_dir = 'dataset'
    train_val_ratio = 0.8
    split_dataset(input_dir, output_base_dir, train_val_ratio)
