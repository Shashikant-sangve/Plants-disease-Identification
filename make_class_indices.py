# make_class_indices.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json, os

os.makedirs("model", exist_ok=True)
base = "dataset/train" if os.path.isdir("dataset/train") else "dataset"

gen = ImageDataGenerator().flow_from_directory(
    base, target_size=(224,224), batch_size=1, shuffle=False
)

with open("model/class_indices.json","w") as f:
    json.dump(gen.class_indices, f, indent=2)

print("Saved:", gen.class_indices)
