import os
import shutil
import random
import kagglehub

# Download dataset
path = kagglehub.dataset_download("emmarex/plantdisease")

# IMPORTANT: actual dataset location
source_dir = os.path.join(path, "PlantVillage")

train_dir = "dataset/train"
test_dir = "dataset/test"

split_ratio = 0.8

classes = {
    "healthy": [
        "Pepper__bell___healthy",
        "Potato___healthy",
        "Tomato_healthy"
    ],

    "early_blight": [
        "Potato___Early_blight",
        "Tomato_Early_blight"
    ],

    "late_blight": [
        "Potato___Late_blight",
        "Tomato_Late_blight"
    ],

    "mosaic_virus": [
        "Tomato__Tomato_mosaic_virus"
    ]
}

for disease, folders in classes.items():

    images = []

    for folder in folders:

        folder_path = os.path.join(source_dir, folder)

        if not os.path.exists(folder_path):
            print("Folder not found:", folder_path)
            continue

        for img in os.listdir(folder_path):
            images.append(os.path.join(folder_path, img))

    random.shuffle(images)

    split = int(len(images) * split_ratio)

    train = images[:split]
    test = images[split:]

    os.makedirs(os.path.join(train_dir, disease), exist_ok=True)
    os.makedirs(os.path.join(test_dir, disease), exist_ok=True)

    for img in train:
        shutil.copy(img, os.path.join(train_dir, disease))

    for img in test:
        shutil.copy(img, os.path.join(test_dir, disease))

    print(disease, "Train:", len(train), "Test:", len(test))

print("\nDataset prepared successfully!")