import os
import kagglehub

path = kagglehub.dataset_download("emmarex/plantdisease")
source_dir = os.path.join(path, "PlantVillage")

print("Dataset path:", source_dir)
print("\nFolders inside dataset:\n")

for folder in os.listdir(source_dir):
    print(folder)