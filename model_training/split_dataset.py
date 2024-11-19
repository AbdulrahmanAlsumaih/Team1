import os
import shutil
from sklearn.model_selection import train_test_split

# Set paths
dataset_dir = './train'
train_dir = './my_train_dataset/train_split'
val_dir = './my_train_dataset/val_split'
test_dir = './my_train_dataset/test_split'

# Create directories if they don't exist
for dir_path in [train_dir, val_dir, test_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Get all image directories
image_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

# Split into train, val, test (e.g., 70% train, 15% val, 15% test)
train_ids, temp_ids = train_test_split(image_dirs, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Helper function to move directories
def move_dirs(image_ids, dest_dir):
    for image_id in image_ids:
        src_path = os.path.join(dataset_dir, image_id)
        dest_path = os.path.join(dest_dir, image_id)
        shutil.move(src_path, dest_path)

# Move directories to train, val, and test folders
move_dirs(train_ids, train_dir)
move_dirs(val_ids, val_dir)
move_dirs(test_ids, test_dir)

print(f"Moved {len(train_ids)} directories to {train_dir}")
print(f"Moved {len(val_ids)} directories to {val_dir}")
print(f"Moved {len(test_ids)} directories to {test_dir}")
