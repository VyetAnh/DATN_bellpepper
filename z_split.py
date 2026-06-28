import os
import shutil
import random
random.seed(42)
def split_data(source_dir, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15):
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
            all_files = os.listdir(class_path)
            random.shuffle(all_files)
            train_count = int(len(all_files) * train_size)
            val_count = int(len(all_files) * val_size)
            train_files = all_files[:train_count]
            val_files = all_files[train_count:train_count + val_count]
            test_files = all_files[train_count + val_count:]
            for file in train_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_dir, class_name, file))
            for file in val_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(val_dir, class_name, file))
            for file in test_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(test_dir, class_name, file))
            print(
                f"{class_name}: "
                f"Train={len(train_files)}, "
                f"Val={len(val_files)}, "
                f"Test={len(test_files)}"
            )
source_dir = 'FullDataset/dataleaf'

train_dir = 'FullDataset/dataset_cnn/train'
val_dir   = 'FullDataset/dataset_cnn/val'
test_dir  = 'FullDataset/dataset_cnn/test'
'''
if os.path.exists("FullDataset/dataset_cnn"):
    shutil.rmtree("FullDataset/dataset_cnn")
    
''' 
split_data(source_dir, train_dir, val_dir, test_dir)
