import os
import random
import shutil

src_dir = os.path.join('..', 'data', 'train')
val_dir = os.path.join('..', 'data', 'validation')
val_ratio = 0.3  

for label in os.listdir(src_dir):
    src_label_dir = os.path.join(src_dir, label)
    dest_label_dir = os.path.join(val_dir, label)
    os.makedirs(dest_label_dir, exist_ok=True)
    files = [
        f for f in os.listdir(src_label_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    random.shuffle(files)
    n_val = int(len(files) * val_ratio)
    for f in files[:n_val]:
        shutil.move(
            os.path.join(src_label_dir, f),
            os.path.join(dest_label_dir, f)
        )
    print(f'label "{label}": moved {n_val} files to validation')
