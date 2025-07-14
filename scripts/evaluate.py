import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

val_dir = os.path.join('..', 'data', 'validation')

class_names = sorted(os.listdir(val_dir))
class_indices = {name: idx for idx, name in enumerate(class_names)}
inv_class_indices = {v: k for k, v in class_indices.items()}

model = load_model('household_classifier.h5')

def predict(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    x = image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    label = inv_class_indices[idx]
    prob = preds[idx]
    return label, prob

for label in class_names:
    print(f'\n=== {label.upper()} ===')
    folder = os.path.join(val_dir, label)
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in images[:3]:
        path = os.path.join(folder, img_name)
        lbl, p = predict(path)
        print(f'{img_name}: predicted → {lbl} ({p:.2f})')
