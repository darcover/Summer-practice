import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'household_classifier.h5')
model = load_model(model_path)

class_names = sorted(os.listdir(os.path.join('data', 'train')))
inv_class_indices = {i: name for i, name in enumerate(class_names)}

rus_labels = {
    'fridge':   'холодильник',
    'cupboard': 'шкаф',
    'tv':       'телевизор',
    'chair':    'стул'
}

def predict_img(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    label_eng = inv_class_indices[idx]
    confidence = preds[idx]
    return label_eng, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_url = None

    if request.method == 'POST':
        img = request.files.get('image')
        if img:
            upload_dir = os.path.join(app.root_path, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            upload_path = os.path.join(upload_dir, img.filename)
            img.save(upload_path)

            label_eng, conf = predict_img(upload_path)
            label_ru = rus_labels.get(label_eng, label_eng)
            result = f'Это: {label_ru} ({conf:.2%})'

            static_uploads = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(static_uploads, exist_ok=True)
            final_path = os.path.join(static_uploads, img.filename)
            os.replace(upload_path, final_path)
            img_url = url_for('static', filename=f'uploads/{img.filename}')

    return render_template('index.html', result=result, img_url=img_url)

if __name__ == '__main__':
    os.makedirs(os.path.join(app.root_path, 'uploads'), exist_ok=True)
    app.run(debug=True)
