from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import os
import gdown
import tensorflow as tf

app = Flask(__name__)

# Path dan URL model
MODEL_PATH = "densenet_class5.keras"
MODEL_URL = "https://drive.google.com/uc?id=1OcPnl9MBhAYBuRy-7SIVmarTmd1b5f4T"

# Pastikan folder upload ada
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download model kalau belum ada
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Label kelas
class_names = ['Bukan Cabai', 'Kering', 'Matang', 'Mentah', 'Setengah Matang']

# Deskripsi tiap kelas
descriptions = {
    'Bukan Cabai': 'Gambar yang diunggah bukan cabai.',
    'Kering': 'Cabai telah mengering dan kehilangan kadar air.',
    'Matang': 'Cabai berwarna merah merata, siap untuk dipanen.',
    'Mentah': 'Cabai masih berwarna hijau atau belum matang sepenuhnya.',
    'Setengah Matang': 'Cabai berada di tahap transisi, sebagian merah dan sebagian masih hijau.'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None
    description = None

    if request.method == 'POST':
        file = request.files.get('file')  # Pakai get biar aman
        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = file.filename

            # Preprocessing gambar
            image = Image.open(filepath).convert("RGB")
            image = image.resize((256, 256))
            img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

            # Prediksi
            pred = model.predict(img_array)
            predicted_class = class_names[np.argmax(pred)]
            confidence = round(float(np.max(pred)) * 100, 2)  # Dibulatkan 2 desimal
            prediction = predicted_class
            description = descriptions.get(prediction, "Deskripsi tidak tersedia.")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        description=description
    )

if __name__ == '__main__':
    app.run(debug=True)
