from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
import io
import cv2

# 1. Muat model Keras dari file .h5
model = tf.keras.models.load_model('final_model.h5')
print("Model loaded successfully")

# Daftar kelas prediksi
CLASS_NAMES = ['baik', 'rusak_ringan', 'rusak_sedang', 'rusak_berat']

app = Flask(__name__)
CORS(app)

@app.route('/predict-base64', methods=['POST'])
def predict_base64():
    """
    Menerima JSON: { "image": "data:image/jpeg;base64,..." }
    Mengembalikan: { "predictedClass": "<kelas>" }
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No base64 image provided"}), 400

        # Ambil string base64 dan hapus prefix jika ada
        base64_image = data['image']
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]

        # Decode base64 -> bytes
        image_bytes = base64.b64decode(base64_image)
        # Konversi bytes menjadi array numpy
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image menggunakan OpenCV (hasilnya dalam format BGR)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Konversi BGR ke RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize image sesuai input model (misal 150x150)
        img = cv2.resize(img, (150, 150))
        # Normalisasi pixel (0-1)
        image_array = img.astype('float32') / 255.0
        # Tambahkan dimensi batch -> (1,150,150,3)
        image_array = np.expand_dims(image_array, axis=0)

        # Prediksi
        prediction = model.predict(image_array)
        max_index = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[max_index]

        # Kembalikan hasil prediksi dalam format JSON
        return jsonify({"predictedClass": predicted_class})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": "Prediction error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
