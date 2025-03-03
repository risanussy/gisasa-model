from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf  # meski namanya sama, ini adalah tensorflow-cpu
import numpy as np
import base64
from PIL import Image
import io

model = tf.keras.models.load_model('final_model.h5')  # tetap sama
CLASS_NAMES = ['baik', 'rusak_ringan', 'rusak_sedang', 'rusak_berat']

app = Flask(__name__)
CORS(app)

@app.route('/predict-base64', methods=['POST'])
def predict_base64():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No base64 image provided"}), 400

        base64_image = data['image']
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]

        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))

        image = image.resize((150, 150))
        image_array = np.array(image, dtype='float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        max_index = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[max_index]
        return jsonify({"predictedClass": predicted_class})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": "Prediction error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
