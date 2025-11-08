from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import io

# ------------------ CONFIG ------------------
app = Flask(__name__)

MODEL_PATH = 'my_cnn_model.h5'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
cnn_loaded = load_model(MODEL_PATH)

# Example class labels — change to your dataset’s classes
class_labels = ['CORROSION', 'NOCORROSION']
IMG_SIZE = (128, 128)
# --------------------------------------------

@app.route('/')
def home():
    return "✅ CNN Model Flask API is running!"


# 🔹 1️⃣ Predict route
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint: /predict
    Method: POST
    Body: form-data with key 'file' → image file
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    preds = cnn_loaded.predict(img_array)
    pred_prob = float(preds[0][0])
    predicted_class = 1 if pred_prob > 0.5 else 0

    return jsonify({
        'prediction': class_labels[predicted_class],
        'confidence': f"{pred_prob:.2%}"
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # default for local, Render uses its own
    app.run(host='0.0.0.0', port=port)
