from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import os
from flask_cors import CORS 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load TensorFlow Lite model once (lazy loading)
TFLITE_MODEL_PATH = "my_cnn_model.tflite"
interpreter = None

def get_tflite_model():
    global interpreter
    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
    return interpreter

# Model input details
IMG_SIZE = (128, 128)
CLASS_LABELS = ['CORROSION', 'NOCORROSION']


@app.route('/')
def home():
    return "✅ CNN TFLite Model API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    form-data:
        key: 'file' -> image file
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        file.save(temp.name)
        temp_path = temp.name

    try:
        # Load image and preprocess
        img = Image.open(temp_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Load interpreter
        interpreter = get_tflite_model()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get prediction
        preds = interpreter.get_tensor(output_details[0]['index'])
        prob = float(preds[0][0])
        pred_class = 1 if prob > 0.5 else 0

        return jsonify({
            'prediction': CLASS_LABELS[pred_class],
            'confidence': f"{prob:.2%}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Always clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
