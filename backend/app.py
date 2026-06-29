from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import traceback

# =========================
# FLASK APP
# =========================
app = Flask(__name__, 
            template_folder='../frontend', 
            static_folder='../frontend', 
            static_url_path='')

# =========================
# LOAD MODEL
# =========================
print("\nLoading Model...")

model = tf.keras.models.load_model(
    "brain_tumor_model_4cnn.keras"
)

print("Model Loaded Successfully!")

# =========================
# CLASS LABELS
# SAME ORDER AS DATASET
# =========================
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():

    return render_template('index.html')

# =========================
# PREDICTION ROUTE
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    try:

        print("\n===== PREDICT ROUTE CALLED =====")

        # =========================
        # CHECK IMAGE
        # =========================

        if 'image' not in request.files:

            return jsonify({
                "error": "No image uploaded"
            })

        file = request.files['image']

        if file.filename == '':

            return jsonify({
                "error": "No selected file"
            })

        # =========================
        # OPEN IMAGE
        # =========================

        print("Opening image...")

        img = Image.open(file)

        # RGB conversion
        img = img.convert("RGB")

        # SAME SIZE AS TRAINING
        img = img.resize((224, 224))

        # =========================
        # CONVERT TO ARRAY
        # =========================

        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        print("Image Shape:", img_array.shape)

        # =========================
        # MODEL PREDICTION
        # =========================

        print("Before prediction...")

        prediction = model(
            img_array,
            training=False
        ).numpy()

        print("After prediction!")

        print("Prediction:", prediction)

        # =========================
        # GET RESULT
        # =========================

        predicted_index = np.argmax(prediction)

        predicted_class = class_names[predicted_index]

        confidence = round(
            float(np.max(prediction)) * 100,
            2
        )

        print("Class:", predicted_class)

        print("Confidence:", confidence)

        # =========================
        # RETURN RESULT
        # =========================

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })

    except Exception as e:

        print("\n===== ERROR =====")

        traceback.print_exc()

        return jsonify({
            "error": str(e)
        })

# =========================
# RUN FLASK
# =========================
if __name__ == '__main__':

    app.run(
        debug=True,
        use_reloader=False
    )