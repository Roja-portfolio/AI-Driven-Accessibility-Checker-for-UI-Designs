import os
import joblib
from PIL import Image
import numpy as np

def predict_ml_score(img_path):
    # Load the model from the same directory as this script
    model_path = os.path.join(os.path.dirname(__file__), "ml_model.pkl")
    model = joblib.load(model_path)

    # Preprocess the image
    image = Image.open(img_path).convert("L").resize((128, 128))
    img_array = np.array(image).flatten().reshape(1, -1)

    predicted_score = model.predict(img_array)[0]
    return round(float(predicted_score), 2)
