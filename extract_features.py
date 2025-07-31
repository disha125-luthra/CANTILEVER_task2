import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from preprocess import preprocess_image  # uses your existing function

# Load InceptionV3 model for feature extraction
print("🔹 Initializing InceptionV3 for feature extraction...")
base_model = InceptionV3(weights='imagenet')
feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(image_path: str):
    """
    Extract CNN features for a single image.
    """
    img = preprocess_image(image_path)        # Preprocess to 299x299
    feature = feature_model.predict(img, verbose=0)
    return feature
