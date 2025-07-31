import os
import numpy as np
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Paths
IMAGE_FOLDER = "data/Flickr8k_Dataset"
CAPTION_FILE = "data/Flickr8k_text/Flickr8k.token.txt"
FEATURES_PATH = "data/features.pkl"
CAPTIONS_PATH = "data/captions.pkl"

# Load InceptionV3 model for feature extraction (only if we run as main)
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def preprocess_image(image_path):
    """
    Preprocess a single image to feed into InceptionV3.
    """
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

if __name__ == "__main__":
    # Only run this if we execute: python preprocess.py
    features = {}
    print("🔹 Extracting features from images...")
    for i, img_name in enumerate(os.listdir(IMAGE_FOLDER)):
        if img_name.lower().endswith(".jpg"):
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            img = preprocess_image(img_path)
            feature = model.predict(img, verbose=0)
            features[img_name] = feature.flatten()
            
            if (i+1) % 500 == 0:
                print(f"Processed {i+1} images...")

    # Save extracted features
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(features, f)
    print(f" Saved image features to {FEATURES_PATH}")

    # 2. Load captions
    print("🔹 Processing captions...")
    captions = {}

    with open(CAPTION_FILE, 'r') as f:
        lines = f.readlines()

    image_files = [img for img in os.listdir(IMAGE_FOLDER) if img.lower().endswith(".jpg")]
    image_files.sort()  # Ensure consistent order

    if len(lines) != len(image_files):
        print(f"⚠ Warning: {len(lines)} captions for {len(image_files)} images. Will map sequentially until min length.")

    for img_name, caption in zip(image_files, lines):
        caption = caption.strip()
        captions[img_name] = ["startseq " + caption + " endseq"]

    # Save captions
    with open(CAPTIONS_PATH, 'wb') as f:
        pickle.dump(captions, f)

    print(f"✅ Saved captions to {CAPTIONS_PATH}")
    print("🎉 Preprocessing complete!")
