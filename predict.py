import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess_image
from extract_features import extract_features

# Paths
MODEL_PATH = "image_caption_model_light.h5"  # lightweight trained model
TOKENIZER_PATH = "data/tokenizer.pkl"
MAX_LEN = 30  # same as in training

# Load model and tokenizer
print("🔹 Loading model and tokenizer...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)


def generate_caption(image_path):
    # Extract features from image
    feature = extract_features(image_path)
    feature = feature.reshape((1, 2048))

    # Generate sequence
    caption = '<start>'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_word[yhat]
        caption += ' ' + word
        if word == '<end>':
            break

    #  Clean the caption
    caption = caption.replace('<start>', '').replace('<end>', '').strip()

    # Remove file-like tokens accidentally learned from dataset
    import re
    caption = re.sub(r'\b\S+\.(jpg|jpeg|png)\b', '', caption).strip()

    return caption

