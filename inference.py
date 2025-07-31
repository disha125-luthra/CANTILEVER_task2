import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
MODEL_PATH = "image_caption_model_light.h5"
TOKENIZER_PATH = "data/tokenizer.pkl"

# Load model & tokenizer
print("🔹 Loading model and tokenizer...")
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = model.input_shape[1][1]  # sequence length from training

# Load InceptionV3 for feature extraction
base_model = InceptionV3(weights='imagenet')
cnn_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_features(image_path):
    img = preprocess_image(image_path)
    feature = cnn_model.predict(img, verbose=0)
    return feature

def generate_caption(image_path):
    feature = extract_features(image_path)
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = np.argmax(model.predict([feature, sequence], verbose=0))
        word = ''
        for w, idx in tokenizer.word_index.items():
            if idx == yhat:
                word = w
                break
        if word == '' or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()

# Test with an image
test_image = "data/Flickr8k_Dataset/1000268201_693b08cb0e.jpg"
caption = generate_caption(test_image)
print("Generated Caption:", caption)
