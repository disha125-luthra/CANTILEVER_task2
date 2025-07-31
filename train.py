import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import ModelCheckpoint

# -----------------------------
# 1. Load preprocessed data
# -----------------------------
print("🔹 Loading preprocessed features and captions...")

with open('data/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('data/captions.pkl', 'rb') as f:
    captions = pickle.load(f)

with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(caption.split()) for cap_list in captions.values() for caption in cap_list)

print(f"✅ Vocabulary Size: {vocab_size}")
print(f"✅ Max Caption Length: {max_len}")
print(f"✅ Total Images: {len(features)}")

# -----------------------------
# 2. Data generator to save RAM
# -----------------------------
def data_generator(captions, features, tokenizer, max_len, vocab_size, batch_size=64):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for img, caps in captions.items():
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    X1.append(features[img])
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1

                    if n == batch_size:
                        yield [np.array(X1), np.array(X2)], np.array(y)
                        X1, X2, y = list(), list(), list()
                        n = 0

# -----------------------------
# 3. Define the Captioning Model
# -----------------------------
embedding_dim = 256

# Feature extractor model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Sequence model
inputs2 = Input(shape=(max_len,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# Decoder (combine)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# -----------------------------
# 4. Train the model using generator
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 20

steps = sum(len(caps) for caps in captions.values()) // BATCH_SIZE
print(f"🔹 Steps per epoch: {steps}")

checkpoint = ModelCheckpoint('image_caption_model_full.h5', save_best_only=True, verbose=1)

model.fit(
    data_generator(captions, features, tokenizer, max_len, vocab_size, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    steps_per_epoch=steps,
    verbose=1,
    callbacks=[checkpoint]
)

print("🎉 Training complete! Model saved as image_caption_model_full.h5")
