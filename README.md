# CANTILEVER_task2e
#  Image Captioning 
 This project generates **descriptive captions for images** using a **CNN + LSTM Encoder-Decoder
 model**.  
The web app is built with **Flask**, allowing users to upload images and get AI-generated captions.--
##  Project Structure
 Image_captioning/
 app.py                  # Flask web app
 train.py                # Model training script
 preprocess.py           # Feature & caption preprocessing
 inference.py            # Inference script for testing
 predict.py              # Helper functions for predictions
 extract_features.py     # Extract CNN features from images
 data/                   # Dataset & processed files
    Flickr8k_Dataset/   # Flickr8k images
    Flickr8k_text/      # Flickr8k captions
    features.pkl        # Extracted image features
    captions.pkl        # Processed captions
 model/                  # Trained model(s)
    image_caption_model_light.h5
 static/                 # Uploaded images (for Flask)
 templates/              # HTML files for Flask UI--
##  Features- **CNN Encoder (InceptionV3):** Extracts image features- **LSTM Decoder:** Generates sequential captions- **Flask Web App:** Upload an image & get a caption instantly- **Lightweight Training:** Option to train on a subset for testing- **Full Training Ready:** Compatible with **Flickr8k (8,000 images)** on Colab--
##  How to Run Locally
 1 **Clone Repository**
 git clone https://github.com/disha125-luthra/CANTILEVER_task2.git
 cd CANTILEVER_task2
 2 **Create Virtual Environment**
python -m venv image_captioning_env
 image_captioning_env\Scripts\activate   # Windows
 # OR
 source image_captioning_env/bin/activate  # Mac/Linux
 3
 **Install Requirements**
 pip install -r requirements.txt
 4
 **Run Flask App**
 python app.py
 Visit `http://127.0.0.1:5000` to upload an image and get a caption.--
## 
 Training on Colab (Full 8K Images)- Upload the project folder to **Google Drive**- Open `train.py` in **Colab**- Ensure `preprocess.py` is run to generate `features.pkl` & `captions.pkl`- Train the model and download the `.h5` file to `model/` for inference--
## 
 Dependencies- Python 3.8+- TensorFlow / Keras- NumPy, Pickle- Flask (for web app)- Scikit-learn (for train-test split)
 Install all via:
 pip install -r requirements.txt--
## 
 License
 This project is for **educational & research purposes** under the Cantilever internship.--
## 
 Author
 **Disha Luthra**  
CANTILEVER Internship - Task 
