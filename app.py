import os
from flask import Flask, render_template, request
from predict import generate_caption

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            # Save the uploaded file
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Generate caption
            caption = generate_caption(filepath)
            image_url = filepath  # for displaying in browser

    return render_template("index.html", caption=caption, image_url=image_url)

if __name__ == "__main__":
    print("✅ Flask server started at http://127.0.0.1:5000")
    app.run(debug=True)
