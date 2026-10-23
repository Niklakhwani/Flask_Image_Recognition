"""Flask web app that predicts the digit shown in an uploaded hand-sign image."""
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Render the home page with the image upload form."""
    return render_template("index.html")


@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Preprocess the uploaded image, run the model and render the result page."""
    try:
        img = preprocess_img(request.files['file'].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))

    # KeyError: no "file" field in the form
    # OSError: Pillow cannot identify/read the upload (includes UnidentifiedImageError)
    # ValueError: image cannot be reshaped to the model's input shape
    except (KeyError, OSError, ValueError):
        error = "File cannot be processed."
        return render_template("result.html", err=error), 400


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
