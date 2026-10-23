"""Model loading, image preprocessing and prediction for hand-sign digits."""
from functools import lru_cache
from pathlib import Path

from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / "digit_model.h5"


@lru_cache(maxsize=1)
def get_model():
    """Load the trained Keras model once and reuse it for later predictions."""
    return load_model(MODEL_PATH)


def preprocess_img(img_path):
    """Load an image and convert it into the model's input format.

    Args:
        img_path: A file path or file-like object containing the image.

    Returns:
        A float array of shape (1, 224, 224, 3) with values scaled to [0, 1].
    """
    # The model expects 3 colour channels; grayscale (L), transparent (RGBA)
    # and palette (P) images would otherwise fail to reshape.
    op_img = Image.open(img_path).convert("RGB")
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


def predict_result(predict):
    """Run the model on a preprocessed image and return the predicted digit (0-9)."""
    pred = get_model().predict(predict)
    return np.argmax(pred[0], axis=-1)
