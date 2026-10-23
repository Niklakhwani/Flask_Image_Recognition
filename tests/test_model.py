"""Tests for image preprocessing and prediction in model.py."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import UnidentifiedImageError

import model
from tests.conftest import SAMPLE_IMAGE, PROJECT_ROOT, make_image_bytes


# ---------- preprocess_img ----------

def test_preprocess_returns_model_input_shape():
    """A dataset image becomes a single batch of 224x224 RGB pixels."""
    result = model.preprocess_img(SAMPLE_IMAGE)
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_scales_pixels_to_unit_range():
    """Pixel values are normalised to floats between 0 and 1."""
    result = model.preprocess_img(SAMPLE_IMAGE)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_accepts_file_like_object():
    """Uploads arrive as streams, so file-like objects must work too."""
    result = model.preprocess_img(make_image_bytes(size=(50, 80)))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_resizes_non_square_images():
    """Images of any size are resized to the model's 224x224 input."""
    result = model.preprocess_img(make_image_bytes(size=(640, 120), fmt="JPEG"))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_scales_known_colour():
    """A solid image of value 255 becomes all ones after scaling."""
    result = model.preprocess_img(make_image_bytes(color=(255, 255, 255)))
    assert np.allclose(result, 1.0)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"], ids=["grayscale", "rgba", "palette"])
def test_preprocess_handles_non_rgb_images(mode):
    """Grayscale, transparent and palette images are converted to 3 channels."""
    result = model.preprocess_img(make_image_bytes(mode=mode))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image_file(tmp_path):
    """A text file cannot be opened as an image."""
    text_file = tmp_path / "not_an_image.txt"
    text_file.write_text("hello")
    with pytest.raises(UnidentifiedImageError):
        model.preprocess_img(text_file)


# ---------- predict_result (mocked model) ----------

def test_predict_result_returns_index_of_highest_probability():
    """predict_result returns the class with the highest score."""
    fake_model = MagicMock()
    scores = np.zeros((1, 10))
    scores[0, 7] = 0.9
    fake_model.predict.return_value = scores

    with patch.object(model, "get_model", return_value=fake_model):
        assert model.predict_result(np.zeros((1, 224, 224, 3))) == 7

    fake_model.predict.assert_called_once()


def test_predict_result_with_tied_scores_picks_first():
    """Edge case: equal scores resolve to the lowest digit (numpy argmax)."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.full((1, 10), 0.1)

    with patch.object(model, "get_model", return_value=fake_model):
        assert model.predict_result(np.zeros((1, 224, 224, 3))) == 0


def test_get_model_loads_once_and_caches():
    """The model file is loaded only once, then reused."""
    model.get_model.cache_clear()
    with patch.object(model, "load_model", return_value="loaded") as mock_load:
        assert model.get_model() == "loaded"
        assert model.get_model() == "loaded"
    mock_load.assert_called_once_with(model.MODEL_PATH)
    model.get_model.cache_clear()


# ---------- Real model (integration) ----------

@pytest.mark.integration
@pytest.mark.parametrize("digit", [0, 3, 8])
def test_real_model_predicts_dataset_images(digit):
    """End to end with the real model: dataset images are classified correctly."""
    image_path = next((PROJECT_ROOT / "test_images" / str(digit)).glob("*.jpeg"))
    assert model.predict_result(model.preprocess_img(image_path)) == digit
