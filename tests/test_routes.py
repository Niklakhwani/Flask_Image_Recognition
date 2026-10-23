"""Tests for the Flask routes in app.py.

The model's predict_result is mocked in these tests so they run fast and
do not depend on TensorFlow inference.
"""
import io
from unittest.mock import patch

import pytest

from tests.conftest import make_image_bytes

ERROR_MESSAGE = "File cannot be processed."


def post_file(client, data):
    """POST multipart form data to the prediction route."""
    return client.post("/prediction", data=data, content_type="multipart/form-data")


# ---------- Home page ----------

def test_home_page_returns_200(client):
    """The home page loads successfully."""
    response = client.get("/")
    assert response.status_code == 200


def test_home_page_contains_upload_form(client):
    """The home page renders the upload form that posts to /prediction."""
    html = client.get("/").get_data(as_text=True)
    assert "Hand Sign Digit Language Detection" in html
    assert 'action="/prediction"' in html
    assert 'name="file"' in html
    assert 'enctype="multipart/form-data"' in html


def test_static_css_is_served(client):
    """Static resources referenced by the templates are available."""
    response = client.get("/static/css/custom.css")
    assert response.status_code == 200


# ---------- Valid requests ----------

@patch("app.predict_result", return_value=3)
def test_valid_image_shows_prediction(mock_predict, client, sample_image_bytes):
    """A valid image upload returns 200 and shows the predicted digit."""
    response = post_file(client, {"file": (io.BytesIO(sample_image_bytes), "sign3.jpeg")})

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Prediction" in html
    assert ">3</h2>" in html
    assert ERROR_MESSAGE not in html
    mock_predict.assert_called_once()


@patch("app.predict_result", return_value=0)
def test_prediction_receives_preprocessed_array(mock_predict, client, sample_image_bytes):
    """The route passes a correctly shaped array to the model."""
    post_file(client, {"file": (io.BytesIO(sample_image_bytes), "sign3.jpeg")})

    (image_array,), _ = mock_predict.call_args
    assert image_array.shape == (1, 224, 224, 3)


@patch("app.predict_result", return_value=5)
def test_png_upload_is_accepted(_mock_predict, client):
    """A PNG image (not just JPEG) is accepted."""
    response = post_file(client, {"file": (make_image_bytes(fmt="PNG"), "hand.png")})
    assert response.status_code == 200
    assert ">5</h2>" in response.get_data(as_text=True)


# ---------- Missing file ----------

def test_missing_file_returns_400(client):
    """Submitting the form with no file field is a client error."""
    response = post_file(client, {})

    assert response.status_code == 400
    assert ERROR_MESSAGE in response.get_data(as_text=True)


def test_empty_file_selection_returns_400(client):
    """Submitting the form without choosing a file (empty filename) is rejected."""
    response = post_file(client, {"file": (io.BytesIO(b""), "")})

    assert response.status_code == 400
    assert ERROR_MESSAGE in response.get_data(as_text=True)


# ---------- Invalid or unexpected input ----------

@pytest.mark.parametrize(
    "content, filename",
    [
        (b"this is just text, not an image", "notes.txt"),
        (b"\x89PNG\r\n\x1a\n corrupted data", "broken.png"),
        (b"", "empty.jpeg"),
    ],
    ids=["text-file", "corrupted-png", "zero-byte-file"],
)
def test_invalid_file_returns_400(client, content, filename):
    """Non-image or corrupted uploads show the error message with HTTP 400."""
    response = post_file(client, {"file": (io.BytesIO(content), filename)})

    assert response.status_code == 400
    assert ERROR_MESSAGE in response.get_data(as_text=True)


def test_get_on_prediction_route_not_allowed(client):
    """The prediction route only accepts POST."""
    assert client.get("/prediction").status_code == 405


def test_unknown_route_returns_404(client):
    """Unknown URLs return 404."""
    assert client.get("/does-not-exist").status_code == 404
