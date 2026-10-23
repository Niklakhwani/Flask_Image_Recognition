"""Shared pytest fixtures for the Flask image recognition app."""
import io
from pathlib import Path

import pytest
from PIL import Image

from app import app as flask_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = PROJECT_ROOT / "test_images" / "3" / "Sign 3 (30).jpeg"


@pytest.fixture(name="client")
def client_fixture():
    """Flask test client with testing mode enabled."""
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.fixture(name="sample_image_bytes")
def sample_image_bytes_fixture():
    """Bytes of a real hand-sign image of the digit 3 from the dataset."""
    return SAMPLE_IMAGE.read_bytes()


def make_image_bytes(mode="RGB", size=(64, 64), fmt="PNG", color=128):
    """Create an in-memory image file in the given colour mode and format."""
    buffer = io.BytesIO()
    Image.new(mode, size, color=color).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer
