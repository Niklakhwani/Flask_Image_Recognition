# Hand Digit Image Recognition

[![CI](https://github.com/Niklakhwani/Flask_Image_Recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/Niklakhwani/Flask_Image_Recognition/actions/workflows/ci.yml)

This web app is made using Flask framework and is all about predicting the hand sign digit images based on the deep learning model trained using transfer learning.

![Python](https://img.shields.io/badge/python-8338ec)
![Keras](https://img.shields.io/badge/keras-219ebc)
![Numpy](https://img.shields.io/badge/numpy-ffafcc)
![Pillow](https://img.shields.io/badge/pillow-00bbf9)
![Flask](https://img.shields.io/badge/flask-f72585)
![Bootstrap](https://img.shields.io/badge/bootstrap-57cc99)

# Glimpse

https://user-images.githubusercontent.com/72191416/201943098-c8f5fd8b-ec7d-4e5d-883d-8b69109b946f.mp4

# Dependencies

- Flask
- TensorFlow (Keras)
- Numpy
- Pillow (PIL)

## UI Framework

- Bootstrap

# Get Started

Clone the repository in your project directory.

```commandline
git clone https://github.com/Niklakhwani/Flask_Image_Recognition.git
```
OR

Download as **[ZIP](https://github.com/Sachin-crypto/Flask_Image_Recognition/archive/refs/heads/main.zip)** file and extract it in your project directory.

# Install Dependencies

Requires **Python 3.11** (TensorFlow 2.15 supports Python 3.9-3.11 on macOS Apple Silicon, Linux and Windows).

```commandline
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

# Run

**cd** into the current directory and run the script.

```commandline
flask --app app.py run --debug
```
or `python app.py` to start it on http://127.0.0.1:9000.

# Test

- Choose an image from the test images folder.
- You will see a preview of the uploaded image.
- Click on **Submit** button and see the magic.

# Development

Install the development tools (Pylint, pytest, pre-commit) and the Git hooks:

```commandline
pip install -r requirements-dev.txt
pre-commit install
```

| Command | What it does |
|---|---|
| `pylint $(git ls-files '*.py')` | Lint all Python files using `.pylintrc` (fails below 9.0) |
| `pytest` | Run the test suite in `tests/` |
| `pre-commit run --all-files` | Run every pre-commit hook on the whole repository |

Every push and pull request to `main` runs the same Pylint and pytest checks in
GitHub Actions (`.github/workflows/ci.yml`).
