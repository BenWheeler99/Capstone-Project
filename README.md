---
title: AI-Powered Book Recommendation System
emoji: "📚"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: copy_of_gui.py
pinned: false
---

# AI-Powered Book Recommendation System

This project is a Gradio-powered web app that recommends books based on your description, using a fine-tuned T5 model and semantic search with FAISS.

## Requirements

- Python 3.9 (recommended)
- pip

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/Capstone-Project.git
   cd Capstone-Project
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python3.9 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   If you don’t have a `requirements.txt`, install manually:
   ```sh
   pip install pandas sentence-transformers faiss-cpu transformers torch gradio sentencepiece
   ```

4. **Download or place your model and dataset:**
   - Place your fine-tuned T5 model files in `book-recommender-model-v3/`
   - Place your dataset CSV as `sampled_dataset_no_nulls_only_EN_NEW.csv` in the project root

## Running the App

```sh
python copy_of_gui.py
```

The app will open automatically in your browser at [http://127.0.0.1:7860](http://127.0.0.1:7860).

## Troubleshooting

- If you see errors about missing packages, install them with pip as shown above.
- If you see errors about missing model files, ensure your model directory contains `pytorch_model.bin` or `model.safetensors`.
- For Mac users: If you have issues installing `sentencepiece`, make sure you have `cmake` and `pkg-config` installed:
  ```sh
  brew install cmake pkg-config
  ```

## Deploying on Hugging Face Spaces

To make your app public and runnable from a browser, you can use [Hugging Face Spaces](https://huggingface.co/spaces):

1. **Create a new Space** at [https://huggingface.co/spaces](https://huggingface.co/spaces) and select the **Gradio** SDK.
2. **Upload your files** (`copy_of_gui.py`, model folder, dataset, and `requirements.txt`).
3. **Configure your Space** with a `README.md` and a `README.md`-style YAML block at the top (see below).
4. **Rename your main file** to `app.py` or set `app_file: copy_of_gui.py` in the YAML block.

Example configuration block for Spaces (put at the top of your README or in a separate `README.md`):

```yaml
---
title: AI-Powered Book Recommendation System
emoji: "📚"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: copy_of_gui.py
pinned: false
---
```

**Note:**  
- Make sure your `requirements.txt` lists all dependencies.
- The app will run automatically when the Space is launched.

---

Let me know if you want this added directly to your README.md file!







