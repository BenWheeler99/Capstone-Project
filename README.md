---
title: AI-Powered Book Recommendation System
emoji: "📚"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# AI-Powered Book Recommendation System

This project is a Gradio-powered web app that recommends books based on your description. It combines semantic search with FAISS and a fine-tuned T5 model to produce a concise recommendation from similar books in the corpus.

## Highlights

- hybrid recommendation pipeline with retrieval and generation
- local model loading for offline inference
- clean Gradio UI with example prompts and top-match inspection
- repository-ready setup with explicit dependencies and ignore rules

## Requirements

- Python 3.9 (recommended)
- pip

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/Capstone-Project.git
   cd Capstone-Project
   ```

   If the dataset or model was committed with Git LFS, fetch the large files after cloning:
   ```sh
   git lfs pull
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
   If you need to install manually, use:
   ```sh
   pip install pandas sentence-transformers faiss-cpu transformers torch gradio sentencepiece
   ```

4. **Download or place your model and dataset:**
   - Place your fine-tuned T5 model files in `book-recommender-model-v3/`
   - Place your dataset CSV as `sampled_dataset_no_nulls_only_EN_NEW.csv` in the project root
   - Keep the file names unchanged so the app can load them automatically

## Project Structure

- `app.py`: the real runtime entrypoint
- `copy_of_gui.py`: compatibility launcher for older workflows
- `book_system.py`: archived Colab notebook export used for training and data prep
- `book-recommender-model-v3/`: saved local model artifacts
- `sampled_dataset_no_nulls_only_EN_NEW.csv`: the book corpus used at inference time

## Running the App

```sh
python app.py
```

The app will open automatically in your browser at [http://127.0.0.1:7860](http://127.0.0.1:7860).

## Troubleshooting

- If you see errors about missing packages, install them with pip as shown above.
- If you see errors about missing model files, ensure your model directory contains `pytorch_model.bin` or `model.safetensors`.
- For Mac users: If you have issues installing `sentencepiece`, make sure you have `cmake` and `pkg-config` installed:
  ```sh
  brew install cmake pkg-config
  ```
  
```yaml
---
title: AI-Powered Book Recommendation System
emoji: "📚"
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---
```

**Note:**  
- Make sure your `requirements.txt` lists all dependencies.
- The app will run automatically when the Space is launched.








