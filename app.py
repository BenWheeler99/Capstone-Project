# -*- coding: utf-8 -*-
"""Interactive book recommendation app.

This module loads the local dataset and fine-tuned T5 model, then serves a
Gradio interface that combines semantic retrieval with text generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import faiss
import gradio as gr
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "sampled_dataset_no_nulls_only_EN_NEW.csv"
MODEL_PATH = BASE_DIR / "book-recommender-model-v3"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
MAX_GENERATION_TOKENS = 40
MAX_INPUT_TOKENS = 192


def load_books() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. If this repository uses Git LFS, run `git lfs pull`."
        )

    books = pd.read_csv(DATASET_PATH)
    required_columns = {"name", "summary"}
    missing_columns = required_columns.difference(books.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    books = books.dropna(subset=["name", "summary"]).copy()
    books["name"] = books["name"].astype(str).str.strip()
    books["summary"] = books["summary"].astype(str).str.strip()
    books = books[(books["name"] != "") & (books["summary"] != "")].reset_index(drop=True)
    return books


def load_retrieval_assets(books: pd.DataFrame) -> Tuple[SentenceTransformer, faiss.IndexFlatL2]:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(
        books["summary"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return embedding_model, index


def load_generator() -> Tuple[T5Tokenizer, T5ForConditionalGeneration, torch.device]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model directory not found at {MODEL_PATH}. Place the fine-tuned model folder in the repository root."
        )

    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


BOOKS = load_books()
EMBEDDING_MODEL, INDEX = load_retrieval_assets(BOOKS)
TOKENIZER, GENERATOR, DEVICE = load_generator()


def recommend_books(prompt: str, top_k: int) -> Tuple[str, pd.DataFrame]:
    prompt = (prompt or "").strip()
    if not prompt:
        return "Enter a description so the model has something to work with.", pd.DataFrame(columns=["name", "summary", "distance"])

    top_k = max(1, min(int(top_k), len(BOOKS)))

    query_embedding = EMBEDDING_MODEL.encode([prompt], convert_to_numpy=True).astype("float32")
    distances, indices = INDEX.search(query_embedding, top_k)

    recommendations = BOOKS.iloc[indices[0]].copy()
    recommendations["distance"] = distances[0]

    retrieved_titles = ", ".join(recommendations["name"].head(3).tolist())
    generator_prompt = (
        "You are helping a reader find a book. "
        f"The strongest matches are: {retrieved_titles}. "
        f"Reader request: {prompt}"
    )

    inputs = TOKENIZER(
        generator_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = GENERATOR.generate(
            **inputs,
            max_new_tokens=MAX_GENERATION_TOKENS,
            num_beams=4,
            do_sample=False,
        )

    recommendation = TOKENIZER.decode(output_ids[0], skip_special_tokens=True).strip()
    if not recommendation:
        recommendation = recommendations.iloc[0]["name"]

    return recommendation, recommendations[["name", "summary", "distance"]].reset_index(drop=True)


EXAMPLES = [
    ["A fast-paced mystery set in a small coastal town.", DEFAULT_TOP_K],
    ["A hopeful science fiction story about first contact.", DEFAULT_TOP_K],
    ["A reflective literary novel about grief, memory, and family.", DEFAULT_TOP_K],
]


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="amber", secondary_hue="teal", neutral_hue="stone"),
    title="AI Book Recommendation System",
) as demo:
    gr.Markdown(
        """
        # AI Book Recommendation System
        A hybrid retrieval + generation demo that searches a curated book corpus, then produces a concise recommendation based on the closest matches.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Describe the book you want",
                placeholder="Example: a cozy fantasy with strong world-building and found family",
                lines=4,
            )
            top_k = gr.Slider(1, 10, value=DEFAULT_TOP_K, step=1, label="How many matches to inspect")
            submit = gr.Button("Recommend books", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown(
                """
                **What this project shows**
                
                - semantic search with FAISS
                - a fine-tuned T5 generation step
                - a clean Gradio web interface
                - local model and dataset loading
                """
            )

    gr.Markdown("### Recommendation")
    recommendation = gr.Markdown()
    matches = gr.Dataframe(label="Top matches", interactive=False, wrap=True)

    gr.Examples(examples=EXAMPLES, inputs=[prompt, top_k])

    submit.click(fn=recommend_books, inputs=[prompt, top_k], outputs=[recommendation, matches])
    prompt.submit(fn=recommend_books, inputs=[prompt, top_k], outputs=[recommendation, matches])


def launch() -> None:
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    launch()
