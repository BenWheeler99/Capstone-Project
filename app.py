# -*- coding: utf-8 -*-
"""Command-line book recommendation app.

This module loads the local dataset and fine-tuned T5 model, then provides a
simple terminal interface that combines semantic retrieval with text generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "sampled_dataset_no_nulls_only_EN_NEW.csv"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


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


BOOKS = load_books()
EMBEDDING_MODEL, INDEX = load_retrieval_assets(BOOKS)


def recommend_books(prompt: str, top_k: int) -> Tuple[str, pd.DataFrame]:
    prompt = (prompt or "").strip()
    if not prompt:
        return "Enter a description so the model has something to work with.", pd.DataFrame(columns=["name", "summary", "distance"])

    top_k = max(1, min(int(top_k), len(BOOKS)))

    query_embedding = EMBEDDING_MODEL.encode([prompt], convert_to_numpy=True).astype("float32")
    distances, indices = INDEX.search(query_embedding, top_k)

    recommendations = BOOKS.iloc[indices[0]].copy()
    recommendations["distance"] = distances[0]

    recommendation = (
        f"Recommended book: {recommendations.iloc[0]['name']}\n"
        f"Why it fits: {recommendations.iloc[0]['summary']}"
    )

    return recommendation, recommendations[["name", "summary", "distance"]].reset_index(drop=True)


def format_matches(matches: pd.DataFrame) -> str:
    lines = []
    for index, row in matches.iterrows():
        lines.append(f"{index + 1}. {row['name']} (distance: {row['distance']:.4f})")
        lines.append(f"   {row['summary']}")
    return "\n".join(lines)


def run(prompt: str, top_k: int = DEFAULT_TOP_K) -> None:
    recommendation, matches = recommend_books(prompt, top_k)
    print("\nAI Recommendation:")
    print(recommendation)
    print("\nTop Matches:")
    print(format_matches(matches))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid book recommendation system")
    parser.add_argument("prompt", nargs="?", help="A description of the book you want")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many matches to inspect")
    return parser


def launch() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        prompt = input("Describe the book you want: ").strip()

    if not prompt:
        raise SystemExit("A description is required.")

    run(prompt, args.top_k)


if __name__ == "__main__":
    launch()
