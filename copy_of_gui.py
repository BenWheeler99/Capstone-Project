# -*- coding: utf-8 -*-

# Load dataset
dataset = pd.read_csv("/content/drive/MyDrive/Final/sampled_dataset_no_nulls_only_EN_NEW.csv")

# https://faiss.ai/index.html This link is for FAISS documentation
# Load FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
index.add(embedding_model.encode(dataset["summary"].tolist(), convert_to_numpy=True))

# Load your fine-tuned LLM
model_path = "/content/drive/MyDrive/Final/book-recommender-model-v3"  # Update with your saved model path
tokenizer = T5Tokenizer.from_pretrained(model_path) # this is the tokenizer that matches my t5-base model
model = T5ForConditionalGeneration.from_pretrained(model_path) # This allows for text generation from my t5-base model
device = "cuda" if torch.cuda.is_available() else "cpu" # This tells the colab to use GPU if possible. If not, use CPU
model.to(device)

# Book Recommendation Function
def recommend_books(prompt):
    # Step 1: Embed the input and search FAISS
    query_embedding = embedding_model.encode([prompt], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 5)  # Retrieve top 5 matches

    # Step 2: Retrieve books from dataset
    retrieved_books = dataset.iloc[indices[0]][["name", "summary"]]

    # Debugging: Ensure books are retrieved
    if retrieved_books.empty:
        return "⚠️ No books found. Try a different description!"

    summaries = retrieved_books["summary"].tolist()

    # Step 3: Generate a recommendation using the LLM
    input_text = f"Recommend a book based on these descriptions: {' '.join(summaries)}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)

    recommendation = tokenizer.decode(output[0], skip_special_tokens=True)

    # Step 4: Format the book list output
    book_list = "\n\n".join([f"📖 **{row['name']}**\n{row['summary']}" for _, row in retrieved_books.iterrows()])

    return f"🤖 **AI Recommendation:** {recommendation}\n\n🔎 **Top Matches:**\n\n{book_list}"

# Create a Gradio interface with a dedicated output box
interface = gr.Interface(
    fn=recommend_books,
    inputs=gr.Textbox(label="Enter a book description", placeholder="Describe the type of book you're looking for..."),
    outputs=gr.Textbox(label="Recommended Books", lines=10, interactive=False),  # Dedicated output box
    title="📚 AI-Powered Book Recommendation System",
    description="Enter a book description and get AI-generated recommendations based on similar books!",
)

# Launch the app
interface.launch()