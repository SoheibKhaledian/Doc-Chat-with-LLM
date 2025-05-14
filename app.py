import os
import gradio as gr
import hashlib
import pickle
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.schema import Document
from llama_parse import LlamaParse

# API Keys
llama_cloud_key = os.environ.get("LLAMA_CLOUD_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")
if not (llama_cloud_key and groq_key):
    raise ValueError("API Keys not found! Ensure they are passed to the Docker container.")

# LLM and Embedding model setup
llm = Groq(model="llama3-70b-8192", api_key=groq_key)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    cache_folder="./models"
)
parser = LlamaParse(api_key=llama_cloud_key, result_type="markdown")

# File format extractor
file_extractor = {
    ".pdf": parser, ".docx": parser, ".doc": parser, ".txt": parser, ".csv": parser,
    ".xlsx": parser, ".pptx": parser, ".html": parser, ".jpg": parser,
    ".jpeg": parser, ".png": parser, ".webp": parser, ".svg": parser,
}

# Global vector index
vector_index = None

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_cache_path(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return os.path.join("./models", f"{hash_md5.hexdigest()}.pkl")

def load_files(file_path: str):
    global vector_index
    if not file_path:
        return "No file provided. Please upload a file."
    
    valid_extensions = ', '.join(file_extractor.keys())
    if not any(file_path.endswith(ext) for ext in file_extractor):
        return f"Invalid file type. Supported types: {valid_extensions}"

    cache_path = get_cache_path(file_path)
    filename = os.path.basename(file_path)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            vector_index = pickle.load(f)
        print(f"Loaded cached index for: {filename}")
        return f"✅ Ready to chat based on: {filename} (cached)"

    document = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
    all_chunks = []
    for doc in document:
        if hasattr(doc, "text"):
            chunks = chunk_text(doc.text)
            all_chunks.extend(chunks)
        else:
            all_chunks.append(doc)


    print(f"Embedding {len(all_chunks)} chunks...")
    vector_index = VectorStoreIndex.from_documents(
        [Document(text=chunk) for chunk in all_chunks],
        embed_model=embed_model
    )

    with open(cache_path, "wb") as f:
        pickle.dump(vector_index, f)

    print(f"File loaded: {filename}")
    return f"✅ Ready to chat based on: {filename}"

def build_prompt(user_message):
    return (
        "لطفاً به سؤال زیر با جزئیات کامل، ساختارمند و زیبا (در صورت امکان با فرمت Markdown) پاسخ بده. "
        "اگر لازم است، مثال بزن و نکات کلیدی را بولت‌پوینت کن. به فارسی جواب بده\n\n"
        f"سؤال: {user_message}\n\n"
    )

def respond(message, history):
    try:
        prompt = build_prompt(message)
        query_engine = vector_index.as_query_engine(streaming=True, llm=llm)
        streaming_response = query_engine.query(prompt)
        partial_text = ""
        for new_text in streaming_response.response_gen:
            partial_text += new_text
            yield partial_text
    except Exception as e:
        print(f"Error during query: {e}")
        yield "❌ Please upload a file first."

def clear_state():
    global vector_index
    vector_index = None
    return [None, None, None]

with gr.Blocks(
    theme=gr.themes.Default(primary_hue="green", secondary_hue="blue", font=[gr.themes.GoogleFont("Poppins")]),
    css="footer {visibility: hidden}"
) as demo:
    gr.Markdown("# 🤖📄 Doc Chat with LlamaIndex")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_count="single", type="filepath", label="📁 Upload your document")
            with gr.Row():
                btn = gr.Button("Submit", variant="primary")
                clear = gr.Button("Clear")
            output = gr.Textbox(label="Status")
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(height=300, type="messages"),
                theme="soft",
                show_progress="full",
                textbox=gr.Textbox(
                    placeholder="Ask questions about the uploaded document...",
                    container=False,
                ),
            )

    btn.click(fn=load_files, inputs=file_input, outputs=output)
    clear.click(fn=clear_state, outputs=[file_input, output])

if __name__ == "__main__":
    demo.launch()
