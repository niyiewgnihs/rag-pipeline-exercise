"""
submission date: 23.09.2026 (wedn)
zoom call: 24.09.2026, 15:00-16:00 (thur)
TODO:
    1. Dokumente aus dem Ordner data/ einlesen: load_documents(data/) liest und gibt das gelesene zurück, verschiedene loader für pdf (PyPDFLoader) und doc (Docx2txtLoader)
    2. Inhalte sinnvoll in Chunks aufteilen: split_into_chunks(documents) mit RecursiveCharacterTextSplitter
    3. Die Chunks per Embeddings in einer Vektordatenbank speichern mit FastEmbedEmbeddings und Chroma
    4. Bei einer Frage relevante Chunks abrufen
    5. Die gefundenen Inhalte zusammen mit einem LLM zur Antwortgenerierung verwenden
"""
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

model = ChatOpenRouter(
    model="openrouter/free",
)

def load_documents(folder="data"):
    """
    reads all pdf and docx files from the given folder and returns a list
    of langchain document objects. each file type needs its own loader,
    which is picked based on the file extension.
    """
    documents = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith("docx"):
            loader = Docx2txtLoader(path)
        else:
            continue    # skip unsupported file types
        documents.extend(loader.load())
    return documents

def split_into_chunks(documents):
    """
    splits documents into smaller, overlapping chunks.
    RecursiveCharacterTextSplitter prefers to break at paragraph/sentence
    boundaries instead of cutting mid-word.
    chunk_size = 800: reasonable middle ground for texts in this dataset,
    which are pretty technical.
    chunk_overlap = 100: prevents sentences from being cut off at chunk
    boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 100,
    )
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    """
    converts each chunk into an embedding vector and stores it in a local
    chroma vector store.
    model choice: intfloat/multilingual-eg-large via fastembed (ONNX-based,
    no PyTorch required). Multilingual because the source documents are in
    english but questions are asked in german.
    """
    embeddings = FastEmbedEmbeddings(
            model_name = "intfloat/multilingual-e5-large"
    )
    return Chroma.from_documents(chunks, embeddings)

def main():
    docs = load_documents()
    chunks = split_into_chunks(docs)
    print(f"{len(docs)} document pages loaded.")
    print(f"{len(chunks)} chunks created.")
    vector_store = build_vector_store(chunks)
    print("Vector store built successfully")
    """
    messages = [
        (
            "system",
            "Du bist ein hilfreicher Assistent. Antworte auf Deutsch.",
        ),
        ("human", "Was ist ein RAG-System? Erkläre es in einem Satz."),
    ]
    ai_msg = model.invoke(messages)
    print(ai_msg.content)
    """


if __name__ == "__main__":
    main()
