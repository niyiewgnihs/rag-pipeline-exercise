"""
minimal rag pipeline.
loads .pdf/.docx files, chunks the text, creates embeddings, and uses a free LLM via OpenRouter to answer questions
based on the documents.
"""
import warnings
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    raise RuntimeError("Missing OPENROUTER_API_KEY. Add it to your .env file before running this script.")

model = ChatOpenRouter(
    model="openrouter/free",
    temperature=0,
)

def load_documents(folder="data"):
    """
    loads all supported files (.pdf & .docx) from the given folder and returns a list of langchain document objects
    along with the count of successfully loaded files.
    
    added try-except block so that if one file is corrupted or unreadable, the pipeline just skips it and continues
    with the rest instead of crashing the whole program.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise RuntimeError(f"Data folder '{folder}' not found.")

    documents = []
    file_count = 0

    for file_path in folder_path.iterdir():
        if file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            continue

        try:
            loaded = loader.load()
            print(f"    - {file_path.name} loaded.")
            documents.extend(loaded)
            file_count += 1
        except Exception as e:
            print(f"Warning: failed to load {file_path.name}, skipping. ({e})")
    
    if not documents:
        raise RuntimeError(f"No supported files (.pdf/.docx) found in '{folder}'.")

    return documents, file_count

def split_into_chunks(documents):
    """
    splits documents into smaller, overlapping chunks.

    WHY 800/100: ~150-200 words. reasonable middle ground for technical texts. not empirically tuned.
    would need a labeled eval set (q -> exp. chunk) to do that properly. out of scope.
    overlap: 12.5% of chunk size, enough to avoid sentences being cut off exactly at chunk boundary.
    """
    splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 100,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError("Splitting produced no chunks - check that the source documents contain extractble text.")

    # print("\n---------- example chunk ----------")
    # example_chunk = chunks[15] if len(chunks) > 15 else chunks[0]
    # print(f"source: {Path(example_chunk.metadata.get('source', '')).name}")
    # print(f"content: \n{example_chunk.page_content}")
    # print("------------------------------------")
   
    return chunks
    
def build_vector_store(chunks):
    """
    converts each chunk into an embedding vector and stores it in a local Chroma vector store.
    
    WHY multilingual: source documents are in english, example questions in the README are german. monolingual model
    would match german queries against english chunks worse.

    WHY -large: gave noticeably better retrieval than paraphrase-multilingual-MiniLM-L12-v2 in testing. -small exists
    but is missing in fastembed's supported model registry. (TextEmbedding.list_supported_models())
    
    WHY fastembed instead of sentence-transformers/PyTorch: runs on ONNX runtime, lightweight, doesn't need PyTorch.
    sentence-transformers requires PyTorch, which doesn't have a build for my (intel) mac anymore. pip install failed
    entirely, no matter what version I tried.

    FUTURE TODO: the vector store is created in-memory and rebuilt every run instead of being cached to disk. fine for
    demo. add persist_directory=... for production.
    """
    embeddings = FastEmbedEmbeddings(
            model_name = "intfloat/multilingual-e5-large"
            # model_name = "intfloat/multilingual-e5-small"
            # model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    store = Chroma.from_documents(chunks, embeddings)
    sample_vector = embeddings.embed_query(chunks[0].page_content)
    print(f"embedding dimension: {len(sample_vector)}")
    return store

def answer_question(vector_store, question: str, k: int = 4) -> str:
    """
    retrieves the k most relevant chunks for the question from the vector store and asks the LLM to generate an answer
    from them.

    WHY k=4: trade-off between enough context and drowning the model in irrelevant chunks ("lost in the middle" effect).
    not tuned against an eval set. default. same story as chunk_size.
    core point of RAG. "only based on the given context" forces the model to stay grounded in the retrieved chunks
    instead of falling back on its own (possibly outdated or wrong) knowledge.
    
    the free OpenRouter models can be overloaded (see README), so the LLM call is wrapped to fail gracefully instead of
    crashing the REPL loop in main(). also catches the weird moderation string "User Safety: safe".
    """
    results = vector_store.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in results)

    messages = [
        (
            "system",
            "You're a helpful assistant. Answer the question only based on the given context. "
            "If the context isn't sufficient to answer, say so honestly rather than guessing. "
            "Respond in the language of the question."
        ),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    
    try:
        answer = model.invoke(messages)
        if "User Safety: safe" in answer.content:
            return "[Error: the selected free-tier model has returned a faulty API response. Please try asking again.]"
    except Exception as e:
        return f"[Error: the model could not be reached ({e}). The free OpenRouter models can be overloaded. Try again in a moment.]"

    return answer.content   

def main() -> None:
    print("Loading documents...")
    docs, file_count = load_documents()
    print(f"{file_count} files loaded.")

    print("\nSplitting into chunks...")
    chunks = split_into_chunks(docs)
    print(f"{len(chunks)} chunks created.")
    
    print("\nBuilding vector store...")
    vector_store = build_vector_store(chunks)
    print("Vector store ready.")

    print("\nAsk your question (or type 'exit' to quit):") # Type 'debug db' to inspect vectors:
    while True:
        question = input("\n> ")
        if question.lower() == "exit":
            break
        
        # if question.lower() == "debug db":
        #    db_data = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
        #    print(f"\nTotal entries in Chroma DB: {len(db_data['ids'])}")
        #    print(f"Source of the first entry: {db_data['metadatas'][0]['source']}")
        #    print(f"Vector of the first entry (first 10 of 1024 dimensions):\n{db_data['embeddings'][0][:10]}")
        #    continue

        answer = answer_question(vector_store, question)
        print(f"\n{answer}")

if __name__ == "__main__":
    main()
