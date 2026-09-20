from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
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


def main():
    docs = load_documents()
    print(f"{len(docs)} document pages loaded.")
    messages = [
        (
            "system",
            "Du bist ein hilfreicher Assistent. Antworte auf Deutsch.",
        ),
        ("human", "Was ist ein RAG-System? Erkläre es in einem Satz."),
    ]
    ai_msg = model.invoke(messages)
    print(ai_msg.content)


if __name__ == "__main__":
    main()
