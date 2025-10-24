import os
import json
import tempfile
from dotenv import load_dotenv

# --- LangChain imports (1.x) ---
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Environment setup ---
load_dotenv()
VECTOR_STORE_PATH = "vector_store"
HISTORY_FILE = os.path.join(VECTOR_STORE_PATH, "conversation_history.json")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# --- Embeddings ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- File processing ---
def process_files(files, chunk_size=1000, chunk_overlap=100):
    docs = []
    for file in files:
        file_ext = os.path.splitext(file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        # Select loader
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_ext == ".csv":
            loader = CSVLoader(tmp_path)
        elif file_ext == ".txt":
            loader = TextLoader(tmp_path)
        else:
            continue

        docs.extend(loader.load())

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # --- Store vectors ---
    Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_STORE_PATH)
    print("✅ Documents processed and stored successfully in Chroma DB.")


# --- Question Answering ---
def ask_question(query, k=3):
    vectordb = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # fallback to safe version
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    result = qa.run(query)

    # Save history
    sources = retriever.get_relevant_documents(query)
    log_result(query, result, sources)

    return result, [doc.metadata for doc in sources]


# --- Logging conversation history ---
def log_result(query, answer, sources):
    entry = {
        "query": query,
        "answer": answer,
        "sources": [doc.metadata for doc in sources]
    }

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

