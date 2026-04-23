# 📄 AI Document Assistant — RAG System (LangChain + Gemini)

An advanced **Retrieval-Augmented Generation (RAG)** based AI system that allows users to query information from documents such as **PDFs, CSVs, and text files** using natural language.

Built using **LangChain, Google Gemini API, and ChromaDB**, this project enables intelligent, context-aware question answering with conversational memory.

---

## 🚀 Features

* 📂 Upload multiple file types: **PDF, CSV, TXT**
* 🔍 Semantic search using **vector embeddings (ChromaDB)**
* 🤖 Context-aware responses using **Google Gemini LLM**
* 💬 Maintains **conversation history**
* ⚡ Configurable:

  * Chunk size
  * Chunk overlap
  * Top-K document retrieval
* 🧠 Multi-document querying with accurate retrieval

---

## 🏗️ Tech Stack

* **LLM:** Google Gemini (via LangChain)
* **Framework:** LangChain
* **Vector DB:** ChromaDB
* **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Frontend:** Streamlit
* **Backend:** Python

---

## 📂 Project Structure

```
├── rag_utils.py          # Core RAG pipeline (processing + QA)
├── ui.py                 # Streamlit UI
├── vector_store/         # ChromaDB storage
├── .env                  # API keys
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-document-assistant.git
cd ai-document-assistant
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run ui.py
```

---

## 🧪 How It Works

1. Upload documents (PDF/CSV/TXT)
2. Documents are:

   * Loaded and parsed
   * Split into chunks
   * Converted into embeddings
   * Stored in Chroma vector database
3. User asks a query
4. System retrieves relevant chunks
5. Gemini LLM generates a contextual answer

---

## 📸 Example Use Cases

* 📊 Query insights from CSV datasets
* 📄 Ask questions from research papers
* 📚 Summarize large text documents
* 🧾 Extract key information from reports

---


