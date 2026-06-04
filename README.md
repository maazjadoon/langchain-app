# 📄 LangChain Document Q&A App

> A fully **local, privacy-first** Retrieval-Augmented Generation (RAG) application that lets you paste any document and ask AI-powered questions — with **zero API keys** and zero cloud dependency.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey?style=flat-square&logo=flask)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)

---

## 🚀 Features

- **📝 Document Q&A** — Paste any text document and ask natural language questions
- **🧠 RAG Pipeline** — LangChain's modern LCEL (LangChain Expression Language) chain
- **⚡ FAISS Vector Store** — Fast in-memory semantic search over your documents
- **🦙 Local LLM** — Powered by `llama3.2` via Ollama — runs entirely on your machine
- **🔒 100% Private** — No data leaves your machine; no API keys required
- **🐳 Docker Support** — One-command deployment via Docker Compose

---

## 🏗️ Architecture

```
User Input (Document + Question)
        │
        ▼
Text Splitter (RecursiveCharacterTextSplitter)
        │
        ▼
FAISS Vector Store ◄─── nomic-embed-text (Ollama Embeddings)
        │
        ▼
RAG Chain (LCEL) ◄─── llama3.2 (Ollama LLM)
        │
        ▼
   Answer (Flask REST API → Frontend)
```

| Component | Technology |
|---|---|
| LLM | `llama3.2` via Ollama |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector Store | FAISS (in-memory) |
| Backend | Flask REST API |
| Chain Style | LCEL (LangChain 1.x) |
| Containerization | Docker + Docker Compose |

---

## 🛠️ Setup & Installation

### Prerequisites
- [Python 3.10+](https://www.python.org/)
- [Ollama](https://ollama.com/) installed and running locally
- Required Ollama models pulled:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

### Option A: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/maazjadoon/langchain-app.git
cd langchain-app

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env to set your Ollama base URL if needed

# 5. Run the app
python app.py
```

Open your browser at `http://localhost:5000`

### Option B: Run with Docker

```bash
docker compose up --build
```

---

## ⚙️ Configuration (`.env`)

```env
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Upload document text for indexing |
| `POST` | `/ask` | Ask a question against the indexed document |
| `GET` | `/health` | Health check |

---

## 🧠 How It Works

1. **Ingest** — Your document is split into overlapping chunks using `RecursiveCharacterTextSplitter`
2. **Embed** — Each chunk is converted to a vector using `nomic-embed-text` via Ollama
3. **Store** — Vectors are stored in a FAISS in-memory index for fast similarity search
4. **Query** — Your question is embedded and the most relevant chunks are retrieved
5. **Generate** — `llama3.2` generates an answer strictly grounded in the retrieved context

---

## 📦 Project Structure

```
langchain-app/
├── app.py              # Flask app + RAG chain (LCEL)
├── list_models.py      # Utility to list available Ollama models
├── templates/
│   └── index.html      # Frontend UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 👤 Author

**Muhammad Maaz Jadoon**
- GitHub: [@maazjadoon](https://github.com/maazjadoon)
- Email: mohazjadoon@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
