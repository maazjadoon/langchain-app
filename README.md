# 🧠 DocMind — LangChain Document Q&A App

A simple full-stack app that lets you paste any document and ask questions about it using LangChain's RAG (Retrieval-Augmented Generation) pipeline.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Framework | LangChain |
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | FAISS (in-memory) |
| Backend | Flask (Python) |
| Frontend | Vanilla HTML/CSS/JS |

## How It Works

```
User Pastes Text
      │
      ▼
Text Splitter (RecursiveCharacterTextSplitter)
      │  chunks of 500 tokens, 50 overlap
      ▼
OpenAI Embeddings
      │  converts each chunk to a vector
      ▼
FAISS Vector Store
      │  stores all chunk vectors in memory
      ▼
User Asks Question
      │
      ▼
Similarity Search (top 3 relevant chunks retrieved)
      │
      ▼
RetrievalQA Chain (stuffs chunks into prompt)
      │
      ▼
GPT-3.5-turbo generates answer
      │
      ▼
Answer shown in chat UI
```

## Setup

1. **Clone / download this project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your OpenAI API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your key
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Open your browser** at `http://localhost:5000`

## Usage

1. Paste any text into the left panel (article, report, book chapter, etc.)
2. Click **Index Document** — this embeds and stores your text
3. Type a question in the chat box and press Enter
4. Get an AI-powered answer based only on your document!

## Key LangChain Components Used

- `RecursiveCharacterTextSplitter` — splits text into overlapping chunks
- `OpenAIEmbeddings` — creates vector representations of text
- `FAISS` — fast vector similarity search
- `RetrievalQA` — the full RAG chain
- `PromptTemplate` — custom prompt to keep answers grounded in context
- `ChatOpenAI` — the LLM that generates final answers

## Project Structure

```
langchain-qa/
├── app.py              # Flask backend + LangChain logic
├── templates/
│   └── index.html      # Chat UI
├── requirements.txt
├── .env.example
└── README.md
```
