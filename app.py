"""
LangChain Document Q&A App  --  Powered by Ollama (100% local, free!)
- Paste any text document
- Ask questions and get AI-powered answers
- Uses LangChain's RAG pipeline with FAISS vector store
- LLM        : llama3.2 (via Ollama)
- Embeddings : nomic-embed-text (via Ollama)
- Chain style: LCEL (LangChain Expression Language, modern 1.x API)
- No API keys required!
"""

import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# -- LangChain 1.x imports ----------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = Flask(__name__)

# Read config from .env (with sensible defaults)
LLM_MODEL    = os.getenv("OLLAMA_LLM_MODEL",   "llama3.2")
EMBED_MODEL  = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")
BASE_URL     = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")

# Global RAG chain for the current session
rag_chain = None

# -- Prompt Template ----------------------------------------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions strictly based on the provided document context.

Context:
{context}

Question: {question}

Instructions:
- Answer only from the provided context.
- If the answer is not found in the context, say "I couldn't find that in the document."
- Be concise and clear.

Answer:"""
)


def format_docs(docs):
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(text: str):
    """Split text -> embed -> FAISS store -> LCEL RAG chain (all via Ollama)."""

    # 1. Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)

    # 2. Embed locally with Ollama and store in FAISS
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=BASE_URL
    )
    vector_store = FAISS.from_texts(chunks, embeddings)

    # 3. Local LLM via Ollama
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=BASE_URL,
        temperature=0.2
    )

    # 4. Build LCEL chain: retriever | prompt | llm | parser
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


# -- Routes -------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Check if Ollama is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen(BASE_URL, timeout=3)
        return jsonify({"status": "ok", "ollama": BASE_URL, "llm": LLM_MODEL, "embed": EMBED_MODEL})
    except Exception:
        return jsonify({"status": "error", "message": f"Cannot reach Ollama at {BASE_URL}. Is it running?"}), 503


@app.route("/load", methods=["POST"])
def load_document():
    """Load document text and build the RAG chain."""
    global rag_chain

    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) < 50:
        return jsonify({"error": "Document too short. Please provide more content."}), 400

    try:
        rag_chain = build_rag_chain(text)
        word_count = len(text.split())
        return jsonify({
            "success": True,
            "message": f"Document loaded! ({word_count:,} words indexed)"
        })
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            return jsonify({"error": "Cannot connect to Ollama. Make sure Ollama is running (ollama serve)."}), 503
        return jsonify({"error": f"Failed to load document: {error_msg}"}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """Answer a question using the loaded document."""
    global rag_chain

    if rag_chain is None:
        return jsonify({"error": "No document loaded. Please load a document first."}), 400

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        answer = rag_chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            return jsonify({"error": "Lost connection to Ollama. Make sure it is still running."}), 503
        return jsonify({"error": f"Failed to answer: {error_msg}"}), 500


# -- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  LangChain Document Q&A  --  Ollama (Local AI)")
    print("=" * 55)
    print(f"  LLM model   : {LLM_MODEL}")
    print(f"  Embed model : {EMBED_MODEL}")
    print(f"  Ollama URL  : {BASE_URL}")
    print("=" * 55)
    print("  App running at: http://localhost:5000")
    print("  Press CTRL+C to quit")
    print("=" * 55)
    app.run(host="0.0.0.0", debug=True, port=5000)
