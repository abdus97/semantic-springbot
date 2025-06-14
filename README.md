
# 🔍 Semantic SpringBoot Code Search

This project enables **semantic search** across a Spring Boot codebase using **Sentence Transformers** for local embeddings and **ChromaDB** as a vector store—no OpenAI API key needed. Ask natural language questions like "How do we handle Slack events?" to retrieve relevant code snippets.

## 🚀 Features

- 🔍 Semantic search using natural language
- 🧠 Local embeddings with [Sentence Transformers](https://www.sbert.net/)
- 🗃️ Indexes `.java`, `.yml`, `.yaml`, `.md` files
- 💾 Stores embeddings in [ChromaDB](https://www.trychroma.com/)
- ⚡ Fast and private—no cloud dependencies

## 🛠️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdus97/semantic-springboot.git
cd semantic-springboot
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ingest Your Codebase
```bash
python ingest_code.py --path "/path/to/your/springboot-project"
```
> **Note**: Replace `/path/to/your/springboot-project` with the absolute path to your Spring Boot project. This scans, chunks, and indexes files.

### 5. Ask a Question
```bash
python query_bot.py --query "How do we handle Slack events?"
```
Returns relevant code snippets with file names and similarity scores.

## 🗂️ Project Structure
```plaintext
semantic springboot/
├── ingest_code.py         # Indexes codebase
├── query_bot.py           # Runs semantic queries
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignores venv, .env, etc.
├── .env.example           # Sample env file
└── README.md              # This file
```

## ⚙️ Technologies
- 🐍 Python 3.9+
- 🧠 sentence-transformers (`all-MiniLM-L6-v2`)
- 🗃️ chromadb
- 🌳 argparse, glob, os

## 📝 Notes
- Use absolute paths for `ingest_code.py`.
- Delete ChromaDB's database directory to re-index.
- Explore other Sentence Transformer models for better accuracy (edit `ingest_code.py`).



