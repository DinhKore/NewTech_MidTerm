# Midterm Exam: RAG Chatbot System

**Course:** New Technologies in Software Engineering (MTSE431179E)

## 1. System Specifications
- **Primary LLM:** Google `gemini-3.8-flash` (via Google Generative AI)
- **Backup LLM:** `llama3-8b-8192` (via Groq) - Prepared for rate-limit scenarios.
- **Embedding Model:** Google `models/gemini-embedding-001`
- **Vector Database:** ChromaDB
- **Chunking Strategy:** `RecursiveCharacterTextSplitter`
  - Chunk Size: 1000 characters
  - Chunk Overlap: 200 characters

## 2. Features Implemented
- Document ingestion and vector indexing.
- Context-aware Q&A using LCEL (LangChain Expression Language).
- Strict hallucination prevention (Answers "I don't know" if out of context).
- Source citation and retrieved chunk visualization using Streamlit expander.
- Session State management to prevent redundant PDF processing.

## 3. How to Run the Bot

**Step 1: Create a virtual environment**
```bash
python -m venv venv
```

**Step 2: Activate the environment**
* **For Windows:**
  ```bash
  .\venv\Scripts\activate
  ```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Configure API Keys**
Create a `.env` file in the root directory and add your API keys:
```text
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**Step 5: Run the application**
```bash
streamlit run app.py
```