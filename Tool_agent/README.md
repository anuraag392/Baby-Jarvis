Multi-Agent RAG Assistant (Tool Agent)

A powerful multi-agent AI system built with Gemini 2.5, capable of:

🔍 Intelligent document ingestion (PDF, DOCX, TXT, CSV)

📚 RAG (Retrieval-Augmented Generation) using FAISS

🧠 Multi-Agent Reasoning

👨‍💻 Code generation & debugging

✍️ Advanced writing assistance

📅 Scheduler agent (Google Calendar ready — per-user OAuth coming soon)

🧩 Supervisor agent for automatic task planning

💬 Streamlit conversational interface

This Space shows how modern LLMs can cooperate through specialized agents to achieve complex workflows autonomously.

🧠 Agents Included
1. Supervisor Agent

Breaks user requests into steps

Routes tasks to appropriate agents

Orchestrates multi-agent pipelines

Example:

“Summarize the PDF, write Python code for a graph, and create a meeting agenda.”

Supervisor converts this into a plan and executes:

Researcher → Summarization

Coder → Python code

Writer → Agenda

2. Researcher Agent

Handles document questions

Runs RAG queries

Extracts insights

Summarizes content

Performs deep analysis

Supports ingestion of:

PDF

DOCX

TXT

CSV

3. Coder Agent

Generates code

Fixes bugs

Explains algorithms

Writes multi-file projects

Creates patches

Has safety layers to prevent harmful execution.

4. Writer Agent

Writes emails

Summaries

Reports

Creative content

Professional documents

5. Scheduler Agent

(calendar login comes in next update)

Can schedule, list, modify, delete calendar events

Google Calendar integration ready

Per-user OAuth coming soon

📚 RAG (Retrieval-Augmented Generation)

The app includes a full RAG pipeline:

Embedding model (SentenceTransformer)

Chunking engine

Vector database using FAISS

Search + context building

Integrated into Researcher Agent

Upload → Ingest → Ask questions.

📁 File Ingestion

Supported formats:

.pdf

.docx

.txt

.csv

Upload your file → Click Ingest file → Ask any question.

💬 Conversational UI

Built with Streamlit.

Features:

Chat message history

Multi-agent routing

File uploader

Workspace file viewer

RAG ingestion controls

🚀 Deployment

This Space runs using:

Streamlit (UI)

FAISS (vector storage)

Gemini 2.5 Flash (LLM)

Google Generative AI Python SDK

HuggingFace Spaces CPU

🔧 Environment Variables Required

Add these in Settings → Variables & Secrets:

GEMINI_API_KEY


Optional:

SERPER_API_KEY


(For search tool integration)

Upcoming:

GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET


(for per-user Google Calendar login)

📦 Requirements

See requirements.txt in the repo:

streamlit==1.33.0
google-generativeai==0.5.2
faiss-cpu==1.7.4
sentence-transformers==2.2.2
numpy
pypdf
python-docx
python-dotenv
google-api-python-client
google-auth
google-auth-oauthlib
protobuf==4.25.3
requests

🔜 Upcoming Features

🔐 Per-user Google Calendar login (OAuth)

📸 OCR for scanned PDFs

📊 Table extraction + spreadsheet reasoning

🧠 Persistent long-term memory

🔁 Background agents (auto-research agents)

🕹️ Voice input + TTS output

🌑 Dark mode UI

🗂️ Multi-file RAG ingestion

🧑‍💻 Author

Built by Anuraag Das
Multi-agent AI • RAG • LLM orchestration • Agentic automation

⭐ If you like this Space — give it a ⭐ on HuggingFace!