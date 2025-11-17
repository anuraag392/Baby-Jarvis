import os
import json
import base64
import requests
import numpy as np
from io import BytesIO
from datetime import datetime
import io
import contextlib
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import google.generativeai as genai

# Document extraction
from pypdf import PdfReader
import docx
import csv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# === NEW: Local Embeddings + FAISS RAG ===
import faiss
import google.generativeai as genai

class GeminiEmbeddingModel:
    def __init__(self, model_name="models/embedding-001"):
        self.model_name = model_name

    def encode(self, texts):
        # Ensure list input
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for t in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=t,
                task_type="retrieval_document"
            )
            vectors.append(result["embedding"])
        return vectors


# ============================================
# 1. ENV SETUP
# ============================================

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

FILE_SANDBOX = "workspace"
os.makedirs(FILE_SANDBOX, exist_ok=True)

RAG_INDEX_FILE = "rag_index.faiss"
RAG_METADATA_FILE = "rag_store.json"

# Chunk size for RAG (requested = 1200 chars)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# ============================================
# 2. LOAD SENTENCE TRANSFORMER MODEL (LOCAL)
# ============================================

# Fast, accurate, lightweight model — perfect for RAG
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = GeminiEmbeddingModel()
# ============================================
# 3. FAISS INDEX + RAG METADATA HANDLERS
# ============================================

def load_faiss():
    """Load FAISS index + metadata from disk, or initialize empty index."""
    if os.path.exists(RAG_INDEX_FILE) and os.path.exists(RAG_METADATA_FILE):
        index = faiss.read_index(RAG_INDEX_FILE)
        with open(RAG_METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    dim = 384  # embedding dimension for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dim)
    metadata = []
    return index, metadata


def save_faiss(index, metadata):
    """Persist FAISS index + metadata back to disk."""
    faiss.write_index(index, RAG_INDEX_FILE)
    with open(RAG_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ============================================
# 4. FILE EXTRACTION (PDF, DOCX, TXT, CSV)
# ============================================

def extract_text_from_file(filepath: str) -> str:
    ext = filepath.split(".")[-1].lower()

    # PDF
    if ext == "pdf":
        try:
            text = ""
            reader = PdfReader(filepath)
            for page in reader.pages:
                extracted = page.extract_text() or ""
                text += extracted + "\n"
            if not text.strip():
                return "PDF contains no extractable text (likely scanned)."
            return text
        except Exception as e:
            return f"PDF extraction error: {e}"

    # DOCX
    if ext == "docx":
        try:
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            return f"DOCX extraction error: {e}"

    # TXT
    if ext == "txt":
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            return f"TXT extraction error: {e}"

    # CSV
    if ext == "csv":
        try:
            rows = []
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(", ".join(row))
            return "\n".join(rows)
        except Exception as e:
            return f"CSV extraction error: {e}"

    return "Unsupported file type."


# ============================================
# 5. CHUNKING FOR RAG
# ============================================

def chunk_text(text: str):
    """Split text into overlapping chunks for RAG."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - CHUNK_OVERLAP  # overlapping for context
    return chunks


# ============================================
# 6. RAG INGESTION PIPELINE
# ============================================

def rag_ingest_file(filename: str) -> str:
    """Full ingestion pipeline: extract → chunk → embed → store."""
    filepath = os.path.join(FILE_SANDBOX, filename)

    if not os.path.exists(filepath):
        return f"File '{filename}' not found in workspace."

    # 1) Extract text
    text = extract_text_from_file(filepath)
    if not text.strip():
        return f"Could not extract text (empty file)."

    # 2) Chunking
    chunks = chunk_text(text)

    # 3) Embeddings
    texts_to_embed = [c for c in chunks]
    embeddings = np.array(embedding_model.encode(chunks), dtype="float32")
    index.add(embeddings)
    # 4) Load FAISS + metadata
    index, metadata = load_faiss()

    # 5) Add embeddings + metadata
    start_id = len(metadata)
    for i, chunk in enumerate(chunks):
        metadata.append({
            "chunk_id": start_id + i,
            "filename": filename,
            "text": chunk
        })

    index.add(embeddings)

    # 6) Save
    save_faiss(index, metadata)

    return f"Ingested '{filename}' with {len(chunks)} chunks."


# ============================================
# 7. RAG QUERY PIPELINE
# ============================================

def rag_query(question: str, top_k: int = 5) -> str:
    """Retrieve relevant chunks using FAISS and return small text for LLM."""
    index, metadata = load_faiss()

    if len(metadata) == 0:
        return "RAG store is empty. Please ingest a document first."

    # Embed question
    query_emb = np.array(embedding_model.encode([query]), dtype="float32")
    D, I = index.search(query_emb, top_k)

    retrieved_chunks = []
    for idx in I[0]:
        if idx < len(metadata):
            retrieved_chunks.append(metadata[idx]["text"])

    # Combine into a small context block
    context = "\n\n".join(retrieved_chunks)
    return context or "No relevant text found."
# ============================================
# 8. BASIC FILE TOOLS
# ============================================

def list_files() -> str:
    return "\n".join(os.listdir(FILE_SANDBOX))


def read_file(filename: str) -> str:
    p = os.path.join(FILE_SANDBOX, filename)
    if not os.path.exists(p):
        return f"File '{filename}' not found."
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return "Error reading file."


def write_file(filename: str, content: str) -> str:
    with open(os.path.join(FILE_SANDBOX, filename), "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {filename}."


def append_to_file(filename: str, content: str) -> str:
    with open(os.path.join(FILE_SANDBOX, filename), "a", encoding="utf-8") as f:
        f.write(content)
    return f"Appended to {filename}."


def delete_file(filename: str) -> str:
    p = os.path.join(FILE_SANDBOX, filename)
    if not os.path.exists(p):
        return "File missing."
    os.remove(p)
    return f"Deleted {filename}."


# ============================================
# 9. MEMORY TOOLS
# ============================================

MEMORY_FILE = "agent_memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_memory(mem):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2)


def add_to_memory(text: str) -> str:
    mem = load_memory()
    mem.append({"text": text})
    save_memory(mem)
    return "Stored in memory."


def ask_memory(query: str) -> str:
    mem = load_memory()
    query = query.lower()
    matches = [m["text"] for m in mem if query in m["text"].lower()]
    return "\n".join(matches) if matches else "No memory found."


# ============================================
# 10. SEARCH TOOL
# ============================================

def search(query: str) -> str:
    if not SERPER_API_KEY:
        return "SERPER_API_KEY missing."

    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            data=json.dumps({"q": query}),
        )
        data = r.json()
        out = []
        for item in data.get("organic", [])[:5]:
            out.append(
                f"Title: {item.get('title')}\n"
                f"Snippet: {item.get('snippet')}\n"
                f"URL: {item.get('link')}\n---"
            )
        return "\n".join(out) if out else "No results."
    except Exception as e:
        return f"Search error: {e}"


# ============================================
# 11. PYTHON EXECUTION TOOL
# ============================================

def execute_python_code(code: str) -> str:
    import io
    import contextlib
    import matplotlib.pyplot as plt

    stdout = io.StringIO()
    stderr = io.StringIO()

    safe_globals = {"__builtins__": __builtins__, "plt": plt}
    safe_locals = {}

    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, safe_globals, safe_locals)
    except Exception as e:
        return f"❌ Python error: {e}"

    out = stdout.getvalue()
    err = stderr.getvalue()

    img_data = ""
    fig = plt.gcf()
    if fig.get_axes():
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        plt.close()

    final = ""
    if out.strip():
        final += f"### Output\n```\n{out}\n```"
    if err.strip():
        final += f"\n### Error\n```\n{err}\n```"
    if img_data:
        final += f"\n<img src='data:image/png;base64,{img_data}'/>"

    return final if final else "✔ Code executed."


# ============================================
# 12. GOOGLE CALENDAR TOOL
# ============================================

from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_calendar_service():
    creds = None

    # Load existing token
    if os.path.exists("token.json"):
        try:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        except RefreshError:
            creds = None

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as t:
            t.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def google_calendar_action(
    action: str,
    summary: str = "",
    description: str = "",
    start_datetime: str = "",
    end_datetime: str = "",
    date: str = "",
    start_time: str = "",
    end_time: str = "",
    event_id: str = "",
):
    try:
        service = get_calendar_service()

        # CREATE
        if action == "create":
            if not start_datetime and date and start_time:
                start_datetime = f"{date}T{start_time}:00+05:30"
            if not end_datetime and date and end_time:
                end_datetime = f"{date}T{end_time}:00+05:30"

            event = {
                "summary": summary or "Untitled",
                "description": description,
                "start": {"dateTime": start_datetime, "timeZone": "Asia/Kolkata"},
                "end": {"dateTime": end_datetime, "timeZone": "Asia/Kolkata"},
            }
            created = service.events().insert(calendarId="primary", body=event).execute()
            return f"Created: {created.get('summary')}."

        # LIST
        if action == "list":
            kwargs = {
                "calendarId": "primary",
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if date:
                kwargs["timeMin"] = f"{date}T00:00:00+05:30"
                kwargs["timeMax"] = f"{date}T23:59:59+05:30"

            items = service.events().list(**kwargs).execute().get("items", [])
            if not items:
                return "No events found."

            out = []
            for e in items:
                s = e.get("start", {}).get("dateTime")
                out.append(f"{e['id']} | {e.get('summary')} | {s}")
            return "\n".join(out)

        # DELETE
        if action == "delete":
            if not event_id:
                return "Need event_id"
            service.events().delete(calendarId="primary", eventId=event_id).execute()
            return f"Deleted {event_id}."

        # UPDATE
        if action == "update":
            if not event_id:
                return "Need event_id."
            event = service.events().get(calendarId="primary", eventId=event_id).execute()
            if summary:
                event["summary"] = summary
            if description:
                event["description"] = description
            service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
            return f"Updated event {event_id}."

        return "Unknown action."

    except HttpError as e:
        return f"Google Calendar error: {e}"


# ============================================
# 13. TOOL SCHEMAS
# ============================================

tools = [
    {"name": "search",
     "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}},
    
    {"name": "list_files",
     "parameters": {"type": "OBJECT", "properties": {}, "required": []}},
    
    {"name": "read_file",
     "parameters": {"type": "OBJECT", "properties": {"filename": {"type": "STRING"}}, "required": ["filename"]}},
    
    {"name": "write_file",
     "parameters": {"type": "OBJECT", "properties": {"filename": {"type": "STRING"}, "content": {"type": "STRING"}}, "required": ["filename", "content"]}},
    
    {"name": "append_to_file",
     "parameters": {"type": "OBJECT", "properties": {"filename": {"type": "STRING"}, "content": {"type": "STRING"}}, "required": ["filename", "content"]}},
    
    {"name": "delete_file",
     "parameters": {"type": "OBJECT", "properties": {"filename": {"type": "STRING"}}, "required": ["filename"]}},
    
    {"name": "add_to_memory",
     "parameters": {"type": "OBJECT", "properties": {"text": {"type": "STRING"}}, "required": ["text"]}},
    
    {"name": "ask_memory",
     "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}},
    
    {"name": "execute_python_code",
     "parameters": {"type": "OBJECT", "properties": {"code": {"type": "STRING"}}, "required": ["code"]}},
    
    {"name": "google_calendar_action",
     "parameters": {"type": "OBJECT", "properties": {
         "action": {"type": "STRING"},
         "summary": {"type": "STRING"},
         "description": {"type": "STRING"},
         "start_datetime": {"type": "STRING"},
         "end_datetime": {"type": "STRING"},
         "date": {"type": "STRING"},
         "start_time": {"type": "STRING"},
         "end_time": {"type": "STRING"},
         "event_id": {"type": "STRING"}
     }, "required": ["action"]}},

    {"name": "rag_ingest_file",
     "parameters": {"type": "OBJECT", "properties": {"filename": {"type": "STRING"}}, "required": ["filename"]}},

    {"name": "rag_query",
     "parameters": {"type": "OBJECT", "properties": {"question": {"type": "STRING"}, "top_k": {"type": "NUMBER"}}, "required": ["question"]}},
]

TOOLS_BY_NAME = {t["name"]: t for t in tools}

tool_dispatch = {
    "search": search,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "append_to_file": append_to_file,
    "delete_file": delete_file,
    "add_to_memory": add_to_memory,
    "ask_memory": ask_memory,
    "execute_python_code": execute_python_code,
    "google_calendar_action": google_calendar_action,
    "rag_ingest_file": rag_ingest_file,
    "rag_query": rag_query,
}


# ============================================
# 14. MULTI-AGENT MODELS (Gemini 2.5 Flash)
# ============================================

# ROUTER MODEL
router_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction="""
Decide if the user request should go to:

- supervisor → If the request contains MORE THAN ONE task or has multiple stages.
- researcher → Questions about files, documents, summaries, PDFs, DOCX, TXT, CSV, RAG.
- coder → Writing/fixing/explaining code.
- scheduler → Meetings, calendar, reminders.
- writer → Emails, messages, letters, long-form writing.
- general → everything else.

If the user asks to summarize a previously ingested file:
    - ALWAYS call rag_query(question)
    - Summarize ONLY the returned context
    - NEVER say "I don't have the document"
    - NEVER refuse when RAG index has data

Respond with ONE WORD only.
Rules:
- If request involves PDF, DOCX, TXT, CSV, summarization or question answering → researcher
- If code, python, debugging → coder
- If calendar, meeting, schedule → scheduler
- Otherwise → general

Respond with ONE WORD ONLY.
You are a Research & Document Understanding Agent using RAG.

Rules:
1. For any summarization or analysis request:
   - First call rag_query(question)
   - Then summarize ONLY the returned context.
2. If the user asks to save, export, write, or download a summary:
   - Use write_file(filename, content).
3. If the user wants to add more notes:
   - Use append_to_file.
4. Never load full documents into the model.
5. Always keep results structured and clean.
"""
)

# GENERAL AGENT
model_general = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    tools=tools,
    system_instruction="General assistant with full tool access."
)

# CODER AGENT
model_coder = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    tools=[
        TOOLS_BY_NAME["rag_ingest_file"],
        TOOLS_BY_NAME["rag_query"],
        TOOLS_BY_NAME["search"],
        TOOLS_BY_NAME["read_file"],
        TOOLS_BY_NAME["write_file"],       # ADDED
        TOOLS_BY_NAME["append_to_file"],   # ADDED
        TOOLS_BY_NAME["delete_file"],      # ADDED
        TOOLS_BY_NAME["list_files"],
    ],
    system_instruction="""
You are a coding assistant.

Rules:
1. If the user asks for help debugging code, finding errors, explaining code, reviewing code, improving code, or refactoring code:
   - DO NOT CALL ANY TOOLS.
   - Always produce a full natural language explanation + corrected code.
2. Only call execute_python_code if the user explicitly says:
   "run this", "execute this", "test this code", "plot", or "execute".
3. NEVER produce function calls while debugging or explaining errors.
4. ALWAYS return the complete explanation and complete corrected code.
"""
)

# SCHEDULER AGENT
model_scheduler = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    tools=[
        TOOLS_BY_NAME["google_calendar_action"]
    ],
    system_instruction="Scheduling agent for Google Calendar tasks only."
)
# WRITER AGENT
model_writer = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    tools=[],
    system_instruction="""
You are a Writer Agent.

Your responsibilities:
- Write polished content such as emails, reports, summaries, blog posts, scripts.
- Format content cleanly.
- NEVER call tools.
- Do not execute code.
- Only produce clean text responses.
"""
)
# SUPERVISOR AGENT
model_supervisor = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction="""
You are the Supervisor Agent.

Your role:
1. Understand user request.
2. Break the task into sequential steps.
3. Assign each step to one of these agents:
   - researcher
   - coder
   - scheduler
   - writer
4. Clearly specify each step in this JSON format:

{
  "steps": [
    {"agent": "researcher", "task": "..."},
    {"agent": "coder", "task": "..."},
    {"agent": "scheduler", "task": "..."},
    {"agent": "writer", "task": "..."}
  ]
}

5. Steps MUST be high-level and minimal.
6. Do NOT solve tasks yourself.
7. Do NOT call tools.
8. Output ONLY valid JSON.
"""
)

# RESEARCHER AGENT (WITH RAG!)
model_researcher = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    tools=[
        TOOLS_BY_NAME["rag_ingest_file"],
        TOOLS_BY_NAME["rag_query"],
        TOOLS_BY_NAME["search"],
        TOOLS_BY_NAME["read_file"],
        TOOLS_BY_NAME["list_files"],
    ],
    system_instruction="""
You are a Research & Document Understanding Agent using RAG.

Workflow:
1. If a file is uploaded, use rag_ingest_file to ingest it.
2. For questions about the document, ALWAYS:
   - First call rag_query(question)
   - Then summarize/answer based ONLY on the returned small context.
3. NEVER load entire PDFs into the model.
4. Keep responses concise, accurate, and well-structured.
"""
)

SPECIALIST_MODELS = {
    "general": model_general,
    "coder": model_coder,
    "scheduler": model_scheduler,
    "researcher": model_researcher,
    "writer": model_writer,
    "supervisor": model_supervisor,
}
# ============================================
# 15. TOOL-CALL LOOP FOR EACH SPECIALIST
# ============================================

def _run_with_model(model, user_text, history):
    """
    Runs a single interaction with a specialized model.
    Handles tool calls until a final text response is produced.
    """
    chat = model.start_chat(history=history)
    res = chat.send_message(user_text)

    while True:
        part = res.candidates[0].content.parts[0]

        # ✔ Final text answer
        if getattr(part, "text", None):
            return part.text, chat.history
        if getattr(part, "function_call", None):
            
            # If user wants code explanation/debugging → ignore tool call completely
            debug_keywords = ["debug", "fix", "error", "explain", "issue", "bug"]
            if any(word in user_text.lower() for word in debug_keywords):
                # Return the model’s raw text instead of calling the tool
                return part.function_call.name + " (ignored due to debugging mode)", chat.history

            # Otherwise, process the tool normally
            name = part.function_call.name
            raw_args = dict(part.function_call.args)
            args = {}

            for k, v in raw_args.items():
                if isinstance(v, (dict, list)):
                    args[k] = json.dumps(v)
                else:
                    args[k] = v if v is not None else ""

            tool_fn = tool_dispatch.get(name)
            if not tool_fn:
                result = f"Tool {name} missing."
            else:
                try:
                    result = tool_fn(**args)
                except Exception as e:
                    result = f"Tool {name} error: {e}"

            res = chat.send_message({
                "function_response": {
                    "name": name,
                    "response": {"content": result}
                }
            })
            continue
        

            raw_args = dict(part.function_call.args)
            args = {}

            # --- SANITIZE ALL TOOL ARGUMENTS ---
            for k, v in raw_args.items():
                # Convert dicts/lists into string, because tools expect strings
                if isinstance(v, (dict, list)):
                    args[k] = json.dumps(v, ensure_ascii=False)
                else:
                    args[k] = v if v is not None else ""


            tool_fn = tool_dispatch.get(name)
            if not tool_fn:
                result = f"Tool '{name}' not found."
            else:
                try:
                    result = tool_fn(**args)
                except Exception as e:
                    result = f"Tool '{name}' error: {e}"

            # Send tool response back to the model
            res = chat.send_message({
                "function_response": {
                    "name": name,
                    "response": {"content": result}
                }
            })

def parse_supervisor_output(text):
    clean = (
        text.replace("```json", "")
            .replace("```", "")
            .strip()
    )

    # Find the first '{' and last '}'
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1:
        return None

    clean = clean[start:end+1]

    try:
        return json.loads(clean)
    except:
        return None


def supervisor_chain(user_text, history):
    plan_raw = model_supervisor.generate_content(user_text).text

    plan = parse_supervisor_output(plan_raw)

    if not plan or "steps" not in plan:
        return "Supervisor plan malformed. Falling back to normal answer.", history

    steps = plan["steps"]
    outputs = []

    # Sequential execution of each agent
    for step in steps:
        agent_name = step.get("agent", "general")
        task = step.get("task", "")

        if agent_name not in SPECIALIST_MODELS:
            agent_name = "general"

        model = SPECIALIST_MODELS[agent_name]
        out, history = _run_with_model(model, task, history)

        outputs.append(f"### {agent_name.upper()} OUTPUT\n{out}\n")

    # Writer combines everything
    final_prompt = (
        "Combine these outputs into a polished final answer:\n\n"
        + "\n\n".join(outputs)
    )

    final, history = _run_with_model(model_writer, final_prompt, history)
    return final, history





# ============================================
# 16. MAIN MULTI-AGENT ENTRYPOINT
# ============================================




def run_agent(user_text, history):
    try:
        route = router_model.generate_content(user_text)
        role = route.text.strip().lower()
    except:
        role = "general"

    # Supervisor handles multi-step tasks
    if role == "supervisor":
        return supervisor_chain(user_text, history)

    if role not in SPECIALIST_MODELS:
        role = "general"

    model = SPECIALIST_MODELS[role]
    answer, new_history = _run_with_model(model, user_text, history)

    return answer, new_history


