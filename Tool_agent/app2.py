import streamlit as st
import os
from agent4 import run_agent
from dotenv import load_dotenv

load_dotenv()  # loads .env locally, safe to keep in code

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Ensure workspace exists
os.makedirs("workspace", exist_ok=True)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="RAG Multi-Agent Assistant", layout="wide")

# =========================================================
# SIDEBAR — UPLOAD + INGEST + DOWNLOAD
# =========================================================
st.sidebar.header("📄 Upload Document")

uploaded = st.sidebar.file_uploader(
    "Upload PDF / DOCX / TXT / CSV",
    type=["pdf", "docx", "txt", "csv"]
)

if uploaded:
    save_path = os.path.join("workspace", uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Saved as {uploaded.name}")

    # Ingest button
    if st.sidebar.button("📥 Ingest into RAG"):
        from agent4 import rag_ingest_file
        msg = rag_ingest_file(uploaded.name)
        st.sidebar.info(msg)

# === WORKSPACE DOWNLOADS ===
# st.sidebar.header("📁 Workspace Files")
# files = os.listdir("workspace")

# if len(files) == 0:
#     st.sidebar.caption("No files generated yet.")
# else:
#     for f in files:
#         p = os.path.join("workspace", f)
#         with open(p, "rb") as fp:
#             st.sidebar.download_button(
#                 label=f"⬇️ {f}",
#                 data=fp,
#                 file_name=f,
#                 mime="text/plain"
#             )

# =========================================================
# CHAT UI
# =========================================================
st.title("🤖 Multi-Agent AI Assistant (Conversational Mode)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# =========================================================
# CHAT INPUT
# =========================================================
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Run through your agent system
    response, updated_history = run_agent(user_input, st.session_state.history)
    st.session_state.history = updated_history

    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)

