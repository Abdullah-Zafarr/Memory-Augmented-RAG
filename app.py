"""
app.py
------
Streamlit UI for the Memory-Augmented-RAG system.
Design: High-Tech / Cyber-Dark Aesthetic.
"""

import logging
import streamlit as st
import io

# Local imports
import config
import rag_pipeline
import memory

# Ensure logs show up in terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

user_id = config.DEFAULT_USER_ID

# ---------------------------------------------------------------------------
# Custom CSS (Cyber-Dark Aesthetic)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Inter:wght@300;400;600&display=swap');

    /* Global Variables */
    :root {
        --bg-color: #050505;
        --sidebar-bg: #0a0a0a;
        --text-color: #e0e0e0;
        --accent-color: #00f2ff;
        --accent-glow: 0 0 10px rgba(0, 242, 255, 0.5);
        --card-bg: rgba(20, 20, 20, 0.6);
        --border-color: #333;
    }

    /* Base Styles */
    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    h1 {
        background: linear-gradient(90deg, #fff, #888);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(255,255,255,0.1);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
    }

    /* Buttons */
    .stButton > button {
        background: transparent;
        border: 1px solid var(--accent-color);
        color: var(--accent-color);
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        border-radius: 4px;
    }

    .stButton > button:hover {
        background: var(--accent-color);
        color: #000;
        box-shadow: var(--accent-glow);
        border-color: var(--accent-color);
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: transparent;
        border-bottom: 1px solid #1a1a1a;
    }
    
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }

    /* Custom Cards for Memories */
    .memory-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
    }
    .memory-card:hover {
        border-color: var(--accent-color);
        transform: translateX(5px);
    }
    .memory-id {
        font-size: 0.7rem;
        color: #666;
        font-family: 'Rajdhani', monospace;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #111;
        border-radius: 4px;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Input Field */
    .stChatInputContainer {
        border-top: 1px solid var(--border-color);
        padding-top: 1rem;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 SYSTEM STATUS")
    
    # Memory Section
    st.markdown("### MEMORY BANK")
    
    if st.button("PURGE MEMORY", type="primary"):
        memory.clear_memory(user_id)
        st.success("MEMORY CLEARED")
        st.rerun()

    # Fetch and display memories
    try:
        all_mems = memory.get_all_memories(user_id)
        if all_mems:
            st.markdown(f"<div style='margin-bottom:10px; color:#888; font-size:0.8rem'>{len(all_mems)} ENGRAMS DETECTED</div>", unsafe_allow_html=True)
            for m in all_mems:
                content = m.get("memory") or m.get("text", "Unknown data")
                m_id = m.get('id', 'N/A')
                # Custom HTML card
                st.markdown(f"""
                <div class="memory-card">
                    <div style="color:#ddd; font-size:0.9rem;">{content}</div>
                    <div class="memory-id">ID: {m_id}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("NO MEMORY TRACES FOUND")
    except Exception as e:
        st.error(f"MEMORY SYNC FAILURE: {e}")

    st.markdown("---")
    
    # Document Ingestion
    st.markdown("### DATA INGESTION")
    uploaded_files = st.file_uploader(
        "UPLOAD SOURCE FILES", 
        type=["txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("INITIATE UPLOAD"):
        with st.spinner("PROCESSING DATA STREAMS..."):
            total_chunks = 0
            for uploaded_file in uploaded_files:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                text_content = stringio.read()
                count = rag_pipeline.ingest_text(text_content, uploaded_file.name)
                total_chunks += count
            
            st.success(f"UPLOAD COMPLETE: {total_chunks} FRAGMENTS INDEXED")


# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------
# Header
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.markdown(f"<div style='font-size:3rem; padding-top:10px;'>{config.APP_ICON}</div>", unsafe_allow_html=True)
with col2:
    st.title("NEURAL A.I. ASSISTANT")
    st.markdown(f"<div style='color: #666; font-family: Rajdhani; margin-top: -20px;'>{config.APP_TITLE.upper()} | SYSTEM ONLINE</div>", unsafe_allow_html=True)

st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Input command or query..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("ANALYZING CONTEXT & MEMORY..."):
            try:
                result = rag_pipeline.query_rag(user_id, prompt)
                response_text = result["answer"]
                
                st.markdown(response_text)
                
                # Context Expander
                docs = result.get("retrieved_docs", [])
                mems = result.get("retrieved_memories", [])
                
                if docs or mems:
                    with st.expander("🔎 NEURAL TRACES & SOURCE DATA"):
                        if mems:
                            st.markdown("#### 🧠 RECALLED MEMORIES")
                            for m in mems:
                                st.markdown(f"- {m}")
                        if docs:
                            st.markdown("#### 📄 SOURCE DOCUMENTS")
                            for i, d in enumerate(docs):
                                st.markdown(f"**FRAGMENT {i+1}**: _{d[:150]}..._")

                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"SYSTEM ERROR: {e}")
                logger.exception("RAG Pipeline Error")
