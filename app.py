import streamlit as st, logging, io, config, rag_pipeline as rp, memory as mem
st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="wide")
st.markdown("""<style>@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Inter:wght@300;400;600&display=swap');:root{--bg-color:#050505;--sidebar-bg:#0a0a0a;--text-color:#e0e0e0;--accent-color:#00f2ff;--accent-glow:0 0 10px rgba(0,242,255,0.5);--card-bg:rgba(20,20,20,0.6);--border-color:#333}.stApp{background-color:var(--bg-color);font-family:Inter,sans-serif;color:var(--text-color)}h1,h2,h3,h4,h5,h6{font-family:Rajdhani,sans-serif;font-weight:700;text-transform:uppercase;letter-spacing:1px}h1{background:linear-gradient(90deg,#fff,#888);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-shadow:0 0 20px rgba(255,255,255,0.1)}[data-testid="stSidebar"]{background-color:var(--sidebar-bg);border-right:1px solid var(--border-color)}.stButton>button{background:transparent;border:1px solid var(--accent-color);color:var(--accent-color);font-family:Rajdhani,sans-serif;text-transform:uppercase;letter-spacing:1px;transition:all 0.3s ease;border-radius:4px}.stButton>button:hover{background:var(--accent-color);color:#000;box-shadow:var(--accent-glow);border-color:var(--accent-color)}[data-testid="stChatMessage"]{background:transparent;border-bottom:1px solid #1a1a1a}[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]{font-family:Inter,sans-serif;line-height:1.6}.memory-card{background:var(--card-bg);border:1px solid var(--border-color);padding:1rem;border-radius:8px;margin-bottom:0.5rem;transition:all 0.2s}.memory-card:hover{border-color:var(--accent-color);transform:translateX(5px)}.memory-id{font-size:0.7rem;color:#666;font-family:Rajdhani,monospace}.streamlit-expanderHeader{background-color:#111;border-radius:4px;font-family:Rajdhani,sans-serif}.stChatInputContainer{border-top:1px solid var(--border-color);padding-top:1rem}</style>""", unsafe_allow_html=True)

u_id = config.DEFAULT_USER_ID
if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.markdown("## 🧠 SYSTEM STATUS\n### MEMORY BANK")
    if st.button("PURGE MEMORY", type="primary"):
        mem.clear_memory(u_id); st.success("MEMORY CLEARED"); st.rerun()
    
    memory_search = st.text_input("FILTER MEMORIES", "")
    
    try:
        mems = mem.get_all_memories(u_id)
        if mems:
            filtered_mems = [m for m in mems if not memory_search or memory_search.lower() in (m.get('memory') or m.get('text', '')).lower()]
            st.markdown(f"<div style='color:#888; font-size:0.8rem'>{len(filtered_mems)} ENGRAMS MATCHED</div>", unsafe_allow_html=True)
            for m in filtered_mems:
                col1, col2 = st.columns([0.8, 0.2])
                m_id = m.get('id', 'N/A')
                with col1:
                    st.markdown(f"<div class='memory-card'><div style='color:#ddd; font-size:0.9rem;'>{m.get('memory') or m.get('text', 'Unknown')}</div><div class='memory-id'>ID: {m_id}</div></div>", unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️", key=f"del_{m_id}"):
                        mem.delete_memory(u_id, m_id)
                        st.rerun()
        else: st.info("NO MEMORY")
    except Exception as e: st.error(f"SYNC FAIL: {e}")
    
    st.markdown("---
### HYPERPARAMETERS")
    temperature = st.slider("MODEL TEMPERATURE", min_value=0.0, max_value=1.0, value=config.TEMPERATURE, step=0.05)
    
    st.markdown("---
### DIAGNOSTICS")
    show_telemetry = st.checkbox("SHOW TELEMETRY", value=True)
    
    st.markdown("---
### DATA INGESTION")
    files = st.file_uploader("UPLOAD", type=["txt", "md"], accept_multiple_files=True)
    if files and st.button("INITIATE"):
        with st.spinner("PROCESSING..."):
            valid_files = [f for f in files if f.name.endswith(('.txt', '.md', '.pdf'))]
            if not valid_files:
                st.error("No valid text, markdown or PDF files selected.")
            else:
                total = sum(rp.ingest_text(io.StringIO(f.getvalue().decode('utf-8', errors='ignore')).read(), f.name) for f in valid_files)
                st.success(f"INDEXED {total} FRAGMENTS")

c1, c2 = st.columns([0.1, 0.9])
with c1: st.markdown(f"<div style='font-size:3rem; padding-top:10px;'>{config.APP_ICON}</div>", unsafe_allow_html=True)
with c2: st.title("NEURAL A.I."); st.markdown(f"<div style='color:#666; font-family:Rajdhani; margin-top:-20px;'>{config.APP_TITLE.upper()} | ONLINE</div>", unsafe_allow_html=True)
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if p := st.chat_input("Input command..."):
    p_stripped = p.strip()
    if not p_stripped:
         st.warning("Empty commands cannot be processed.")
    else:
         st.session_state.messages.append({"role": "user", "content": p_stripped})
         with st.chat_message("user"): st.markdown(p_stripped)
         with st.chat_message("assistant"):
             with st.spinner("ANALYZING..."):
                 try:
                     res = rp.query_rag(u_id, p_stripped)
                     st.markdown(res["answer"])
                     d, m = res.get("retrieved_docs", []), res.get("retrieved_memories", [])
                     if (d or m) and show_telemetry:
                         with st.expander("🔎 NEURAL TRACES"):
                             if m: st.markdown("#### 🧠 MEMORIES\n" + "\n".join(f"- {x}" for x in m))
                             if d: st.markdown("#### 📄 DOCUMENTS\n" + "\n".join(f"**{i+1}**: {x[:150]}..." for i, x in enumerate(d)))
                     st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
                 except Exception as e: st.error(f"ERROR: {e}")
