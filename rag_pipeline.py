import logging, uuid, config, llm, memory as mem, vector_store as vs
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import io
from pypdf import PdfReader
from prompts import build_rag_system_prompt
from utils import calculate_chunk_stats

logger = logging.getLogger(__name__)

def parse_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Failed to parse PDF bytes: {e}")
        return ""

def ingest_text(text: str, src: str) -> int:
    ts = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = ts.split_text(text)
    if not chunks: return 0
    
    stats = calculate_chunk_stats(chunks)
    logger.info(f"Ingesting {src}: chunk stats -> {stats}")
    
    ids = [str(uuid.uuid4()) for _ in chunks]
    timestamp = time.time()
    mdata = [
        {
            "source": src, 
            "index": i, 
            "file_type": "markdown" if src.endswith(".md") else ("pdf" if src.endswith(".pdf") else "text"),
            "ingested_at": timestamp
        } for i in range(len(chunks))
    ]
    vs.add_documents(chunks, mdata, ids)
    return len(chunks)

def query_rag(u_id: str, q: str) -> dict:
    docs = vs.similarity_search(q, top_k=config.TOP_K_DOCS)
    d_txts = [d["text"] for d in docs]
    mems = mem.retrieve_memory(u_id, q)
    
    ctx = (f"### Context from Documents:\n{'
---
'.join(d_txts) if d_txts else 'None'}\n\n"
           f"### Context from User Memory:\n- {'
- '.join(mems) if mems else 'None'}")
    
    prompt = build_rag_system_prompt(ctx)
    msgs = llm.build_messages(prompt, q)
    ans = llm.generate(msgs)
    mem.store_memory(u_id, q, ans)
    return {"answer": ans, "retrieved_docs": d_txts, "retrieved_memories": mems}
