import logging, uuid, config, llm, memory as mem, vector_store as vs
from langchain_text_splitters import RecursiveCharacterTextSplitter
logger = logging.getLogger(__name__)

def ingest_text(text: str, src: str) -> int:
    ts = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = ts.split_text(text)
    if not chunks: return 0
    ids = [str(uuid.uuid4()) for _ in chunks]
    mdata = [{"source": src, "index": i} for i in range(len(chunks))]
    vs.add_documents(chunks, mdata, ids)
    return len(chunks)

def query_rag(u_id: str, q: str) -> dict:
    docs = vs.similarity_search(q, top_k=config.TOP_K_DOCS)
    d_txts = [d["text"] for d in docs]
    mems = mem.retrieve_memory(u_id, q)
    
    ctx = (f"### Context from Documents:\n{'\n---\n'.join(d_txts) if d_txts else 'None'}\n\n"
           f"### Context from User Memory:\n- {'\n- '.join(mems) if mems else 'None'}")
    
    prompt = f"You are a helpful assistant with knowledge base and memory.\n{ctx}\nAnswer using context. Prioritize documents."
    msgs = llm.build_messages(prompt, q)
    ans = llm.generate(msgs)
    mem.store_memory(u_id, q, ans)
    return {"answer": ans, "retrieved_docs": d_txts, "retrieved_memories": mems}

