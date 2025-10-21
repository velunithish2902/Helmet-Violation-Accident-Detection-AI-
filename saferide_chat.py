from __future__ import annotations
import os, re, json, uuid
import datetime as dt
from typing import List, Dict, Any

# LangChain-community
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# SQL
from sqlalchemy import create_engine, text as sql_text

# Email
from emailer import send_email  # your email helper

# -----------------------------
# Config
# -----------------------------
RAG_MODEL = os.getenv("RAG_MODEL","gpt-3.5-turbo")
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL","text-embedding-3-small")
INDEX_DIR = os.getenv("RAG_INDEX_DIR","indexes/faiss_reports")
DB_URL = os.getenv("DB_URL")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO","")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# -----------------------------
# LLM & Embeddings (explicit API key)
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

llm = ChatOpenAI(
    model_name=RAG_MODEL,
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# -----------------------------
# FAISS Vector Store
# -----------------------------
_vector = None
_retriever = None
if os.path.isdir(INDEX_DIR):
    try:
        _vector = FAISS.load_local(INDEX_DIR, emb)
        _retriever = _vector.as_retriever(search_kwargs={"k":4})
    except Exception as e:
        print("FAISS load failed:", e)

# -----------------------------
# PostgreSQL engine
# -----------------------------
_engine = None
if DB_URL:
    _engine = create_engine(DB_URL, pool_pre_ping=True)

# -----------------------------
# Helpers
# -----------------------------
def _as_text(x: Any) -> str:
    try: return x.content
    except: return str(x)

def _safe_json_default(obj: Any):
    if isinstance(obj, uuid.UUID): return str(obj)
    if isinstance(obj,(dt.datetime,dt.date)): return obj.isoformat()
    return obj

def _window_from_query(q: str):
    m = re.search(r"last\s+(\d+)\s*(day|days|week|weeks|month|months)", q.lower())
    since = dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=7)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("day"): since = dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=n)
        elif unit.startswith("week"): since = dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=7*n)
        elif unit.startswith("month"): since = dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=30*n)
    return since

# -----------------------------
# Tools
# -----------------------------
def tool_vector_search(query: str) -> Dict[str, Any]:
    if not _retriever:
        return {"answer":"Vector index not available.","sources":[]}
    docs = _retriever.get_relevant_documents(query)
    if not docs:
        return {"answer": _as_text(llm.predict(f"Answer briefly:\n{query}")), "sources":[]}
    blocks, sources = [], []
    for i, d in enumerate(docs,1):
        src = d.metadata.get("s3_key","unknown") if hasattr(d,"metadata") else "unknown"
        sources.append(src)
        blocks.append(f"[Doc {i} | {src}]\n{d.page_content.strip()}")
    prompt = f"Answer concisely using context:\n{os.linesep.join(blocks)}\nQuestion: {query}\nAnswer:"
    ans = llm.predict(prompt)
    return {"answer": ans, "sources": sources}

def tool_sql(query: str, limit: int=50) -> Dict[str, Any]:
    if not _engine:
        return {"rows": [], "answer": "DB not configured."}
    q = query.strip()
    if not q.lower().startswith("select"):
        q = f"SELECT id::text AS id, timestamp, class_name, confidence, s3_image_uri FROM detections ORDER BY timestamp DESC LIMIT 10"
    if "limit" not in q.lower():
        q = f"SELECT * FROM ({q}) AS subq LIMIT {limit}"
    rows = []
    with _engine.connect() as conn:
        res = conn.execute(sql_text(q))
        cols = list(res.keys())
        for r in res.fetchall():
            rows.append({c: (v.isoformat() if hasattr(v,'isoformat') else v) for c,v in zip(cols,r)})
    summary_prompt = f"Summarize table in 1-2 sentences.\nQuestion: {query}\nRows: {json.dumps(rows)}\nAnswer:"
    ans = llm.predict(summary_prompt)
    return {"rows": rows, "answer": ans}

def run_report_email_agent(query: str) -> Dict[str, Any]:
    since = _window_from_query(query)
    kpis, rows = {}, []
    if _engine:
        with _engine.connect() as conn:
            kpis["accidents"] = conn.execute(
                sql_text("SELECT COUNT(*) FROM detections WHERE class_name='accident' AND timestamp>=:s"), {"s": since}
            ).scalar() or 0
            kpis["nohelmet"] = conn.execute(
                sql_text("SELECT COUNT(*) FROM violations WHERE class_name='nohelmet' AND timestamp>=:s"), {"s": since}
            ).scalar() or 0
            res = conn.execute(
                sql_text("SELECT timestamp, source_name, confidence, s3_image_uri FROM violations WHERE timestamp>=:s ORDER BY timestamp DESC LIMIT 10"), {"s": since}
            )
            cols = list(res.keys())
            for r in res.fetchall():
                rows.append({c: (r[i].isoformat() if hasattr(r[i],'isoformat') else r[i]) for i,c in enumerate(cols)})

    recipients = [e.strip() for e in ALERT_EMAIL_TO.split(",") if e.strip()]
    emailed = False
    if recipients and SMTP_USER and SMTP_PASS:
        try:
            body = [f"Safety Summary since {since:%Y-%m-%d} UTC", ""]
            body += [f"- Accidents: {kpis['accidents']}", f"- No-helmet: {kpis['nohelmet']}", ""]
            for r in rows:
                body.append(f"{r['timestamp']} | {r['source_name']} | {r['confidence']} | {r['s3_image_uri']}")
            send_email("[SafeRide] Safety Summary", "\n".join(body), recipients, username=SMTP_USER, password=SMTP_PASS)
            emailed = True
        except Exception as e:
            body.append(f"Email failed: {e}")

    ans = f"Accidents: {kpis.get('accidents',0)}, No-helmet: {kpis.get('nohelmet',0)} since {since:%Y-%m-%d}."
    if emailed: ans += " Email sent."
    return {"mode":"REPORT","answer":ans,"rows":rows,"used_agents":["REPORT","EMAIL"] if emailed else ["REPORT"]}

def tool_presign(uri: str) -> str:
    return uri.strip()

# Wrappers
def run_sql_agent(q: str): return {**tool_sql(q), "mode":"SQL", "used_agents":["SQL"]}
def run_rag_agent(q: str): return {**tool_vector_search(q), "mode":"VECTOR", "used_agents":["RAG"]}

def answer(user_msg: str):
    q = user_msg.lower()
    if any(w in q for w in ["report","email","generate"]): return run_report_email_agent(user_msg)
    if any(w in q for w in ["count","latest","how many"]): return run_sql_agent(user_msg)
    return run_rag_agent(user_msg)

# -----------------------------
# Example queries
# -----------------------------
if __name__ == "__main__":
    queries = [
        "Show me all accidents from last week.",
        "Which camera has the most helmet violations?",
        "Email me a report of today's detections."
    ]
    for q in queries:
        print(f"Query: {q}")
        resp = answer(q)
        print(json.dumps(resp, indent=2, default=_safe_json_default))
        print("-"*50)
