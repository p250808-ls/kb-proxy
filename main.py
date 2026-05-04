"""
行政法規知識庫系統 — 智慧入庫與查詢服務
版本：2.1.0
取代：AnythingLLM + 舊版 kb-proxy
功能：
  POST /ingest  — PDF 解析、切 chunk、Cohere 嵌入、存 Supabase
  POST /query   — 向量搜尋 + Gemini 生成答案
  GET  /health  — 健康檢查
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, os, io, json, re
from typing import Optional

# ── PDF 解析 ──────────────────────────────────────────────
import pdfplumber
from mistralai import Mistral

app = FastAPI(title="KB Service", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 環境變數 ───────────────────────────────────────────────
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
COHERE_API_KEY       = os.environ.get("COHERE_API_KEY", "")
GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY", "")
MISTRAL_API_KEY      = os.environ.get("MISTRAL_API_KEY", "")

COHERE_EMBED_MODEL   = "embed-multilingual-light-v3.0"
GEMINI_MODEL         = "gemini-2.5-flash"
CHUNK_SIZE           = 1200   # 字元數
CHUNK_OVERLAP        = 240
TOP_K                = 4
SIMILARITY_THRESHOLD = 0.25

# ── 意圖 → workspace tag 對應 ──────────────────────────────
INTENT_TAG_MAP = {
    "TRAINING":    "TRAINING",
    "LONGCARE":    "LONGCARE",
    "SUBSIDY":     "SUBSIDY",
    "HR":          "HR",
    "PROCUREMENT": "PROCUREMENT",
    "GENERAL":     "GENERAL",
    "OTHER":       "GENERAL",
}

DEFAULT_SYSTEM_PROMPT = """你是一位行政法規助理，請根據以下知識庫內容回答問題。
請用繁體中文回答，條列重點，並在最後說明資料來源。
若知識庫內容不足以完整回答，請說明不足之處並建議洽詢主管機關。"""

# ════════════════════════════════════════════════════════════
# 工具函數
# ════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """從 PDF 提取每頁文字，回傳 [{page, text}]"""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


def mistral_ocr_from_pdf(pdf_bytes: bytes, filename: str) -> list[dict]:
    """使用 Mistral OCR 解析掃描版 PDF，回傳 [{page, text}]"""
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        uploaded = client.files.upload(
            file={"file_name": filename, "content": pdf_bytes},
            purpose="ocr",
        )
        signed_url = client.files.get_signed_url(file_id=uploaded.id)
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed_url.url},
        )
        pages = [
            {"page": i + 1, "text": p.markdown}
            for i, p in enumerate(ocr_response.pages)
            if p.markdown and p.markdown.strip()
        ]
        client.files.delete(file_id=uploaded.id)
        return pages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mistral OCR 失敗：{str(e)}")


def chunk_text(pages: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    切 chunk：
    - 以段落為優先邊界
    - 保留 page_hint（來源頁碼）
    - chunk_size 字元，chunk_overlap 字元重疊
    """
    chunks = []
    buffer = ""
    current_page = 1

    for page_data in pages:
        page_text = page_data["text"]
        current_page = page_data["page"]
        paragraphs = re.split(r'\n{2,}|\r\n{2,}', page_text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(buffer) + len(para) > chunk_size:
                if buffer.strip():
                    chunks.append({
                        "text": buffer.strip(),
                        "page": current_page
                    })
                # 保留 overlap
                buffer = buffer[-chunk_overlap:] + "\n" + para
            else:
                buffer += ("\n" if buffer else "") + para

    # 最後剩餘
    if buffer.strip():
        chunks.append({
            "text": buffer.strip(),
            "page": current_page
        })

    return chunks


async def get_cohere_embeddings(
    texts: list[str],
    input_type: str = "search_document",
    api_key: str = "",
    model: str = COHERE_EMBED_MODEL,
) -> list[list[float]]:
    """呼叫 Cohere Embed API，回傳向量列表"""
    resolved_key = api_key if api_key else COHERE_API_KEY
    url = "https://api.cohere.com/v2/embed"
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "texts": texts,
        "input_type": input_type,
        "embedding_types": ["float"],
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Cohere error: {resp.text}")
    data = resp.json()
    return data["embeddings"]["float"]


async def classify_document(filename: str, summary: str, llm_model: str = GEMINI_MODEL, llm_api_key: str = "") -> list[str]:
    """用 Gemini 判斷文件應進入哪些 workspace tags"""
    resolved_key = llm_api_key if llm_api_key else GEMINI_API_KEY
    prompt = f"""分析以下文件的檔名與內容摘要，判斷應該分類到哪些知識庫。
只回傳 JSON 陣列，不要其他文字、說明或 markdown 符號。

可選類別（可複選）：
TRAINING（職業訓練、課程、師資、訓練計畫）
LONGCARE（長照法規、照顧服務、長期照顧）
SUBSIDY（補助申請、津貼、補貼、獎勵）
HR（人事規定、薪資、勞保、員工管理）
PROCUREMENT（採購招標、決標、投標）
GENERAL（無法明確分類者）

回傳範例：["TRAINING", "LONGCARE"]

檔名：{filename}
內容摘要（前500字）：{summary[:500]}
代碼："""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{llm_model}:generateContent?key={resolved_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)
    if resp.status_code != 200:
        return ["GENERAL"]

    raw = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

    # 解析 JSON 陣列
    try:
        tags = json.loads(raw)
        if isinstance(tags, list):
            valid = [t for t in tags if t in INTENT_TAG_MAP and t != "OTHER"]
            return valid if valid else ["GENERAL"]
    except Exception:
        pass
    return ["GENERAL"]


async def save_chunks_to_supabase(doc_id: int, chunks: list[dict], embeddings: list[list[float]], workspace_tags: list[str]):
    """批次寫入 document_chunks"""
    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "doc_id": doc_id,
            "chunk_index": i,
            "chunk_text": chunk["text"],
            "embedding": emb,
            "workspace_tags": workspace_tags,
            "page_hint": chunk.get("page"),
        })

    # Supabase REST API 批次插入（每批 50 筆）
    url = f"{SUPABASE_URL}/rest/v1/document_chunks"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Content-Profile": "kb",
        "Prefer": "return=minimal",
    }
    batch_size = 50
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            resp = await client.post(url, json=batch, headers=headers)
            if resp.status_code not in (200, 201):
                raise HTTPException(status_code=500, detail=f"Supabase insert error: {resp.text}")


async def save_doc_index(filename: str, workspace_tags: list[str]) -> int:
    """寫入 document_index，回傳 doc_id"""
    url = f"{SUPABASE_URL}/rest/v1/document_index"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Content-Profile": "kb",
        "Prefer": "return=representation",
    }
    payload = {
        "file_name": filename,
        "file_format": "pdf",
        "target_workspaces": workspace_tags,
        "primary_workspace": workspace_tags[0] if workspace_tags else "GENERAL",
        "classified_by": "auto",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Supabase doc_index error: {resp.text}")
    return resp.json()[0]["id"]


async def vector_search(
    query_embedding: list[float],
    filter_tags: list[str],
    match_count: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """呼叫 Supabase RPC 做向量搜尋"""
    url = f"{SUPABASE_URL}/rest/v1/rpc/search_chunks"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Content-Profile": "kb",
    }
    payload = {
        "query_embedding": query_embedding,
        "filter_tags": filter_tags,
        "match_count": match_count,
        "similarity_threshold": similarity_threshold,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        return []
    return resp.json()


async def generate_answer(
    question: str,
    context_chunks: list[dict],
    system_prompt: str = "",
    llm_model: str = GEMINI_MODEL,
    llm_api_key: str = "",
    temperature: float = 0.3,
) -> str:
    """用 Gemini 根據 context 生成答案"""
    if not context_chunks:
        return "目前知識庫中查無相關規定。\n\n建議您：\n• 換個關鍵字再試一次\n• 直接洽詢主管機關確認最新規定"

    resolved_key = llm_api_key if llm_api_key else GEMINI_API_KEY
    resolved_system = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    context = "\n\n---\n\n".join([c["chunk_text"] for c in context_chunks])
    prompt = f"""{resolved_system}

知識庫內容：
{context}

問題：{question}

回答："""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{llm_model}:generateContent?key={resolved_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini error: {resp.text}")

    return resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()


# ════════════════════════════════════════════════════════════
# API 端點
# ════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.1.0"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    admin_password: str = Form(...),
    chunk_size: int = Form(1200),
    chunk_overlap: int = Form(240),
    max_embed_length: int = Form(2048),
    embed_provider: str = Form("cohere"),
    embed_model: str = Form("embed-multilingual-light-v3.0"),
    embed_api_key: str = Form(""),
):
    """PDF 入庫：解析 → 分類 → 切chunk → 嵌入 → 存 Supabase"""

    # 1. 驗證密碼
    correct_pwd = os.environ.get("ADMIN_UPLOAD_PASSWORD", "")
    if admin_password != correct_pwd:
        raise HTTPException(status_code=403, detail="密碼錯誤")

    # 2. 讀取 PDF
    pdf_bytes = await file.read()
    filename = file.filename or "unknown.pdf"

    # 3. 提取文字（優先 pdfplumber，掃描版改用 Mistral OCR）
    ocr_used = False
    parse_method = "pdfplumber"
    pages = extract_text_from_pdf(pdf_bytes)
    total_text_len = sum(len(p["text"]) for p in pages)

    if total_text_len < 100:
        if not MISTRAL_API_KEY:
            raise HTTPException(status_code=400, detail="掃描版 PDF 需要設定 MISTRAL_API_KEY 環境變數")
        pages = mistral_ocr_from_pdf(pdf_bytes, filename)
        ocr_used = True
        parse_method = "mistral-ocr"

    if not pages:
        raise HTTPException(status_code=422, detail="無法從 PDF 提取文字，請確認非純掃描檔")

    # 4. 自動分類（Gemini 判斷 workspace tags）
    summary_text = " ".join([p["text"] for p in pages[:3]])  # 前3頁作為摘要
    workspace_tags = await classify_document(filename, summary_text)

    # 5. 切 chunk（使用傳入的 chunk_size / chunk_overlap）
    chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="切 chunk 失敗")

    # 6. Cohere 嵌入（批次處理，每批 96 筆）
    all_embeddings = []
    batch_size = 96
    texts = [c["text"][:max_embed_length] for c in chunks]
    for i in range(0, len(texts), batch_size):
        batch_embs = await get_cohere_embeddings(
            texts[i:i + batch_size],
            "search_document",
            api_key=embed_api_key,
            model=embed_model,
        )
        all_embeddings.extend(batch_embs)

    # 7. 寫入 document_index
    doc_id = await save_doc_index(filename, workspace_tags)

    # 8. 批次寫入 document_chunks
    await save_chunks_to_supabase(doc_id, chunks, all_embeddings, workspace_tags)

    return {
        "success": True,
        "filename": filename,
        "workspace_tags": workspace_tags,
        "chunks_count": len(chunks),
        "doc_id": doc_id,
        "ocr_used": ocr_used,
        "parse_method": parse_method,
    }


class QueryRequest(BaseModel):
    intent: str
    question: str
    session_id: str = "default"
    system_prompt: str = ""
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_api_key: str = ""
    temperature: float = 0.3
    embed_provider: str = "cohere"
    embed_model: str = "embed-multilingual-light-v3.0"
    embed_api_key: str = ""
    max_context_snippets: int = 4
    similarity_threshold: float = 0.25


@app.post("/query")
async def query(req: QueryRequest):
    """查詢：向量搜尋 + Gemini 生成答案"""

    # 1. 決定要過濾的 tag
    tag = INTENT_TAG_MAP.get(req.intent.upper(), "GENERAL")
    # GENERAL 查詢不過濾（搜尋所有文件）
    filter_tags = [tag] if tag != "GENERAL" else list(set(INTENT_TAG_MAP.values()) - {"GENERAL"}) + ["GENERAL"]

    # 2. 問題向量化
    query_embeddings = await get_cohere_embeddings(
        [req.question],
        "search_query",
        api_key=req.embed_api_key,
        model=req.embed_model,
    )
    query_embedding = query_embeddings[0]

    # 3. 向量搜尋
    results = await vector_search(
        query_embedding,
        filter_tags,
        match_count=req.max_context_snippets,
        similarity_threshold=req.similarity_threshold,
    )

    # 4. Gemini 生成答案
    answer = await generate_answer(
        req.question,
        results,
        system_prompt=req.system_prompt,
        llm_model=req.llm_model,
        llm_api_key=req.llm_api_key,
        temperature=req.temperature,
    )

    return {
        "answer": answer,
        "sources": [{"chunk_id": r["id"], "page": r.get("page_hint"), "similarity": r.get("similarity")} for r in results],
        "slug": tag.lower(),
        "has_result": len(results) > 0,
    }
