from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, os

app = FastAPI(title="KB Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANYTHINGLLM_URL     = os.environ.get("ANYTHINGLLM_URL", "").rstrip("/")
ANYTHINGLLM_API_KEY = os.environ.get("ANYTHINGLLM_API_KEY", "")

WORKSPACE_MAP = {
    "TRAINING":    os.environ.get("WORKSPACE_TRAINING",    "vocational-training"),
    "LONGCARE":    os.environ.get("WORKSPACE_LONGCARE",    "long-term-care"),
    "SUBSIDY":     os.environ.get("WORKSPACE_SUBSIDY",     "subsidy-rules"),
    "HR":          os.environ.get("WORKSPACE_HR",          "hr-rules"),
    "PROCUREMENT": os.environ.get("WORKSPACE_PROCUREMENT", "procurement"),
    "GENERAL":     os.environ.get("WORKSPACE_GENERAL",     "general"),
    "OTHER":       os.environ.get("WORKSPACE_GENERAL",     "general"),
}


class QueryRequest(BaseModel):
    intent: str
    question: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryRequest):
    slug = WORKSPACE_MAP.get(req.intent.upper(), WORKSPACE_MAP["GENERAL"])
    url  = f"{ANYTHINGLLM_URL}/api/v1/workspace/{slug}/chat"
    headers = {
        "Authorization": f"Bearer {ANYTHINGLLM_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "message":   req.question,
        "mode":      "query",
        "sessionId": req.session_id,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    return {
        "answer":  data.get("textResponse", ""),
        "sources": data.get("sources", []),
        "slug":    slug,
    }
