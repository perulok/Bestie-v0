# backend.py — Bestie → OpenAI, com logs de tokens/custo e meta no retorno
from decimal import Decimal, InvalidOperation
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import sqlite3
from pathlib import Path

# Carrega .env explicitamente do diretório atual
load_dotenv(dotenv_path=".env")

# ==============================
# SQLite persistence (survives restarts)
# ==============================
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "bestie.db"

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            model TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            cost_usd REAL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS serpapi_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            query TEXT,
            results INTEGER
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            query TEXT,
            source TEXT,
            results INTEGER,
            error TEXT
        );
        """
    )
    conn.commit()
    conn.close()

init_db()

def get_daily_budget() -> Decimal:
    raw = os.getenv("DAILY_BUDGET_USD", "1.00")
    try:
        return Decimal(raw)
    except InvalidOperation:
        return Decimal("1.00")  # fallback seguro

# SDK da OpenAI com sua chave (NUNCA hardcode a chave no código)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Preços de referência (confira pricing oficial se mudar)
PRICE_IN_PER_M = Decimal("0.15")   # gpt-4o-mini input $/1M tokens
PRICE_OUT_PER_M = Decimal("0.60")  # gpt-4o-mini output $/1M tokens

# Orçamento diário local (proteção simples do seu lado)
DAILY_BUDGET_USD = get_daily_budget()
accumulated_today = Decimal("0.00")  # reinicia quando o servidor reinicia

# ==============================
# Estado em memória (MVP)
# ==============================
# Histórico por sessão (não persistente)
sessions: Dict[str, List[Dict[str, str]]] = {}

# Log de consumo para janela móvel de 24h
token_cost_log: List[Dict[str, Any]] = []
serpapi_call_log: List[datetime] = []  # timestamps of SerpAPI calls
discovery_log: List[Dict[str, Any]] = []  # recent discovery calls

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def prune_and_summarize_24h():
    cutoff = now_utc() - timedelta(hours=24)
    # remove itens antigos
    alive: List[Dict[str, Any]] = []
    total_cost = Decimal("0.00")
    total_prompt = 0
    total_completion = 0
    for entry in token_cost_log:
        ts: datetime = entry.get("ts")
        if ts and ts >= cutoff:
            alive.append(entry)
            total_cost += Decimal(str(entry.get("cost_usd", "0")))
            total_prompt += int(entry.get("prompt_tokens", 0))
            total_completion += int(entry.get("completion_tokens", 0))
    # substitui em memória
    token_cost_log.clear()
    token_cost_log.extend(alive)
    return {
        "cost_usd": total_cost,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }

def serpapi_calls_last_24h() -> int:
    cutoff = now_utc() - timedelta(hours=24)
    alive = [ts for ts in serpapi_call_log if ts >= cutoff]
    serpapi_call_log.clear()
    serpapi_call_log.extend(alive)
    return len(alive)

def estimate_cost_usd(input_toks: int, output_toks: int) -> Decimal:
    """Estimativa simples de custo com base em tabela de preços atual."""
    inp = (Decimal(input_toks) / Decimal(1_000_000)) * PRICE_IN_PER_M
    out = (Decimal(output_toks) / Decimal(1_000_000)) * PRICE_OUT_PER_M
    return (inp + out).quantize(Decimal("0.000001"))

# FastAPI app
app = FastAPI(title="Bestie backend")

# CORS: libera o front local (ajuste domínios em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health/Root (debug rápido)
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"status": "bestie-backend up"}

# Métricas simples para facilitar debug
@app.get("/metrics")
def metrics():
    summary_24h = prune_and_summarize_24h()
    # historical totals from SQLite
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT IFNULL(SUM(cost_usd),0) FROM chat_usage")
        total_cost_all = float(cur.fetchone()[0])
        cur.execute("SELECT IFNULL(SUM(total_tokens),0) FROM chat_usage")
        total_tokens_all = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(1) FROM serpapi_usage")
        total_serp_calls = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(1) FROM discoveries")
        total_discoveries = int(cur.fetchone()[0])
        conn.close()
    except Exception:
        total_cost_all = 0.0
        total_tokens_all = 0
        total_serp_calls = 0
        total_discoveries = 0
    return {
        "budget_daily_usd": str(DAILY_BUDGET_USD),
        "accumulated_today_usd": str(accumulated_today),
        "serpapi_calls_last_24h": serpapi_calls_last_24h(),
        "totals": {
            "chat_cost_usd": total_cost_all,
            "chat_tokens": total_tokens_all,
            "serpapi_calls": total_serp_calls,
            "discoveries": total_discoveries,
        },
        "rolling_24h": {
            "cost_usd": str(summary_24h["cost_usd"]),
            "prompt_tokens": int(summary_24h["prompt_tokens"]),
            "completion_tokens": int(summary_24h["completion_tokens"]),
            "total_tokens": int(summary_24h["total_tokens"]),
        },
        "active_sessions": len(sessions),
        "recent_calls": [
            {
                "ts": e["ts"].isoformat() if e.get("ts") else None,
                "model": e.get("model"),
                "prompt_tokens": e.get("prompt_tokens", 0),
                "completion_tokens": e.get("completion_tokens", 0),
                "cost_usd": str(e.get("cost_usd", "0")),
            }
            for e in token_cost_log[-10:]
        ],
        "recent_discoveries": discovery_log[-10:],
    }

# Reset de sessão para o front poder começar do zero
@app.post("/session/reset")
def reset_session(payload: dict):
    sid = payload.get("session_id") if isinstance(payload, dict) else None
    if sid and sid in sessions:
        del sessions[sid]
        return {"ok": True, "session_id": sid}
    return {"ok": False, "message": "session_id not found"}

# Payload do front
class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class DiscoverRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ProductOption(BaseModel):
    title: str
    url: str
    price: Optional[float] = None
    currency: Optional[str] = None
    retailer: Optional[str] = None
    image_url: Optional[str] = None

def parse_price(text: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        txt = text.strip().replace("\xa0", " ")
        # e.g. "$119.00" or "US$119.00"
        currency = None
        for sym in ["$", "US$", "R$", "€", "£"]:
            if sym in txt:
                currency = sym
                break
        digits = "".join(ch for ch in txt if ch.isdigit() or ch in ".,")
        digits = digits.replace(",", "")
        if digits:
            return float(digits), currency
    except Exception:
        pass
    return None, None

async def fetch_google_shopping(query: str, limit: int = 8) -> List[ProductOption]:
    url = "https://www.google.com/search"
    # Newer UI also uses udm=28; we allow Google to redirect and follow it
    params = {"tbm": "shop", "q": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    items: List[ProductOption] = []
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(url, params=params, headers=headers)
        # Do not raise for non-200 immediately; attempt to parse when 200 only
        if r.status_code != 200:
            # Try udm=28 explicitly
            r = await client.get(url, params={"q": query, "udm": "28"}, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"google returned status {r.status_code}")
        soup = BeautifulSoup(r.text, "html.parser")
        # Google changes markup often; try a few selectors
        cards = soup.select(".sh-dgr__grid-result, .sh-dgr__content, .sh-dgr__gr-auto, .i0X6df, .sh-pr__product-results .sh-dgr__content")
        seen = set()
        for c in cards:
            title_el = c.select_one(".Xjkr3b, .tAxDx, .sh-np__product-title, a[aria-label], a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            link_el = c.select_one("a[href]")
            href = link_el["href"] if link_el else None
            if not href or not title:
                continue
            if href.startswith("/url?"):
                # google redirects; keep as-is for MVP
                full_url = "https://www.google.com" + href
            elif href.startswith("http"):
                full_url = href
            else:
                full_url = "https://www.google.com" + href
            if full_url in seen:
                continue
            seen.add(full_url)

            price_el = c.select_one(".a8Pemb, .T14wmb, .QIrs8, .dD8iuc")
            price_val, curr = parse_price(price_el.get_text()) if price_el else (None, None)
            retailer_el = c.select_one(".aULzUe, .zE5r7b, .E5ocAb, .aULzUe .IuHnof")
            retailer = retailer_el.get_text(strip=True) if retailer_el else None
            img_el = c.select_one("img[src]")
            image_url = img_el["src"] if img_el else None
            items.append(ProductOption(title=title, url=full_url, price=price_val, currency=curr, retailer=retailer, image_url=image_url))
            if len(items) >= limit:
                break
    return items

async def fetch_serpapi_shopping(query: str, limit: int = 8) -> List[ProductOption]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY not set")
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": api_key,
        "num": max(10, limit * 2),
        "hl": "en",
        "gl": "us",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get("https://serpapi.com/search.json", params=params)
        if r.status_code != 200:
            raise RuntimeError(f"serpapi status {r.status_code}: {r.text[:200]}")
        data = r.json()
        results = []
        for item in data.get("shopping_results", [])[: limit * 3]:
            title = item.get("title")
            link = item.get("link") or item.get("product_link")
            price_str = item.get("extracted_price")
            price = float(price_str) if price_str is not None else None
            currency = item.get("currency") or "$"
            retailer = item.get("source") or item.get("merchant")
            image = item.get("thumbnail") or item.get("product_link")
            if title and link:
                results.append(ProductOption(title=title, url=link, price=price, currency=currency, retailer=retailer, image_url=image))
            if len(results) >= limit:
                break
        serpapi_call_log.append(now_utc())
        # persist serpapi usage
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO serpapi_usage (ts, query, results) VALUES (?,?,?)",
                (now_utc().isoformat(), query, len(results)),
            )
            conn.commit()
            conn.close()
        except Exception as dbe:
            print(f"[Bestie][WARN] failed to persist serpapi_usage: {dbe}")
        return results

# Endpoint principal de chat
@app.post("/chat")
def chat(req: ChatRequest):
    global accumulated_today

    # resolve/gera sessão
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.get(session_id, [])

    # 0) trava de custo diária local
    if accumulated_today >= DAILY_BUDGET_USD:
        msg = "Bestie: limite diário atingido. Tente amanhã."
        print(f"[Bestie] BLOQUEADO por orçamento diário: {accumulated_today} / {DAILY_BUDGET_USD}")
        return {"response": msg, "meta": {"blocked_by_budget": True, "session_id": session_id}}

    # 1) chamada à OpenAI com histórico
    messages = [
        {"role": "system", "content": "You are Bestie, an AI shopping agent. Be concise and clear."},
        *history,
        {"role": "user", "content": req.prompt},
    ]

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=400
        )
    except Exception as e:
        err_msg = f"Erro ao consultar o provedor de IA: {e}"
        print(f"[Bestie][ERROR] {err_msg}")
        # Retorna erro legível ao front mantendo a sessão
        return {
            "response": err_msg,
            "meta": {"error": True, "session_id": session_id},
            "history": history,
        }

    answer = r.choices[0].message.content
    model_name = getattr(r, "model", "unknown")

    # Atualiza histórico em memória
    history.append({"role": "user", "content": req.prompt})
    history.append({"role": "assistant", "content": answer})
    sessions[session_id] = history

    # 2) tokens e custo (se 'usage' vier do SDK)
    meta = {"model": model_name, "session_id": session_id}
    try:
        usage = r.usage
        in_toks = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
        out_toks = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
        total_toks = int(in_toks) + int(out_toks)
        call_cost = estimate_cost_usd(in_toks, out_toks)
        accumulated_today_local = accumulated_today + call_cost  # calcula antes de gravar

        # registra no log 24h
        token_cost_log.append({
            "ts": now_utc(),
            "model": model_name,
            "prompt_tokens": int(in_toks),
            "completion_tokens": int(out_toks),
            "cost_usd": call_cost,
        })
        summary_24h = prune_and_summarize_24h()

        # LOG amigável no terminal
        print(
            "[Bestie] model=%s | prompt_tokens=%s completion_tokens=%s total=%s | cost=$%s | accumulated_today=>$%s | last24h=$%s"
            % (model_name, in_toks, out_toks, total_toks, call_cost, accumulated_today_local, summary_24h["cost_usd"]) 
        )

        # atualiza acumulado do dia (teto simples)
        accumulated_today += call_cost
        # persist chat usage
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO chat_usage (ts, model, prompt_tokens, completion_tokens, total_tokens, cost_usd) VALUES (?,?,?,?,?,?)",
                (now_utc().isoformat(), model_name, int(in_toks), int(out_toks), int(total_toks), float(call_cost)),
            )
            conn.commit()
            conn.close()
        except Exception as dbe:
            print(f"[Bestie][WARN] failed to persist chat_usage: {dbe}")

        # meta no retorno
        meta.update({
            "tokens": {
                "prompt": int(in_toks),
                "completion": int(out_toks),
                "total": int(total_toks),
            },
            "estimated_cost_usd": str(call_cost),
            "accumulated_today_usd": str(accumulated_today),
            "rolling_24h": {
                "cost_usd": str(summary_24h["cost_usd"]),
                "prompt_tokens": int(summary_24h["prompt_tokens"]),
                "completion_tokens": int(summary_24h["completion_tokens"]),
                "total_tokens": int(summary_24h["total_tokens"]),
            }
        })
    except Exception as e:
        # Se por algum motivo não vier 'usage', apenas loga o básico
        print(f"[Bestie] model={model_name} | sem usage/tokens ({e})")

    return {"response": answer, "meta": meta, "history": history}

# ==========================
# Discovery (Google Shopping)
# ==========================
@app.post("/discover")
async def discover(req: DiscoverRequest):
    session_id = req.session_id or str(uuid.uuid4())
    try:
        # Prefer SerpAPI when available and under quota
        serp_key = os.getenv("SERPAPI_API_KEY")
        serp_quota = int(os.getenv("SERPAPI_DAILY_QUOTA", "50"))
        results: List[ProductOption] = []
        source_used = "html_fallback"
        if serp_key and serpapi_calls_last_24h() < serp_quota:
            try:
                results = await fetch_serpapi_shopping(req.query, limit=8)
                source_used = "serpapi"
            except Exception as e:
                print(f"[Bestie][WARN] SerpAPI failed, falling back to HTML: {e}")
        if not results:
            results = await fetch_google_shopping(req.query, limit=8)
            source_used = "html_fallback"
        # Simple top-5 by price ascending if available
        with_prices = [r for r in results if r.price is not None]
        without_prices = [r for r in results if r.price is None]
        with_prices.sort(key=lambda x: x.price)
        ordered = with_prices + without_prices
        top = ordered[:5]
        # log discovery summary
        entry = {
            "ts": now_utc().isoformat(),
            "query": req.query,
            "source": source_used,
            "results": len(top),
        }
        discovery_log.append(entry)
        print(f"[Bestie][DISCOVER] source={source_used} results={len(top)} query='{req.query}'")
        # persist discovery
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO discoveries (ts, query, source, results, error) VALUES (?,?,?,?,?)",
                (entry["ts"], req.query, source_used, len(top), None),
            )
            conn.commit()
            conn.close()
        except Exception as dbe:
            print(f"[Bestie][WARN] failed to persist discovery: {dbe}")
        return {"session_id": session_id, "query": req.query, "results": [r.dict() for r in top]}
    except Exception as e:
        print(f"[Bestie][ERROR] discover failed: {e}")
        discovery_log.append({
            "ts": now_utc().isoformat(),
            "query": req.query,
            "source": "error",
            "results": 0,
            "error": str(e)[:200]
        })
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO discoveries (ts, query, source, results, error) VALUES (?,?,?,?,?)",
                (now_utc().isoformat(), req.query, "error", 0, str(e)[:200]),
            )
            conn.commit()
            conn.close()
        except Exception as dbe:
            print(f"[Bestie][WARN] failed to persist discovery error: {dbe}")
        raise HTTPException(status_code=500, detail=f"discover failed: {e}")

# Debug endpoint to verify keys loaded
@app.get("/debug/whoami")
def whoami():
    key = os.getenv("OPENAI_API_KEY", "")
    org = os.getenv("OPENAI_ORG", "")
    serp = os.getenv("SERPAPI_API_KEY", "")
    def mask(s: str):
        return (s[:4] + "…" + s[-4:]) if len(s) > 8 else (s[:2] + "…")
    return {
        "openai_key": mask(key) if key else None,
        "openai_org": org or None,
        "serpapi_key": mask(serp) if serp else None,
    }

