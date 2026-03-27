"""
AI Coder Dashboard — FastAPI backend with WebSocket streaming and self-evolution.
Uses OpenAI-compatible API: OpenAI, Ollama, Groq, Mistral, LM Studio, etc.
"""
import json
import os
import asyncio
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ai_coder
import evolution

load_dotenv()

app = FastAPI(title="AI Coder Dashboard", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, msg: str):
        for ws in self.active:
            try:
                await ws.send_text(json.dumps({"type": "system", "message": msg}))
            except Exception:
                pass

manager = ConnectionManager()

# ── Request / Response models ─────────────────────────────────────────────────

class CodeRequest(BaseModel):
    prompt: str
    mode: str = "generate"          # generate|analyze|improve|explain|debug|test|chat
    language: str = "python"
    context: str = ""               # optional code to operate on

class FeedbackRequest(BaseModel):
    session_id: int
    positive: bool
    comment: str = ""

# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    return FileResponse("static/index.html")

@app.get("/api/status")
async def get_status():
    state = evolution.get_state()
    base_url = os.getenv("OPENAI_BASE_URL", "")
    model    = os.getenv("OPENAI_MODEL", "gpt-4o")
    if "ollama" in base_url or "11434" in base_url:
        provider = f"Ollama ({model})"
    elif "groq" in base_url:
        provider = f"Groq ({model})"
    elif "mistral" in base_url:
        provider = f"Mistral ({model})"
    elif base_url:
        provider = f"Custom ({model})"
    else:
        provider = f"OpenAI ({model})"
    return {
        "generation": state["generation"],
        "total_requests": state["total_requests"],
        "success_rate": round(evolution.success_rate(state) * 100, 1),
        "positive_feedback": state["positive_feedback"],
        "negative_feedback": state["negative_feedback"],
        "evolution_count": len(state["evolution_history"]),
        "learned_patterns": state["learned_patterns"],
        "current_prompt_preview": state["current_system_prompt"][:200] + "...",
        "provider": provider,
    }

@app.get("/api/history")
async def get_history():
    state = evolution.get_state()
    return {"sessions": state["recent_sessions"][:20]}

@app.get("/api/evolution/history")
async def get_evolution_history():
    state = evolution.get_state()
    return {"history": state["evolution_history"]}

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    evolution.record_feedback(req.session_id, req.positive, req.comment)
    return {"ok": True}

@app.post("/api/evolve")
async def trigger_evolution():
    """Manually trigger an evolution cycle."""
    try:
        record = await evolution.run_evolution_cycle(broadcast_fn=manager.broadcast)
        return {"ok": True, "record": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompt")
async def get_current_prompt():
    return {"prompt": evolution.get_system_prompt()}

# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type = data.get("type")

            if msg_type == "generate":
                await handle_generation(ws, data)

            elif msg_type == "evolve":
                await ws.send_text(json.dumps({
                    "type": "evolution_start",
                    "message": "Starting evolution cycle..."
                }))
                try:
                    record = await evolution.run_evolution_cycle(
                        broadcast_fn=lambda m: ws.send_text(json.dumps({"type": "system", "message": m}))
                    )
                    await ws.send_text(json.dumps({
                        "type": "evolution_complete",
                        "record": record
                    }))
                except Exception as e:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": f"Evolution failed: {str(e)}"
                    }))

            elif msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        manager.disconnect(ws)


async def handle_generation(ws: WebSocket, data: dict):
    prompt = data.get("prompt", "").strip()
    mode = data.get("mode", "generate")
    language = data.get("language", "python")
    context = data.get("context", "")

    if not prompt:
        await ws.send_text(json.dumps({"type": "error", "message": "Prompt is required"}))
        return

    # Signal start
    await ws.send_text(json.dumps({
        "type": "stream_start",
        "mode": mode,
        "language": language,
    }))

    full_response = ""
    try:
        async for chunk in ai_coder.stream_response(prompt, mode, language, context):
            full_response += chunk
            await ws.send_text(json.dumps({"type": "stream_chunk", "text": chunk}))
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        return

    # Record for evolution tracking
    session_id = evolution.record_request(prompt, full_response, mode)

    await ws.send_text(json.dumps({
        "type": "stream_end",
        "session_id": session_id,
        "total_chars": len(full_response),
    }))


# ── Auto-evolution scheduler ──────────────────────────────────────────────────

async def auto_evolution_loop():
    """Trigger evolution every 50 requests if success rate drops below 70%."""
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        try:
            state = evolution.get_state()
            rate = evolution.success_rate(state)
            reqs = state["total_requests"]
            last_evo_reqs = 0
            if state["evolution_history"]:
                last_evo_reqs = state["evolution_history"][-1].get(
                    "metrics_at_evolution", {}
                ).get("total_requests", 0)

            # Evolve if we've had 50+ new requests OR success rate < 70%
            if reqs - last_evo_reqs >= 50 or (rate < 0.7 and reqs > 10):
                await evolution.run_evolution_cycle(broadcast_fn=manager.broadcast)
        except Exception:
            pass


@app.on_event("startup")
async def startup():
    os.makedirs("data", exist_ok=True)
    asyncio.create_task(auto_evolution_loop())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
