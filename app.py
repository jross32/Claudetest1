"""
AI Coder Dashboard — fully local inference via CodeGPT.
No API keys, no internet, no external LLM services required.
"""
import json
import os
import asyncio
import threading
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ai_coder
import evolution
from llm.server import get_server

app = FastAPI(title="AI Coder Dashboard", version="3.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── WebSocket manager ─────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, msg: str):
        for ws in list(self.active):
            try:
                await ws.send_text(json.dumps({"type": "system", "message": msg}))
            except Exception:
                pass

    async def broadcast_json(self, payload: dict):
        text = json.dumps(payload)
        for ws in list(self.active):
            try:
                await ws.send_text(text)
            except Exception:
                pass

manager = ConnectionManager()

# ── Training state ────────────────────────────────────────────────────────────

_train_thread: Optional[threading.Thread] = None
_train_stop   = threading.Event()
_train_status = {"running": False, "phase": None, "step": 0, "loss": None, "tps": 0.0}

def _training_progress_cb(step: int, loss: float, tps: float):
    _train_status.update({"step": step, "loss": round(loss, 4), "tps": round(tps, 0)})
    asyncio.run_coroutine_threadsafe(
        manager.broadcast_json({
            "type":  "training_progress",
            "step":  step,
            "loss":  round(loss, 4),
            "tps":   round(tps, 0),
            "phase": _train_status["phase"],
        }),
        _app_loop,
    )

_app_loop: Optional[asyncio.AbstractEventLoop] = None

# ── Pydantic models ───────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: int
    positive: bool
    comment: str = ""

class TrainRequest(BaseModel):
    steps: int = 80_000
    batch_size: int = 8
    lr: float = 3e-4
    save_every: int = 500
    resume: bool = False
    extensions: list[str] = []   # empty = use all defaults (py + js/ts)

class FinetuneRequest(BaseModel):
    steps: int = 2_000
    batch_size: int = 4
    lr: float = 3e-5

# ── REST: general ─────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    return FileResponse("static/index.html")

@app.get("/api/status")
async def get_status():
    server = get_server()
    state  = evolution.get_state()
    return {
        "model":          server.model_info(),
        "generation":     state["generation"],
        "total_requests": state["total_requests"],
        "success_rate":   round(evolution.success_rate(state) * 100, 1),
        "positive_feedback":  state["positive_feedback"],
        "negative_feedback":  state["negative_feedback"],
        "evolution_count":    len(state["evolution_history"]),
        "learned_patterns":   state["learned_patterns"],
        "current_prompt_preview": state["current_system_prompt"][:200] + "...",
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
    try:
        record = await evolution.run_evolution_cycle(broadcast_fn=manager.broadcast)
        return {"ok": True, "record": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompt")
async def get_prompt():
    return {"prompt": evolution.get_system_prompt()}

# ── REST: model / training ────────────────────────────────────────────────────

@app.get("/api/model/info")
async def model_info():
    return get_server().model_info()

@app.get("/api/train/status")
async def train_status():
    return dict(_train_status)

@app.post("/api/train/start")
async def train_start(req: TrainRequest):
    global _train_thread
    if _train_status["running"]:
        raise HTTPException(status_code=409, detail="Training already running")

    _train_stop.clear()
    _train_status.update({"running": True, "phase": "base", "step": 0, "loss": None})

    _exts = tuple(req.extensions) if req.extensions else None

    def _run():
        from llm.train import train as _train
        try:
            _train(
                steps=req.steps,
                batch_size=req.batch_size,
                lr=req.lr,
                save_every=req.save_every,
                resume=req.resume,
                extensions=_exts,
                progress_callback=_training_progress_cb,
            )
        finally:
            _train_status["running"] = False
            _train_status["phase"]   = None
            get_server().reload()
            asyncio.run_coroutine_threadsafe(
                manager.broadcast_json({"type": "training_done", "phase": "base"}),
                _app_loop,
            )

    _train_thread = threading.Thread(target=_run, daemon=True)
    _train_thread.start()
    return {"ok": True, "message": "Phase-1 training started"}

@app.post("/api/train/stop")
async def train_stop():
    if not _train_status["running"]:
        return {"ok": False, "message": "Not running"}
    _train_stop.set()
    return {"ok": True, "message": "Stop signal sent"}

@app.post("/api/finetune/start")
async def finetune_start(req: FinetuneRequest):
    if _train_status["running"]:
        raise HTTPException(status_code=409, detail="Training already running")

    _train_stop.clear()
    _train_status.update({"running": True, "phase": "finetune", "step": 0, "loss": None})

    def _run():
        from llm.finetune import finetune as _finetune
        try:
            _finetune(
                steps=req.steps,
                batch_size=req.batch_size,
                lr=req.lr,
                progress_callback=_training_progress_cb,
            )
        finally:
            _train_status["running"] = False
            _train_status["phase"]   = None
            get_server().reload()
            asyncio.run_coroutine_threadsafe(
                manager.broadcast_json({"type": "training_done", "phase": "finetune"}),
                _app_loop,
            )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"ok": True, "message": "Phase-2 fine-tuning started"}

# ── WebSocket ─────────────────────────────────────────────────────────────────

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

            t = data.get("type")

            if t == "generate":
                await handle_generation(ws, data)

            elif t == "evolve":
                await ws.send_text(json.dumps({"type": "evolution_start",
                                               "message": "Starting evolution cycle..."}))
                try:
                    record = await evolution.run_evolution_cycle(
                        broadcast_fn=lambda m: ws.send_text(
                            json.dumps({"type": "system", "message": m})
                        ),
                    )
                    await ws.send_text(json.dumps({"type": "evolution_complete",
                                                   "record": record}))
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error",
                                                   "message": f"Evolution failed: {e}"}))

            elif t == "ping":
                server = get_server()
                await ws.send_text(json.dumps({
                    "type":  "pong",
                    "ready": server.is_ready(),
                    "model": server.model_info(),
                }))

    except WebSocketDisconnect:
        manager.disconnect(ws)


async def handle_generation(ws: WebSocket, data: dict):
    prompt = data.get("prompt", "").strip()
    mode   = data.get("mode", "generate")
    lang   = data.get("language", "python")
    ctx    = data.get("context", "")

    if not prompt:
        await ws.send_text(json.dumps({"type": "error", "message": "Prompt required"}))
        return

    server = get_server()
    if not server.is_ready():
        await ws.send_text(json.dumps({
            "type": "error",
            "message": "Model is not ready. Use the Model tab to start training first."
        }))
        return

    await ws.send_text(json.dumps({
        "type": "stream_start", "mode": mode, "language": lang,
    }))

    full = ""
    try:
        async for chunk in ai_coder.stream_response(prompt, mode, lang, ctx):
            full += chunk
            await ws.send_text(json.dumps({"type": "stream_chunk", "text": chunk}))
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        return

    session_id = evolution.record_request(prompt, full, mode)
    await ws.send_text(json.dumps({
        "type": "stream_end", "session_id": session_id, "total_chars": len(full)
    }))

# ── Auto-evolution ────────────────────────────────────────────────────────────

async def _auto_evolve():
    while True:
        await asyncio.sleep(300)
        try:
            server = get_server()
            if not server.is_ready():
                continue
            state = evolution.get_state()
            last  = 0
            if state["evolution_history"]:
                last = state["evolution_history"][-1]["metrics_at_evolution"]["total_requests"]
            rate = evolution.success_rate(state)
            reqs = state["total_requests"]
            if reqs - last >= 50 or (rate < 0.7 and reqs > 10):
                await evolution.run_evolution_cycle(broadcast_fn=manager.broadcast)
        except Exception:
            pass

@app.on_event("startup")
async def startup():
    global _app_loop
    _app_loop = asyncio.get_event_loop()
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    # Load model weights in background — non-blocking
    get_server().load_latest(blocking=False)
    asyncio.create_task(_auto_evolve())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
