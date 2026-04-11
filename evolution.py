"""
Self-Evolution Engine — fully local. Uses the same Ollama model to analyze
feedback and rewrite its own system prompt over time.
"""
import json
import os
import re
from datetime import datetime
from typing import Optional, Callable, Awaitable
from llm.server import get_server

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "evolution.json")

DEFAULT_STATE = {
    "generation": 0,
    "total_requests": 0,
    "positive_feedback": 0,
    "negative_feedback": 0,
    "evolution_history": [],
    "current_system_prompt": (
        "You are an expert AI coding assistant running entirely on a local machine. "
        "Generate clean, efficient, well-commented code. "
        "Provide working implementations with clear explanations. "
        "Follow best practices for the target language. "
        "When fixing bugs, explain the root cause and the fix."
    ),
    "learned_patterns": [],
    "recent_sessions": [],
}

# ── persistence ───────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            data = json.load(f)
        for k, v in DEFAULT_STATE.items():
            if k not in data:
                data[k] = v
        return data
    return dict(DEFAULT_STATE)


def _save(state: dict):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_state() -> dict:
    return _load()

def get_system_prompt() -> str:
    return _load()["current_system_prompt"]

def success_rate(state: dict) -> float:
    total = state["positive_feedback"] + state["negative_feedback"]
    return state["positive_feedback"] / total if total else 1.0

# ── recording ─────────────────────────────────────────────────────────────────

def record_request(prompt: str, response_preview: str, mode: str) -> int:
    state = _load()
    state["total_requests"] += 1
    session = {
        "id": state["total_requests"],
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "prompt": prompt[:200],
        "response_preview": response_preview[:300],
        "feedback": None,
    }
    state["recent_sessions"] = ([session] + state["recent_sessions"])[:50]
    _save(state)
    return session["id"]


def record_feedback(session_id: int, positive: bool, comment: str = ""):
    state = _load()
    for s in state["recent_sessions"]:
        if s["id"] == session_id:
            s["feedback"] = {"positive": positive, "comment": comment}
            break
    if positive:
        state["positive_feedback"] += 1
    else:
        state["negative_feedback"] += 1
    _save(state)

# ── evolution ─────────────────────────────────────────────────────────────────

def _extract_tagged(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag>...</tag>."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_improvements(text: str) -> list[str]:
    """Pull bullet points from a block of text."""
    items = re.findall(r"(?:^|\n)\s*[-*•]\s*(.+)", text)
    return [i.strip() for i in items if i.strip()][:5]


async def run_evolution_cycle(
    broadcast_fn: Optional[Callable[[str], Awaitable[None]]] = None,
) -> dict:
    """
    Ask the local LLM to analyze its own performance and rewrite its system
    prompt. No internet required — runs entirely on-device.
    """
    state = _load()
    rated    = [s for s in state["recent_sessions"] if s["feedback"] is not None]
    positive = [s for s in rated if s["feedback"]["positive"]][:4]
    negative = [s for s in rated if not s["feedback"]["positive"]][:4]

    if broadcast_fn:
        await broadcast_fn("🧬 Starting local evolution cycle...")

    prompt = f"""You are improving an AI coding assistant's system prompt based on user feedback.

CURRENT SYSTEM PROMPT:
{state["current_system_prompt"]}

METRICS:
- Requests: {state["total_requests"]}
- Positive: {state["positive_feedback"]}  Negative: {state["negative_feedback"]}
- Success rate: {success_rate(state)*100:.1f}%
- Evolution #: {state["generation"]}

POSITIVE SESSIONS (worked well):
{json.dumps([{"prompt": s["prompt"], "mode": s["mode"]} for s in positive], indent=2) if positive else "None yet"}

NEGATIVE SESSIONS (needs improvement):
{json.dumps([{"prompt": s["prompt"], "mode": s["mode"], "comment": s["feedback"].get("comment","")} for s in negative], indent=2) if negative else "None yet"}

Write an improved system prompt. Use these exact tags in your response:

<analysis>
Brief analysis of what is working and what is not.
</analysis>

<improvements>
- improvement one
- improvement two
- improvement three
</improvements>

<new_prompt>
The complete new system prompt text here.
</new_prompt>

<impact>
One sentence on the expected improvement.
</impact>"""

    if broadcast_fn:
        await broadcast_fn("🤔 Local model is analyzing performance...")

    server = get_server()
    full_response = ""
    try:
        async for chunk in server.stream_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.4,
        ):
            full_response += chunk
    except Exception as e:
        if broadcast_fn:
            await broadcast_fn(f"❌ Model error: {e}")
        raise

    # Parse structured response
    analysis     = _extract_tagged(full_response, "analysis") or "Analysis not available"
    new_prompt   = _extract_tagged(full_response, "new_prompt")
    impact       = _extract_tagged(full_response, "impact") or ""
    improvements_text = _extract_tagged(full_response, "improvements") or ""
    improvements = _extract_improvements(improvements_text)

    # Sanity-check: must be non-trivial
    if not new_prompt or len(new_prompt) < 50:
        new_prompt = state["current_system_prompt"]
        improvements = ["No valid prompt generated — keeping current prompt"]

    record = {
        "generation": state["generation"] + 1,
        "timestamp": datetime.utcnow().isoformat(),
        "old_prompt": state["current_system_prompt"],
        "new_prompt": new_prompt,
        "analysis": analysis,
        "key_improvements": improvements,
        "expected_impact": impact,
        "metrics_at_evolution": {
            "total_requests": state["total_requests"],
            "success_rate": round(success_rate(state), 3),
        },
    }

    state["generation"] += 1
    state["current_system_prompt"] = new_prompt
    state["evolution_history"].append(record)
    state["learned_patterns"] = (improvements + state.get("learned_patterns", []))[:20]
    _save(state)

    if broadcast_fn:
        await broadcast_fn(f"✅ Evolution complete — generation {state['generation']}")

    return record
