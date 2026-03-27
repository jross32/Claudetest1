"""
Self-Evolution Engine — tracks performance, learns from feedback, and uses the
configured LLM to rewrite its own system prompts over time.
"""
import json
import os
from datetime import datetime
from openai import AsyncOpenAI

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "evolution.json")

DEFAULT_STATE = {
    "generation": 0,
    "total_requests": 0,
    "positive_feedback": 0,
    "negative_feedback": 0,
    "evolution_history": [],
    "current_system_prompt": (
        "You are an expert AI coding assistant. Generate clean, efficient, well-commented code. "
        "When asked to write code, provide working implementations with clear explanations. "
        "Follow best practices for the language being used. "
        "If fixing bugs, explain what was wrong and why the fix works. "
        "Always prefer readability and correctness over cleverness."
    ),
    "learned_patterns": [],
    "recent_sessions": [],
}


def _client() -> AsyncOpenAI:
    base_url = os.getenv("OPENAI_BASE_URL") or None
    api_key  = os.getenv("OPENAI_API_KEY", "no-key")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

def _model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o")


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


def success_rate(state: dict) -> float:
    total = state["positive_feedback"] + state["negative_feedback"]
    if total == 0:
        return 1.0
    return state["positive_feedback"] / total


async def run_evolution_cycle(broadcast_fn=None) -> dict:
    """
    Uses the configured LLM to analyze recent sessions and rewrite
    the system prompt to perform better. This is the self-evolution step.
    """
    state = _load()
    client = _client()

    rated    = [s for s in state["recent_sessions"] if s["feedback"] is not None]
    positive = [s for s in rated if s["feedback"]["positive"]][:5]
    negative = [s for s in rated if not s["feedback"]["positive"]][:5]

    if broadcast_fn:
        await broadcast_fn("🧬 Starting evolution cycle...")

    analysis_prompt = f"""You are the meta-AI responsible for improving an AI coding assistant.

Current system prompt:
<current_prompt>
{state["current_system_prompt"]}
</current_prompt>

Performance metrics:
- Total requests: {state["total_requests"]}
- Positive feedback: {state["positive_feedback"]}
- Negative feedback: {state["negative_feedback"]}
- Success rate: {success_rate(state)*100:.1f}%
- Evolution generation: {state["generation"]}

Sessions with POSITIVE feedback (what worked well):
{json.dumps(positive, indent=2) if positive else "None yet — using general best practices"}

Sessions with NEGATIVE feedback (what needs improvement):
{json.dumps(negative, indent=2) if negative else "None yet — using general best practices"}

Your task: Analyze the patterns, then write an improved system prompt that will perform better.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "analysis": "brief analysis of patterns",
  "key_improvements": ["improvement 1", "improvement 2", "improvement 3"],
  "new_system_prompt": "the complete improved system prompt",
  "expected_impact": "how this should help"
}}"""

    if broadcast_fn:
        await broadcast_fn("🤔 Analyzing performance patterns...")

    try:
        # Try with json_object response format (supported by OpenAI, Groq, some Ollama models)
        # Fall back to plain text parsing if not supported
        try:
            response = await client.chat.completions.create(
                model=_model(),
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            evolution_data = json.loads(response.choices[0].message.content)
        except Exception:
            # Fallback: no response_format constraint
            response = await client.chat.completions.create(
                model=_model(),
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content or ""
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                evolution_data = json.loads(raw[start:end])
            else:
                raise ValueError("No JSON in response")

    except Exception as e:
        if broadcast_fn:
            await broadcast_fn(f"⚠️ Using heuristic evolution (LLM parse failed: {e})")
        evolution_data = {
            "analysis": "Heuristic evolution — no rated sessions available yet.",
            "key_improvements": [
                "Be more concise in explanations",
                "Always include runnable examples",
                "Prefer idiomatic patterns for the target language",
            ],
            "new_system_prompt": state["current_system_prompt"] + (
                "\nAlways include a brief summary of your approach before the code."
            ),
            "expected_impact": "Incremental quality improvement",
        }

    evolution_record = {
        "generation": state["generation"] + 1,
        "timestamp": datetime.utcnow().isoformat(),
        "old_prompt": state["current_system_prompt"],
        "new_prompt": evolution_data.get("new_system_prompt", state["current_system_prompt"]),
        "analysis": evolution_data.get("analysis", ""),
        "key_improvements": evolution_data.get("key_improvements", []),
        "expected_impact": evolution_data.get("expected_impact", ""),
        "metrics_at_evolution": {
            "total_requests": state["total_requests"],
            "success_rate": round(success_rate(state), 3),
        },
    }

    state["generation"] += 1
    state["current_system_prompt"] = evolution_record["new_prompt"]
    state["evolution_history"].append(evolution_record)
    state["learned_patterns"] = (
        evolution_data.get("key_improvements", []) + state.get("learned_patterns", [])
    )[:20]
    _save(state)

    if broadcast_fn:
        await broadcast_fn(f"✅ Evolution complete! Now at generation {state['generation']}")

    return evolution_record
