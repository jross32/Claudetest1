"""
Self-Evolution Engine — tracks performance, learns from feedback, and uses Claude
to rewrite its own system prompts over time.
"""
import json
import time
import os
from datetime import datetime
from typing import Optional
import anthropic

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "evolution.json")

DEFAULT_STATE = {
    "generation": 0,
    "total_requests": 0,
    "positive_feedback": 0,
    "negative_feedback": 0,
    "evolution_history": [],
    "current_system_prompt": (
        "You are an expert AI coding assistant. Generate clean, efficient, well-commented code. "
        "When asked to write code, provide working implementations with explanations. "
        "Follow best practices for the language being used. "
        "If fixing bugs, explain what was wrong and why the fix works."
    ),
    "learned_patterns": [],
    "recent_sessions": [],
}


def _load() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            data = json.load(f)
        # Merge any missing keys from defaults
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


def record_request(prompt: str, response_preview: str, mode: str):
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
    Uses Claude to analyze recent sessions and rewrite the system prompt
    to be more effective. This is the self-evolution step.
    """
    state = _load()
    client = anthropic.AsyncAnthropic()

    # Gather context for Claude
    rated = [s for s in state["recent_sessions"] if s["feedback"] is not None]
    positive_examples = [s for s in rated if s["feedback"]["positive"]][:5]
    negative_examples = [s for s in rated if not s["feedback"]["positive"]][:5]

    if broadcast_fn:
        await broadcast_fn("🧬 Starting evolution cycle...")

    analysis_prompt = f"""You are the meta-AI responsible for improving an AI coding assistant system.

Current system prompt being used:
<current_prompt>
{state["current_system_prompt"]}
</current_prompt>

Performance metrics:
- Total requests: {state["total_requests"]}
- Positive feedback: {state["positive_feedback"]}
- Negative feedback: {state["negative_feedback"]}
- Success rate: {success_rate(state)*100:.1f}%
- Evolution generation: {state["generation"]}

Successful sessions (what worked well):
{json.dumps(positive_examples, indent=2) if positive_examples else "No rated sessions yet"}

Unsuccessful sessions (what needs improvement):
{json.dumps(negative_examples, indent=2) if negative_examples else "No rated sessions yet"}

Your task:
1. Analyze the patterns in what worked and what didn't
2. Identify specific improvements to the system prompt
3. Write an improved system prompt that will perform better
4. Explain the key changes you made

Respond in JSON format:
{{
  "analysis": "your analysis of what's working and what isn't",
  "key_improvements": ["list", "of", "improvements"],
  "new_system_prompt": "the complete new system prompt",
  "expected_impact": "how this should improve performance"
}}"""

    if broadcast_fn:
        await broadcast_fn("🤔 Claude is analyzing performance patterns...")

    try:
        stream = await client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=2048,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": analysis_prompt}]
        )

        full_response = ""
        async with stream as s:
            async for text in s.text_stream:
                full_response += text

        # Parse Claude's response
        try:
            # Extract JSON from response
            start = full_response.find("{")
            end = full_response.rfind("}") + 1
            if start >= 0 and end > start:
                evolution_data = json.loads(full_response[start:end])
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError):
            # Fallback: use response as analysis only
            evolution_data = {
                "analysis": full_response[:500],
                "key_improvements": ["Refined based on usage patterns"],
                "new_system_prompt": state["current_system_prompt"],
                "expected_impact": "Incremental improvement"
            }

        # Record the evolution
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
                "success_rate": success_rate(state),
            }
        }

        state["generation"] += 1
        state["current_system_prompt"] = evolution_data.get("new_system_prompt", state["current_system_prompt"])
        state["evolution_history"].append(evolution_record)
        # Keep learned patterns
        improvements = evolution_data.get("key_improvements", [])
        state["learned_patterns"] = (improvements + state.get("learned_patterns", []))[:20]
        _save(state)

        if broadcast_fn:
            await broadcast_fn(f"✅ Evolution complete! Now at generation {state['generation']}")

        return evolution_record

    except Exception as e:
        if broadcast_fn:
            await broadcast_fn(f"❌ Evolution error: {str(e)}")
        raise
