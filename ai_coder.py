"""
AI Coding Engine — uses OpenAI-compatible API (OpenAI, Ollama, Groq, Mistral, etc.)
Streams tokens back through an async generator.
"""
import os
from typing import AsyncGenerator
from openai import AsyncOpenAI
from evolution import get_system_prompt

def _client() -> AsyncOpenAI:
    base_url = os.getenv("OPENAI_BASE_URL") or None
    api_key  = os.getenv("OPENAI_API_KEY", "no-key")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

def _model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o")

MODE_INSTRUCTIONS = {
    "generate": (
        "Generate complete, working code based on the user's description. "
        "Include inline comments for non-obvious logic. "
        "Show usage examples at the end."
    ),
    "analyze": (
        "Analyze the provided code. Identify: bugs, performance issues, security concerns, "
        "style violations, and improvement opportunities. Be specific with line references."
    ),
    "improve": (
        "Improve the provided code. Fix bugs, optimize performance, enhance readability, "
        "add error handling. Show the improved version with a summary of changes."
    ),
    "explain": (
        "Explain the provided code in clear, beginner-friendly terms. "
        "Break down what each section does, and explain the overall logic flow."
    ),
    "debug": (
        "Debug the provided code. Identify the root cause of issues, explain why they occur, "
        "and provide the fixed version with explanation."
    ),
    "test": (
        "Write comprehensive tests for the provided code. Cover happy paths, edge cases, "
        "and error conditions. Use an appropriate testing framework for the language."
    ),
    "chat": (
        "Help with the user's coding question. Be concise, accurate, and practical. "
        "Provide code examples when helpful."
    ),
}


async def stream_response(
    prompt: str,
    mode: str = "generate",
    language: str = "python",
    extra_context: str = "",
) -> AsyncGenerator[str, None]:
    """Stream a response for the given coding task."""
    client = _client()
    base_system = get_system_prompt()
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["generate"])

    system = f"{base_system}\n\nTask mode: {mode_instruction}"
    if language and mode != "chat":
        system += f"\n\nTarget language: {language}"

    user_content = prompt
    if extra_context:
        user_content = f"Code context:\n```\n{extra_context}\n```\n\nRequest:\n{prompt}"

    stream = await client.chat.completions.create(
        model=_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        stream=True,
        temperature=0.2,
        max_tokens=4096,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def quick_response(prompt: str, mode: str = "generate", language: str = "python") -> str:
    """Non-streaming version for internal use."""
    chunks = []
    async for chunk in stream_response(prompt, mode, language):
        chunks.append(chunk)
    return "".join(chunks)
