"""
AI Coding Engine — wraps Claude for code generation, analysis, improvement, and explanation.
Streams tokens back through an async generator.
"""
import anthropic
from typing import AsyncGenerator
from evolution import get_system_prompt

client = anthropic.AsyncAnthropic()

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
        "and error conditions. Use appropriate testing framework for the language."
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
    """
    Stream a response from Claude for the given coding task.
    Yields text chunks as they arrive.
    """
    system = get_system_prompt()
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["generate"])

    system_full = f"{system}\n\nCurrent task mode: {mode_instruction}"
    if language and mode != "chat":
        system_full += f"\n\nTarget language: {language}"

    user_content = prompt
    if extra_context:
        user_content = f"Context:\n{extra_context}\n\nRequest:\n{prompt}"

    async with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system_full,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def quick_response(
    prompt: str,
    mode: str = "generate",
    language: str = "python",
) -> str:
    """Non-streaming version for internal use."""
    chunks = []
    async for chunk in stream_response(prompt, mode, language):
        chunks.append(chunk)
    return "".join(chunks)
