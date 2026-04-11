"""
AI Coding Engine — local inference via CodeGPT, fully offline.
"""
from typing import AsyncGenerator
from llm.server import get_server
from evolution import get_system_prompt

MODE_INSTRUCTIONS = {
    "generate": (
        "Generate complete, working code based on the user's description. "
        "Include inline comments for non-obvious logic. Show usage examples."
    ),
    "analyze": (
        "Analyze the provided code. Identify bugs, performance issues, security concerns, "
        "and improvement opportunities. Be specific with line numbers."
    ),
    "improve": (
        "Improve the provided code. Fix bugs, optimize performance, enhance readability, "
        "add error handling. Show the improved version with a summary of changes."
    ),
    "explain": (
        "Explain the provided code in clear, beginner-friendly terms. "
        "Break down what each section does and explain the overall logic flow."
    ),
    "debug": (
        "Debug the provided code. Identify the root cause of issues, explain why "
        "they occur, and provide the fixed version with explanation."
    ),
    "test": (
        "Write comprehensive tests for the provided code. Cover happy paths, "
        "edge cases, and error conditions using an appropriate test framework."
    ),
    "chat": (
        "Answer the user's coding question. Be concise and practical. "
        "Provide runnable code examples when helpful."
    ),
}


async def stream_response(
    prompt: str,
    mode: str = "generate",
    language: str = "python",
    extra_context: str = "",
) -> AsyncGenerator[str, None]:
    server = get_server()
    system = get_system_prompt()
    instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["generate"])
    system = f"{system}\n\nTask: {instruction}"
    if language and mode != "chat":
        system += f"\nLanguage: {language}"

    user_content = prompt
    if extra_context:
        user_content = f"```\n{extra_context}\n```\n\n{prompt}"

    messages = [{"role": "user", "content": user_content}]

    async for chunk in server.stream_chat(messages, system=system):
        yield chunk
