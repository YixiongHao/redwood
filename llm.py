import aiohttp
from config import OPENROUTER_API_KEY, MODEL


async def call_llm(session: aiohttp.ClientSession, prompt: str, temperature: float = 0.7) -> str:
    """Call Qwen3-235B via OpenRouter. Returns the assistant's message content."""
    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
    ) as resp:
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


def extract_code(response: str) -> str | None:
    """Extract C++ code from a markdown-fenced LLM response."""
    # Try to find ```cpp ... ``` block
    for lang in ("cpp", "c++", "c"):
        marker = f"```{lang}"
        if marker in response.lower():
            start = response.lower().index(marker) + len(marker)
            end = response.index("```", start)
            return response[start:end].strip()
    # Fallback: try generic ``` block
    if "```" in response:
        start = response.index("```") + 3
        # Skip optional language tag on same line
        newline = response.index("\n", start)
        start = newline + 1
        end = response.index("```", start)
        return response[start:end].strip()
    return None
