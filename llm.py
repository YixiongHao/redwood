import re
import asyncio
import aiohttp
from config import OPENROUTER_API_KEY, MODEL, MAX_TOKENS


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from LLM responses."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def call_llm(
    session: aiohttp.ClientSession,
    prompt: str,
    temperature: float = 0.7,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Call Qwen3-235B via OpenRouter. Returns the assistant's message content.

    Retries up to 3 times with exponential backoff on transient errors.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or MAX_TOKENS,
    }

    last_error = None
    for attempt in range(3):
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            ) as resp:
                if resp.status >= 500:
                    last_error = f"HTTP {resp.status}: {await resp.text()}"
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status == 429:
                    last_error = "Rate limited"
                    await asyncio.sleep(2 ** attempt * 2)
                    continue
                data = await resp.json()
                if "choices" not in data or not data["choices"]:
                    last_error = f"No choices in response: {data}"
                    await asyncio.sleep(2 ** attempt)
                    continue
                return data["choices"][0]["message"]["content"]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = str(e)
            await asyncio.sleep(2 ** attempt)
            continue

    raise RuntimeError(f"LLM call failed after 3 retries: {last_error}")


def extract_code(response: str) -> str | None:
    """Extract C++ code from a markdown-fenced LLM response."""
    response = strip_think_blocks(response)

    try:
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
    except ValueError:
        return None
    return None
