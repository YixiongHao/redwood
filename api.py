import asyncio
import aiohttp
from config import BASE_URL


async def get_train_set(session: aiohttp.ClientSession) -> list[dict]:
    async with session.get(f"{BASE_URL}/get_train_set") as resp:
        return await resp.json()


async def get_test_set(session: aiohttp.ClientSession) -> list[dict]:
    async with session.get(f"{BASE_URL}/get_test_set") as resp:
        return await resp.json()


async def _submit_with_retry(
    session: aiohttp.ClientSession, url: str, payload: dict, retries: int = 3
) -> dict:
    """Submit with exponential backoff retry on transient errors."""
    last_error = None
    for attempt in range(retries):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status >= 500:
                    last_error = f"HTTP {resp.status}: {await resp.text()}"
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status == 429:
                    last_error = "Rate limited"
                    await asyncio.sleep(2 ** attempt * 2)
                    continue
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = str(e)
            await asyncio.sleep(2 ** attempt)
            continue
    raise RuntimeError(f"Submit failed after {retries} retries: {last_error}")


async def submit_train(session: aiohttp.ClientSession, problem_id: int, code: str) -> dict:
    return await _submit_with_retry(
        session, f"{BASE_URL}/submit_train", {"id": problem_id, "code": code}
    )


async def submit_test(session: aiohttp.ClientSession, problem_id: int, code: str) -> dict:
    return await _submit_with_retry(
        session, f"{BASE_URL}/submit_test", {"id": problem_id, "code": code}
    )
