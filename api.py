import aiohttp
from config import BASE_URL


async def get_train_set(session: aiohttp.ClientSession) -> list[dict]:
    async with session.get(f"{BASE_URL}/get_train_set") as resp:
        return await resp.json()


async def get_test_set(session: aiohttp.ClientSession) -> list[dict]:
    async with session.get(f"{BASE_URL}/get_test_set") as resp:
        return await resp.json()


async def submit_train(session: aiohttp.ClientSession, problem_id: int, code: str) -> dict:
    async with session.post(
        f"{BASE_URL}/submit_train",
        json={"id": problem_id, "code": code},
    ) as resp:
        return await resp.json()


async def submit_test(session: aiohttp.ClientSession, problem_id: int, code: str) -> dict:
    async with session.post(
        f"{BASE_URL}/submit_test",
        json={"id": problem_id, "code": code},
    ) as resp:
        return await resp.json()
