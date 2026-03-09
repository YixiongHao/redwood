import asyncio
import json
import os
import aiohttp
from config import MAX_LLM_CONCURRENCY, MAX_SUBMIT_CONCURRENCY
from api import submit_train, submit_test
from llm import call_llm, extract_code

ROLLOUTS_DIR = "rollouts"
ROLLOUT_PROBLEM_COUNT = 5

SYSTEM_PROMPT = """\
You are an expert C++ code golfer. Write the shortest possible C++ solution to the given programming problem.

Rules:
- The code must compile with g++ -O3 and run within 4 seconds.
- Minimize character count aggressively: use short variable names, omit unnecessary whitespace, use macros, combine statements, etc.
- Output ONLY the code in a single ```cpp block. No explanation."""

SHORTEN_PROMPT = """\
Here is a correct C++ solution to a programming problem. Make it shorter (fewer characters) while keeping it correct.

Problem:
{problem_statement}

Current solution ({current_length} chars):
```cpp
{code}
```

Shorten it as much as possible. Use every code golf trick: short var names, macros, combined statements, remove unnecessary whitespace/newlines, use scanf/printf instead of cin/cout if shorter, etc.
Output ONLY the shortened code in a single ```cpp block."""


async def generate_solution(
    session: aiohttp.ClientSession,
    problem: dict,
    temperature: float = 0.7,
) -> tuple[str, str, str | None]:
    """Generate a single C++ solution. Returns (prompt, raw_response, extracted_code)."""
    prompt = f"{SYSTEM_PROMPT}\n\nProblem:\n{problem['problem_statement']}"
    response = await call_llm(session, prompt, temperature=temperature)
    return prompt, response, extract_code(response)


async def shorten_solution(
    session: aiohttp.ClientSession,
    problem: dict,
    code: str,
    temperature: float = 0.7,
) -> tuple[str, str, str | None]:
    """Ask the LLM to shorten an existing correct solution. Returns (prompt, raw_response, extracted_code)."""
    prompt = SHORTEN_PROMPT.format(
        problem_statement=problem["problem_statement"],
        code=code,
        current_length=len(code),
    )
    response = await call_llm(session, prompt, temperature=temperature)
    return prompt, response, extract_code(response)


def save_rollout(pid: int, rollout_entries: list[dict]):
    """Save rollout log for a problem to the rollouts folder."""
    os.makedirs(ROLLOUTS_DIR, exist_ok=True)
    path = os.path.join(ROLLOUTS_DIR, f"problem_{pid}.json")
    with open(path, "w") as f:
        json.dump(rollout_entries, f, indent=2)


async def solve_problem(
    session: aiohttp.ClientSession,
    problem: dict,
    submit_fn,
    sem_llm: asyncio.Semaphore,
    sem_submit: asyncio.Semaphore,
    num_attempts: int = 5,
    num_shorten: int = 3,
    log_rollout: bool = False,
) -> dict:
    """Full pipeline for one problem: generate candidates, submit, shorten best."""
    pid = problem["id"]
    baseline = problem["baseline_length"]
    best_score = 0
    best_code = None
    rollout = []

    # Phase 1: Generate multiple candidates in parallel
    async def attempt(temp):
        async with sem_llm:
            prompt, response, code = await generate_solution(session, problem, temperature=temp)
        result = None
        if code:
            async with sem_submit:
                result = await submit_fn(session, pid, code)
        return prompt, response, code, result

    temps = [0.5 + 0.1 * i for i in range(num_attempts)]
    tasks = [asyncio.create_task(attempt(t)) for t in temps]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            if log_rollout:
                rollout.append({"phase": "generate", "attempt": i, "error": str(r)})
            continue
        prompt, response, code, result = r
        if log_rollout:
            rollout.append({
                "phase": "generate",
                "attempt": i,
                "temperature": temps[i],
                "prompt": prompt,
                "response": response,
                "extracted_code": code,
                "submit_result": result,
            })
        if code and result and result.get("correct") and result.get("score", 0) >= best_score:
            best_score = result["score"]
            best_code = code

    # Phase 2: Iteratively shorten the best correct solution
    if best_code:
        for i in range(num_shorten):
            async with sem_llm:
                prompt, response, shorter = await shorten_solution(
                    session, problem, best_code, temperature=0.4 + 0.1 * i
                )
            if log_rollout:
                rollout.append({
                    "phase": "shorten",
                    "attempt": i,
                    "temperature": 0.4 + 0.1 * i,
                    "prompt": prompt,
                    "response": response,
                    "extracted_code": shorter,
                })
            if not shorter or len(shorter) >= len(best_code):
                continue
            async with sem_submit:
                result = await submit_fn(session, pid, shorter)
            if log_rollout:
                rollout[-1]["submit_result"] = result
            if result.get("correct"):
                best_code = shorter
                best_score = max(best_score, result.get("score", 0))

    if log_rollout:
        save_rollout(pid, rollout)

    print(f"Problem {pid}: score={best_score}, len={len(best_code) if best_code else 'N/A'}, baseline={baseline}")
    return {"id": pid, "score": best_score, "code": best_code}


async def run_train(problems: list[dict]):
    """Run the scaffold on training problems."""
    sem_llm = asyncio.Semaphore(MAX_LLM_CONCURRENCY)
    sem_submit = asyncio.Semaphore(MAX_SUBMIT_CONCURRENCY)

    # Sort by id so the first N are deterministic
    problems_sorted = sorted(problems, key=lambda p: p["id"])
    rollout_ids = {p["id"] for p in problems_sorted[:ROLLOUT_PROBLEM_COUNT]}

    async with aiohttp.ClientSession() as session:
        tasks = [
            solve_problem(
                session, p, submit_train, sem_llm, sem_submit,
                log_rollout=(p["id"] in rollout_ids),
            )
            for p in problems
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    scores = [r["score"] for r in results if isinstance(r, dict)]
    print(f"\nTotal score: {sum(scores)}")
    print(f"Problems solved: {sum(1 for s in scores if s > 0)}/{len(problems)}")
    return results


async def run_test(problems: list[dict]):
    """Run the scaffold on test problems."""
    sem_llm = asyncio.Semaphore(MAX_LLM_CONCURRENCY)
    sem_submit = asyncio.Semaphore(MAX_SUBMIT_CONCURRENCY)

    problems_sorted = sorted(problems, key=lambda p: p["id"])
    rollout_ids = {p["id"] for p in problems_sorted[:ROLLOUT_PROBLEM_COUNT]}

    async with aiohttp.ClientSession() as session:
        tasks = [
            solve_problem(
                session, p, submit_test, sem_llm, sem_submit,
                log_rollout=(p["id"] in rollout_ids),
            )
            for p in problems
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    scores = [r["score"] for r in results if isinstance(r, dict)]
    print(f"\nTotal score: {sum(scores)}")
    print(f"Problems solved: {sum(1 for s in scores if s > 0)}/{len(problems)}")
    return results
