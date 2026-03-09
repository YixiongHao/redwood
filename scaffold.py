import asyncio
import json
import os
import random
import re
import time
import aiohttp
from config import (
    MAX_LLM_CONCURRENCY,
    MAX_SUBMIT_CONCURRENCY,
    NUM_GENERATIONS,
    GEN_TEMP_MIN,
    GEN_TEMP_MAX,
    USE_THINK_MODE,
    NUM_RETRY_GENERATIONS,
    NUM_SHORTEN_ROUNDS,
    NUM_SHORTEN_ATTEMPTS,
    SHORTEN_TEMP_MIN,
    SHORTEN_TEMP_MAX,
    SHORTEN_STOP_ROUNDS,
    SHORTEN_TARGET_RATIO,
    NUM_EVOLVE_ROUNDS,
    NUM_EVOLVE_ATTEMPTS,
    ROLLOUT_PROBLEM_COUNT,
)
from api import submit_train, submit_test
from llm import call_llm, extract_code

ROLLOUTS_DIR = "rollouts"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert competitive programmer. Your PRIMARY goal is to write a CORRECT solution. \
An incorrect solution is worthless regardless of length. Your secondary goal is to keep the code short.

Rules:
- The code MUST be correct on all inputs, including edge cases.
- It must compile with g++ -O3 and run within 4 seconds.
- Use #include<bits/stdc++.h> and using namespace std; for brevity.
- Keep variable names short and code compact, but never at the expense of correctness.
- Output ONLY the code in a single ```cpp block. No explanation."""

PROMPT_VARIANTS = [
    # Variant 0: C-style focus
    """\
Solve this competitive programming problem in the shortest C++ code possible.
Prefer C-style I/O (scanf/printf), macros, and minimal whitespace.
The code must compile with g++ -O3 and run within 4 seconds.
Output ONLY the code in a single ```cpp block.

Problem:
{problem_statement}""",

    # Variant 1: Algorithm-focused
    """\
Solve this problem with the most efficient and shortest C++ code.
Choose the algorithm that leads to the shortest implementation.
Must compile with g++ -O3 and run within 4 seconds.
Output ONLY the code in a single ```cpp block.

Problem:
{problem_statement}""",

    # Variant 2: Extreme brevity
    """\
Write the absolute shortest C++ solution to this problem. Every character counts.
Use #define for repeated tokens, comma operators, ternary expressions.
Omit all unnecessary whitespace, braces, and semicolons.
Must compile with g++ -O3, run in 4s.
```cpp block only. No explanation.

Problem:
{problem_statement}""",

    # Variant 3: Direct
    """\
Shortest correct C++ for this problem. Golf it hard.
Must compile g++ -O3, run <4s. ```cpp block only.

Problem:
{problem_statement}""",
]

SHORTEN_PROMPT = """\
Shorten this correct C++ solution. Current: {current_length} chars. Target: under {target_length} chars.  The length of the code is absolutely critical to our mission and you must strive to reduce it as much as possible, even if it requires more complex golfing tricks.  Above all, ensure that the code remains correct and runs under 4s.

Problem:
{problem_statement}

Current solution:
```cpp
{code}
```

You should go through this checklist and implement all reasonable optimizations (with examples):
- Identifier renaming: int count=0; → int c=0;
- Brace removal around single statements: for(i=0;i<n;i++){{a++;}} → for(i=0;i<n;i++)a++;
- Remove return 0 from main: int main(){{...;return 0;}} → int main(){{...;}}
- Collapse onto fewer lines: remove all newlines between non-preprocessor statements
- Whitespace removal: int a = b + c; → int a=b+c;
- Ternary replacement: if(x>0)a=1;else a=2; → a=x>0?1:2;
- Comma operator to eliminate braces: if(x){{a=1;f();}} → if(x)a=1,f();
- for absorption: i=0;while(i<n){{f();i++;}} → for(i=0;i<n;i++)f();
- Short-circuit instead of if: if(x)f(); → x&&f();
- Comparison shrinking: a!=b → a-b (boolean context), a==0 → !a
- ASCII values instead of char literals: 'A' → 65, ' ' → 32, '\n' → 10
- Scientific notation: 1000000 → 1e6
- Absorb increment into expressions: a=b;b++; → a=b++;
- puts over printf for string+newline: printf("foo\n") → puts("foo")
- Collapse declarations: int a;int b;int c; → int a,b,c;
- Merge loops where possible
- Remove unnecessary #includes
- Drop "using namespace std;" if only using scanf/printf (saves 18 chars)
- Use char arrays with scanf/printf instead of string with cin/cout: string s;cin>>s;...s.begin(),s.end() → char s[N];scanf("%s",s);...s,s+n (much shorter)
- Remove unused #define macros — only add a macro if it actually saves chars overall

Output ONLY the shortened code in a single ```cpp block."""

EVOLVE_PROMPT = """\
Here are {n} different correct C++ solutions to the same problem. Identify and combine the best tricks from each to create an even shorter solution.

Problem:
{problem_statement}

{solutions_text}

Create the shortest possible correct solution by combining the best ideas from all solutions above.
Output ONLY the code in a single ```cpp block."""

RETRY_PROMPT = """\
Solve this competitive programming problem in the shortest correct C++ code.
Previous attempts were all INCORRECT. Here are some of the wrong solutions — learn from their mistakes and avoid them:

{wrong_solutions_text}

Problem:
{problem_statement}

Write a correct and short C++ solution. Focus on CORRECTNESS first, then brevity.
Must compile with g++ -O3, run in 4s. Output ONLY the code in a single ```cpp block."""


# ---------------------------------------------------------------------------
# Deterministic minifier
# ---------------------------------------------------------------------------

def minify_cpp(code: str) -> str:
    """Deterministic C++ minifier: strip comments, collapse whitespace, remove
    unnecessary spaces around operators. Careful with string literals and
    preprocessor directives."""
    if not code:
        return code

    result = []
    i = 0
    in_string = False
    string_char = None
    in_line_comment = False
    in_block_comment = False
    in_preprocessor = False

    # First pass: strip comments, preserve strings and preprocessor directives
    while i < len(code):
        c = code[i]

        # --- inside a block comment ---
        if in_block_comment:
            if c == '*' and i + 1 < len(code) and code[i + 1] == '/':
                in_block_comment = False
                i += 2
                result.append(' ')  # replace comment with single space
            else:
                i += 1
            continue

        # --- inside a line comment ---
        if in_line_comment:
            if c == '\n':
                in_line_comment = False
                result.append('\n')
            i += 1
            continue

        # --- inside a string/char literal ---
        if in_string:
            result.append(c)
            if c == '\\' and i + 1 < len(code):
                result.append(code[i + 1])
                i += 2
                continue
            if c == string_char:
                in_string = False
            i += 1
            continue

        # --- check for comment start ---
        if c == '/' and i + 1 < len(code):
            if code[i + 1] == '/':
                in_line_comment = True
                i += 2
                continue
            if code[i + 1] == '*':
                in_block_comment = True
                i += 2
                continue

        # --- check for string/char literal start ---
        if c in ('"', "'"):
            in_string = True
            string_char = c
            result.append(c)
            i += 1
            continue

        # --- track preprocessor directives ---
        if c == '#':
            in_preprocessor = True

        if c == '\n' and in_preprocessor:
            in_preprocessor = False
            result.append('\n')
            i += 1
            continue

        result.append(c)
        i += 1

    text = ''.join(result)

    # Split into lines, process preprocessor vs code lines separately
    lines = text.split('\n')
    processed = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            # Preprocessor: collapse internal whitespace but keep on own line
            processed.append(re.sub(r'[ \t]+', ' ', stripped))
        else:
            processed.append(stripped)

    # Rejoin: preprocessor directives need newlines, code lines don't
    parts = []
    for line in processed:
        if line.startswith('#'):
            if parts and not parts[-1].endswith('\n'):
                parts.append('\n')
            parts.append(line)
            parts.append('\n')
        else:
            # If previous was preprocessor, already has newline
            parts.append(line)

    text = ''.join(parts)

    # Second pass: collapse whitespace in non-string, non-preprocessor regions
    result2 = []
    i = 0
    in_string = False
    string_char = None

    while i < len(text):
        c = text[i]

        if in_string:
            result2.append(c)
            if c == '\\' and i + 1 < len(text):
                result2.append(text[i + 1])
                i += 2
                continue
            if c == string_char:
                in_string = False
            i += 1
            continue

        if c in ('"', "'"):
            in_string = True
            string_char = c
            result2.append(c)
            i += 1
            continue

        # Preserve newlines around preprocessor lines
        if c == '\n':
            result2.append(c)
            i += 1
            continue

        # Collapse spaces/tabs
        if c in (' ', '\t'):
            # Look at neighbors to decide if space is needed
            # Space is needed between two alphanumeric/underscore chars
            prev = result2[-1] if result2 else ''
            # Skip whitespace, find next non-space char
            j = i + 1
            while j < len(text) and text[j] in (' ', '\t'):
                j += 1
            nxt = text[j] if j < len(text) else ''

            prev_alnum = prev.isalnum() or prev == '_'
            next_alnum = nxt.isalnum() or nxt == '_' or nxt == '#'

            if prev_alnum and next_alnum:
                result2.append(' ')
            # else: skip the space
            i = j
            continue

        result2.append(c)
        i += 1

    text = ''.join(result2).strip()

    # Clean up any remaining double newlines
    while '\n\n' in text:
        text = text.replace('\n\n', '\n')

    return text


# ---------------------------------------------------------------------------
# Solution helpers
# ---------------------------------------------------------------------------

def _has_main(code: str) -> bool:
    """Quick check that code looks like it has a main function."""
    return 'main' in code


def _spread_temps(n: int, lo: float, hi: float) -> list[float]:
    """Generate n temperatures evenly spread between lo and hi."""
    if n == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def solve_problem(
    session: aiohttp.ClientSession,
    problem: dict,
    submit_fn,
    sem_llm: asyncio.Semaphore,
    sem_submit: asyncio.Semaphore,
    log_rollout: bool = False,
    run_dir: str = "",
) -> dict:
    """Full pipeline for one problem: generate, shorten, evolve."""
    pid = problem["id"]
    baseline = problem["baseline_length"]
    best_score = 0
    best_code = None
    best_len = float("inf")
    correct_solutions: dict[str, int] = {}  # code -> length, for crossover
    incorrect_codes: list[str] = []  # for retry phase
    rollout = []

    think_suffix = " /think" if USE_THINK_MODE else ""
    nothink_suffix = " /nothink" if USE_THINK_MODE else ""

    # ---- Phase 1: Best-of-N generation ----

    async def gen_attempt(variant_idx: int, temp: float, attempt_id: int):
        prompt_template = PROMPT_VARIANTS[variant_idx % len(PROMPT_VARIANTS)]
        prompt = prompt_template.format(problem_statement=problem["problem_statement"])
        prompt += think_suffix

        async with sem_llm:
            response = await call_llm(
                session, prompt, temperature=temp, system_prompt=SYSTEM_PROMPT,
            )
        code = extract_code(response)
        if not code:
            return attempt_id, prompt, response, None, None, temp

        code = minify_cpp(code)
        if not code or not _has_main(code):
            return attempt_id, prompt, response, code, None, temp

        async with sem_submit:
            result = await submit_fn(session, pid, code)
        return attempt_id, prompt, response, code, result, temp

    temps = _spread_temps(NUM_GENERATIONS, GEN_TEMP_MIN, GEN_TEMP_MAX)
    tasks = [
        asyncio.create_task(gen_attempt(i, temps[i], i))
        for i in range(NUM_GENERATIONS)
    ]

    for coro in asyncio.as_completed(tasks):
        try:
            attempt_id, prompt, response, code, result, temp = await coro
        except Exception as e:
            if log_rollout:
                rollout.append({"phase": "generate", "error": str(e)})
            continue

        if log_rollout:
            rollout.append({
                "phase": "generate",
                "attempt": attempt_id,
                "temperature": temp,
                "prompt": prompt[:200],
                "extracted_code": code,
                "code_len": len(code) if code else None,
                "submit_result": result,
            })

        if code and result and result.get("correct"):
            code_len = len(code)
            correct_solutions[code] = code_len
            if result.get("score", 0) > best_score or (
                result.get("score", 0) == best_score and code_len < best_len
            ):
                best_score = result["score"]
                best_code = code
                best_len = code_len
        elif code:
            incorrect_codes.append(code)

    # Snapshot after Phase 1
    p1_score, p1_len = best_score, (best_len if best_code else None)

    # ---- Phase 1b: Retry with incorrect solutions as negative examples ----

    if not best_code and NUM_RETRY_GENERATIONS > 0 and incorrect_codes:
        sampled = random.sample(incorrect_codes, min(5, len(incorrect_codes)))
        wrong_solutions_text = "\n".join(
            f"Wrong solution {i+1}:\n```cpp\n{code}\n```"
            for i, code in enumerate(sampled)
        )

        async def retry_attempt(temp: float, attempt_id: int):
            prompt = RETRY_PROMPT.format(
                problem_statement=problem["problem_statement"],
                wrong_solutions_text=wrong_solutions_text,
            )
            prompt += think_suffix

            async with sem_llm:
                response = await call_llm(
                    session, prompt, temperature=temp, system_prompt=SYSTEM_PROMPT,
                )
            code = extract_code(response)
            if not code:
                return attempt_id, prompt, response, None, None, temp

            code = minify_cpp(code)
            if not code or not _has_main(code):
                return attempt_id, prompt, response, code, None, temp

            async with sem_submit:
                result = await submit_fn(session, pid, code)
            return attempt_id, prompt, response, code, result, temp

        temps_r = _spread_temps(NUM_RETRY_GENERATIONS, GEN_TEMP_MIN, GEN_TEMP_MAX)
        retry_tasks = [
            asyncio.create_task(retry_attempt(temps_r[j], j))
            for j in range(NUM_RETRY_GENERATIONS)
        ]

        for coro in asyncio.as_completed(retry_tasks):
            try:
                attempt_id, prompt, response, code, result, temp = await coro
            except Exception as e:
                if log_rollout:
                    rollout.append({"phase": "retry", "error": str(e)})
                continue

            if log_rollout:
                rollout.append({
                    "phase": "retry",
                    "attempt": attempt_id,
                    "temperature": temp,
                    "prompt": prompt[:200],
                    "extracted_code": code,
                    "code_len": len(code) if code else None,
                    "submit_result": result,
                })

            if code and result and result.get("correct"):
                code_len = len(code)
                correct_solutions[code] = code_len
                if result.get("score", 0) > best_score or (
                    result.get("score", 0) == best_score and code_len < best_len
                ):
                    best_score = result["score"]
                    best_code = code
                    best_len = code_len

    # Snapshot after Phase 1b
    p1b_score, p1b_len = best_score, (best_len if best_code else None)

    if log_rollout:
        rollout.append({
            "phase": "selected_for_shorten",
            "code": best_code,
            "code_len": best_len if best_code else None,
            "score": best_score,
        })

    # ---- Phase 2: Parallel iterative refinement ----

    if best_code:
        no_improvement_rounds = 0
        for round_idx in range(NUM_SHORTEN_ROUNDS):
            if no_improvement_rounds >= SHORTEN_STOP_ROUNDS:
                break

            round_best_before = best_len
            target = int(best_len * SHORTEN_TARGET_RATIO)

            async def shorten_attempt(temp: float, attempt_id: int):
                prompt = SHORTEN_PROMPT.format(
                    problem_statement=problem["problem_statement"],
                    code=best_code,
                    current_length=best_len,
                    target_length=target,
                )
                prompt += nothink_suffix

                async with sem_llm:
                    response = await call_llm(
                        session, prompt, temperature=temp, system_prompt=SYSTEM_PROMPT,
                    )
                code = extract_code(response)
                if not code:
                    return attempt_id, prompt, response, None, None, temp

                code = minify_cpp(code)
                if not code or not _has_main(code) or len(code) >= best_len:
                    return attempt_id, prompt, response, code, None, temp

                async with sem_submit:
                    result = await submit_fn(session, pid, code)
                return attempt_id, prompt, response, code, result, temp

            temps_s = _spread_temps(NUM_SHORTEN_ATTEMPTS, SHORTEN_TEMP_MIN, SHORTEN_TEMP_MAX)
            shorten_tasks = [
                asyncio.create_task(shorten_attempt(temps_s[j], j))
                for j in range(NUM_SHORTEN_ATTEMPTS)
            ]

            for coro in asyncio.as_completed(shorten_tasks):
                try:
                    attempt_id, prompt, response, code, result, temp = await coro
                except Exception as e:
                    if log_rollout:
                        rollout.append({"phase": "shorten", "round": round_idx, "error": str(e)})
                    continue

                if log_rollout:
                    rollout.append({
                        "phase": "shorten",
                        "round": round_idx,
                        "attempt": attempt_id,
                        "temperature": temp,
                        "extracted_code": code,
                        "code_len": len(code) if code else None,
                        "submit_result": result,
                    })

                if code and result and result.get("correct"):
                    code_len = len(code)
                    correct_solutions[code] = code_len
                    if code_len < best_len:
                        best_code = code
                        best_len = code_len
                        best_score = max(best_score, result.get("score", 0))

            if best_len >= round_best_before:
                no_improvement_rounds += 1
            else:
                no_improvement_rounds = 0

    # Snapshot after Phase 2
    p2_score, p2_len = best_score, (best_len if best_code else None)

    # ---- Phase 3: Evolutionary crossover ----

    if len(correct_solutions) >= 2:
        # Pick top 3 shortest distinct solutions
        sorted_sols = sorted(correct_solutions.items(), key=lambda x: x[1])[:3]

        if log_rollout:
            rollout.append({
                "phase": "selected_for_evolve",
                "solutions": [
                    {"code": code, "code_len": length}
                    for code, length in sorted_sols
                ],
            })

        for evolve_round in range(NUM_EVOLVE_ROUNDS):
            solutions_text = "\n".join(
                f"Solution {i+1} ({length} chars):\n```cpp\n{code}\n```"
                for i, (code, length) in enumerate(sorted_sols)
            )

            async def evolve_attempt(temp: float, attempt_id: int):
                prompt = EVOLVE_PROMPT.format(
                    n=len(sorted_sols),
                    problem_statement=problem["problem_statement"],
                    solutions_text=solutions_text,
                )
                prompt += nothink_suffix

                async with sem_llm:
                    response = await call_llm(
                        session, prompt, temperature=temp, system_prompt=SYSTEM_PROMPT,
                    )
                code = extract_code(response)
                if not code:
                    return attempt_id, prompt, response, None, None, temp

                code = minify_cpp(code)
                if not code or not _has_main(code) or len(code) >= best_len:
                    return attempt_id, prompt, response, code, None, temp

                async with sem_submit:
                    result = await submit_fn(session, pid, code)
                return attempt_id, prompt, response, code, result, temp

            temps_e = _spread_temps(NUM_EVOLVE_ATTEMPTS, SHORTEN_TEMP_MIN, SHORTEN_TEMP_MAX)
            evolve_tasks = [
                asyncio.create_task(evolve_attempt(temps_e[j], j))
                for j in range(NUM_EVOLVE_ATTEMPTS)
            ]

            for coro in asyncio.as_completed(evolve_tasks):
                try:
                    attempt_id, prompt, response, code, result, temp = await coro
                except Exception as e:
                    if log_rollout:
                        rollout.append({"phase": "evolve", "round": evolve_round, "error": str(e)})
                    continue

                if log_rollout:
                    rollout.append({
                        "phase": "evolve",
                        "round": evolve_round,
                        "attempt": attempt_id,
                        "temperature": temp,
                        "extracted_code": code,
                        "code_len": len(code) if code else None,
                        "submit_result": result,
                    })

                if code and result and result.get("correct"):
                    code_len = len(code)
                    if code_len < best_len:
                        best_code = code
                        best_len = code_len
                        best_score = max(best_score, result.get("score", 0))

    if log_rollout:
        rollout.append({
            "phase": "final_best",
            "code": best_code,
            "code_len": best_len if best_code else None,
            "score": best_score,
        })
        save_rollout(pid, rollout, run_dir)

    # Snapshot after Phase 3
    p3_score, p3_len = best_score, (best_len if best_code else None)

    def _fmt(s, l):
        return f"score={s},len={l if l else 'N/A'}"

    print(
        f"Problem {pid}: "
        f"P1[{_fmt(p1_score, p1_len)}] "
        f"P1b[{_fmt(p1b_score, p1b_len)}] "
        f"P2[{_fmt(p2_score, p2_len)}] "
        f"P3[{_fmt(p3_score, p3_len)}] "
        f"baseline={baseline}, "
        f"correct_variants={len(correct_solutions)}"
    )
    return {
        "id": pid,
        "score": best_score,
        "code": best_code,
        "baseline": baseline,
        "correct_variants": len(correct_solutions),
        "phases": {
            "p1": {"score": p1_score, "len": p1_len},
            "p1b": {"score": p1b_score, "len": p1b_len},
            "p2": {"score": p2_score, "len": p2_len},
            "p3": {"score": p3_score, "len": p3_len},
        },
    }


def save_rollout(pid: int, rollout_entries: list[dict], run_dir: str):
    """Save rollout log for a problem into the run's subfolder."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"problem_{pid}.json")
    with open(path, "w") as f:
        json.dump(rollout_entries, f, indent=2)


async def _run_problems(problems: list[dict], submit_fn, label: str):
    """Run the scaffold on a list of problems."""
    from datetime import datetime
    sem_llm = asyncio.Semaphore(MAX_LLM_CONCURRENCY)
    sem_submit = asyncio.Semaphore(MAX_SUBMIT_CONCURRENCY)

    problems_sorted = sorted(problems, key=lambda p: p["id"])
    rollout_ids = {p["id"] for p in problems_sorted[:ROLLOUT_PROBLEM_COUNT]}

    # Create a timestamped subfolder for this run's rollouts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ROLLOUTS_DIR, ts)

    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            solve_problem(
                session, p, submit_fn, sem_llm, sem_submit,
                log_rollout=(p["id"] in rollout_ids),
                run_dir=run_dir,
            )
            for p in problems
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - t0
    valid = [r for r in results if isinstance(r, dict) and "score" in r]
    scores = [r["score"] for r in valid]
    print(f"\n[{label}] Total score: {sum(scores)}")
    print(f"[{label}] Problems solved: {sum(1 for s in scores if s > 0)}/{len(problems)}")
    print(f"[{label}] Elapsed: {elapsed:.1f}s")

    # --- Phase-on-phase gain report ---
    phase_pairs = [("p1", "p1b"), ("p1b", "p2"), ("p2", "p3")]
    for prev_phase, cur_phase in phase_pairs:
        gains = []
        improved = 0
        for r in valid:
            phases = r.get("phases", {})
            prev_s = phases.get(prev_phase, {}).get("score", 0)
            cur_s = phases.get(cur_phase, {}).get("score", 0)
            gain = cur_s - prev_s
            gains.append(gain)
            if gain > 0:
                improved += 1
        avg_gain = sum(gains) / len(gains) if gains else 0
        total_gain = sum(gains)
        print(
            f"[{label}] {prev_phase}→{cur_phase}: "
            f"avg_gain={avg_gain:.2f}, total_gain={total_gain:.1f}, "
            f"improved={improved}/{len(gains)}"
        )

    # --- Save metrics.json ---
    metrics = {
        "label": label,
        "elapsed_s": round(elapsed, 1),
        "total_score": sum(scores),
        "problems_solved": sum(1 for s in scores if s > 0),
        "problems_total": len(problems),
        "problems": [
            {
                "id": r["id"],
                "score": r["score"],
                "baseline": r.get("baseline"),
                "correct_variants": r.get("correct_variants", 0),
                "phases": r.get("phases", {}),
            }
            for r in valid
        ],
    }
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{label}] Metrics saved to {metrics_path}")

    return results


async def run_train(problems: list[dict]):
    """Run the scaffold on training problems."""
    return await _run_problems(problems, submit_train, "Train")


async def run_test(problems: list[dict]):
    """Run the scaffold on test problems."""
    return await _run_problems(problems, submit_test, "Test")
