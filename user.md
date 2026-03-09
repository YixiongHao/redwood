# Code Golf Scaffold - Usage Guide

## Setup

### 1. Install uv (if not already installed)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Initialize the project

```bash
# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
```

### 3. Configure

Edit `config.py` and set your `SESSION_ID`:

```python
SESSION_ID = "your-actual-session-id"
```

Or set it via environment variable:

```bash
export SESSION_ID="your-actual-session-id"
export OPENROUTER_API_KEY="your-key"  # optional, default is already set
```

## Running

Always activate the venv first or prefix commands with `uv run`:

```bash
# Using uv run (no need to activate venv)
uv run python run.py --mode train

# Or activate venv first
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
python run.py --mode train
```

### Train phase

```bash
# Run on all 192 training problems
uv run python run.py --mode train

# Run on specific problems only
uv run python run.py --mode train --problems 0,1,2,3

# Save results to a file
uv run python run.py --mode train --save results.json
```

### Test phase

```bash
# Run on all 64 test problems (only works during test phase)
uv run python run.py --mode test --save test_results.json
```

## Module Reference

### `config.py`

Central configuration. All settings in one place:
- `OPENROUTER_API_KEY` - your OpenRouter key
- `SESSION_ID` - your session ID for the code golf API
- `MAX_SUBMIT_CONCURRENCY` - concurrent submissions (default 100, keep between 50-200)
- `MAX_LLM_CONCURRENCY` - concurrent LLM calls (default 50)

### `api.py`

Async API client functions:
- `get_train_set(session)` - fetch 192 training problems
- `get_test_set(session)` - fetch 64 test problems (test phase only)
- `submit_train(session, problem_id, code)` - submit a solution to training
- `submit_test(session, problem_id, code)` - submit a solution to test

### `llm.py`

LLM interaction:
- `call_llm(session, prompt, temperature)` - call Qwen3-235B via OpenRouter
- `extract_code(response)` - extract C++ code from markdown-fenced LLM response

### `scaffold.py`

Core pipeline:
- `generate_solution(session, problem, temperature)` - generate one C++ solution
- `shorten_solution(session, problem, code, temperature)` - ask LLM to shorten existing code
- `solve_problem(...)` - full pipeline for one problem (generate N candidates, submit, shorten best)
- `run_train(problems)` / `run_test(problems)` - run pipeline on a list of problems

### `run.py`

CLI entry point. Args:
- `input` - path to local JSON file with problems
- `--mode train|test` - which phase (default: train)
- `--problems 0,1,2` - specific problem IDs (default: all)
- `--save results.json` - save results to file

### `analyze.py`

Post-run diagnostics. Reads a rollout folder, finds worst-scoring problems, and diagnoses failure patterns.

```bash
# Analyze 10 worst problems (default)
uv run python analyze.py rollouts/20260309_143000

# Analyze 20 worst problems
uv run python analyze.py rollouts/20260309_143000 -n 20

# Save analysis to JSON
uv run python analyze.py rollouts/20260309_143000 -o analysis.json
```

Args:
- `folder` - path to rollout run folder (contains metrics.json + problem_N.json files)
- `-n / --top` - number of worst problems to analyze (default: 10)
- `-o / --output` - save analysis to JSON file

Diagnosed issues include: `NEVER_CORRECT`, `RETRY_SAVED`, `SHORTEN_NO_GAIN`, `EVOLVE_NO_GAIN`, `LOW_DIVERSITY`, `EXTRACT_FAIL`, `ERRORS`, `ALL_WRONG`, `SHORTEN_RATE`.

## Adding dependencies

```bash
# Add a new package
uv pip install <package>

# Freeze current deps
uv pip freeze > requirements.txt
```
