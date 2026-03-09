import os

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)

SESSION_ID = os.environ.get("SESSION_ID", "yixiong")

BASE_URL = f"https://redwoodscaffolds.com/session/{SESSION_ID}"

MODEL = "qwen/qwen3-235b-a22b-2507"

# Generation
NUM_GENERATIONS = 20          # Best-of-N: candidates per problem
GEN_TEMP_MIN = 0.7            # Temperature range for generation
GEN_TEMP_MAX = 1.2
USE_THINK_MODE = True         # Use /think for generation, /nothink for shortening
NUM_RETRY_GENERATIONS = 8    # Retry attempts when Phase 1 finds no correct solutions

# Shortening
NUM_SHORTEN_ROUNDS = 4        # Max refinement rounds
NUM_SHORTEN_ATTEMPTS = 8      # Parallel attempts per round
SHORTEN_TEMP_MIN = 0.3
SHORTEN_TEMP_MAX = 0.6
SHORTEN_STOP_ROUNDS = 2       # Stop after N consecutive rounds with no improvement
SHORTEN_TARGET_RATIO = 0.5    # Target length = current * ratio (prompt hint)

# Evolutionary crossover
NUM_EVOLVE_ROUNDS = 2
NUM_EVOLVE_ATTEMPTS = 8

# LLM
MAX_TOKENS = 2048             # Max output tokens per LLM call

# Rollouts
ROLLOUT_PROBLEM_COUNT = 192     # Number of problems to save rollout logs for

# Concurrency
MAX_SUBMIT_CONCURRENCY = 100
MAX_LLM_CONCURRENCY = 300
