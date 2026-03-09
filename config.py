import os

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)

SESSION_ID = os.environ.get("SESSION_ID", "yixiong")

BASE_URL = f"https://redwoodscaffolds.com/session/{SESSION_ID}"

MODEL = "qwen/qwen3-235b-a22b-2507"

# Concurrency settings
MAX_SUBMIT_CONCURRENCY = 100  # 50-200 sweet spot, avoid 1000+
MAX_LLM_CONCURRENCY = 50
