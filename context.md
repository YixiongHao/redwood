# Your Task

Your task is to write a scaffold for Qwen3-235B-A22B-instruct that generates short C++ solutions to programming problems.

You can submit solutions for a problem (over an HTTP API) as many times as you like. You are awarded points based on the correctness and length of your best submitted solution on a test set.

## Scoring

- For each problem, if none of your submitted solutions are correct (or if you haven't submitted a solution yet), you get zero points.
- For correct solutions, we count the number of characters in your code and compare it to a simple baseline. You get 1 point if your code is 10% shorter, 4 points if it is 20% shorter, 9 points if it is 30% shorter etc.
- Specifically, your score is `(max(10 × percentage_decrease, 0))²` where `percentage_decrease = (baseline - your_length) / baseline`
- We will only count the highest scoring submission for each problem; so your score will be unaffected if you submit a solution longer than a previous one.

## Train Phase

For the first two hours of the game, you have access to a train set with 192 problems. The goal of this phase is to prepare for the test phase by finding scaffolds that produce the best solutions. We do not evaluate you on your score during the train phase.

You are allowed to use any AI tools to help you write your scaffolds. Please create a GitHub repo and commit your code to it regularly. We may read your code as part of our evaluation.

## Test Phase

After the train phase, you have 15 minutes to submit solutions to 64 new test problems. Do not modify your scaffolds during this phase - only run them. You should prepare your submission scripts before the test phase starts because you are not allowed to write code during this phase.

## Review Phase

After finishing the test, please write a Google Doc describing your overall strategy. There are more specific instructions in the "Review Instructions" tab. We will primarily evaluate you on your test score, but it is still useful to have a qualitative understanding of how you think. We recommend you to take 30-45 minutes on the Google Doc but we give you a soft 4-hour deadline so you do not feel rushed.

## LLM Calls

Please only use Qwen3 235B A22B Instruct through Openrouter. Specifically use `qwen/qwen3-235b-a22b-2507` via OpenRouter. Do not use other models or the base/thinking version.

Your key has $50, you are encouraged to use as much of it as possible! You will need to make many concurrent calls to the API in order to use non-trivial amounts of your budget. Note that openrouter does not have rate limits!

## The Dataset

The problems are of two difficulty levels: "introductory" and "intermediate". These two sets of problems have slightly different distributions, but one is not necessarily harder than the other (they are both selected by a filter that ensures Qwen3 235B A22B has an easy time solving them).

For each problem in the dataset, you will have access to a problem ID, a problem statement, the baseline length you have to beat and the difficulty level.

## Compiler and Runtime Details

- The code is compiled with `g++ -O3`
- The code is run with a timeout of 4 seconds.
- More details about the API are in the "API Documentation" tab

> **Note:** you can directly copy-paste this document into your AI tool and it will be able to set up the basic infrastructure for you!

---

# API Documentation

## OpenRouter

Please only use Qwen3 235B A22B Instruct through Openrouter. Here is an example on how to make a call to the model.

```python
import requests

OPENROUTER_API_KEY = "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"


def call_llm(prompt: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen/qwen3-235b-a22b-2507",
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    return response.json()["choices"][0]["message"]["content"]
```

You must use `qwen/qwen3-235b-a22b-2507` via OpenRouter. Do not use other models or the base/thinking version. Note that openrouter does not have rate limits!

## Code Golf API

You should retrieve coding problems and submit solutions through the following API. All endpoints are accessed through your session URL. See below for examples and documentation.

### Get Problems

```python
import requests

base_url = "https://redwoodscaffolds.com/session/{your_session_id}"

response = requests.get(f"{base_url}/get_train_set")
print(response.json())
```

Returns list of 192 training problems:

```json
{
    "id": 0,
    "problem_statement": "...",
    "baseline_length": 745.0,
    "difficulty": "intermediate"
}
```

Similarly, you should use `get_test_set` to get the 64 problems in the test set. Note that the test set endpoint will only work during the test phase.

### Submit Solutions

You can use this to submit a solution.

```python
import requests

response = requests.post(
    f"{base_url}/submit_train",
    json={
        "id": "...",       # Problem ID (0-191)
        "code": "..."      # Submit your code here
    }
)
print(response.json())
```

Returns:

```json
{
    "solution_length": "...",
    "score": "...",
    "correct": "..."
}
```

- `solution_length`: number of characters in your code
- `score`: points earned (0 if incorrect)
- `correct`: `true` if the output is correct, `false` if the output is incorrect or the code timed out or exited with a non-zero exit code

Similarly, you should use `/submit_test` in order to submit problems to the test set. This endpoint will only accept problems from ID 0 to 63. Note this endpoint will only work during the test phase.

> **IMPORTANT:** You will probably be bottlenecked on the submit solution as opposed to LLM inference from openrouter. The sweet spot is somewhere between 50-200 concurrents, going significantly higher (for example 1000) risks crashing the server which you want to avoid.
