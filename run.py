import asyncio
import argparse
import json
import os
import aiohttp
from scaffold import run_train, run_test
from api import get_train_set, get_test_set


async def fetch_problems(mode: str, path: str):
    """Download problems from the API and save to a local JSON file."""
    async with aiohttp.ClientSession() as session:
        if mode == "train":
            problems = await get_train_set(session)
        else:
            problems = await get_test_set(session)
    with open(path, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"Fetched {len(problems)} {mode} problems → {path}")
    return problems


async def main():
    parser = argparse.ArgumentParser(description="Code Golf Scaffold Runner")
    parser.add_argument("input", type=str, nargs="?", default=None,
                        help="Path to local JSON file with problems (auto-fetched if missing)")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Which phase to run (default: train)")
    parser.add_argument("--problems", type=str, default=None,
                        help="Comma-separated problem IDs to run (default: all)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Default file path based on mode
    if args.input is None:
        args.input = f"{args.mode}_problems.json"

    # Fetch from API if the file doesn't exist
    if not os.path.exists(args.input):
        print(f"{args.input} not found, fetching from API...")
        problems = await fetch_problems(args.mode, args.input)
    else:
        with open(args.input) as f:
            problems = json.load(f)

    # Filter to specific problems if requested
    if args.problems:
        ids = {int(x) for x in args.problems.split(",")}
        problems = [p for p in problems if p["id"] in ids]

    print(f"Running {args.mode} on {len(problems)} problems...")

    if args.mode == "train":
        results = await run_train(problems)
    else:
        results = await run_test(problems)

    if args.save:
        serializable = [r for r in results if isinstance(r, dict)]
        with open(args.save, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
