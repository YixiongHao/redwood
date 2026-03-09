import asyncio
import argparse
import json
from scaffold import run_train, run_test


async def main():
    parser = argparse.ArgumentParser(description="Code Golf Scaffold Runner")
    parser.add_argument("input", type=str,
                        help="Path to local JSON file with problems")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Which phase to run (default: train)")
    parser.add_argument("--problems", type=str, default=None,
                        help="Comma-separated problem IDs to run (default: all)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Load problems from local JSON
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
