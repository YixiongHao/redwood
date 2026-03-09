"""Analyze a rollout folder: find worst-scoring problems and diagnose why."""
import argparse
import json
import os


def load_run(folder: str) -> tuple[dict, dict[int, list]]:
    """Load metrics.json and all problem rollout files from a run folder.
    Returns (metrics_dict, {problem_id: rollout_entries}).
    """
    metrics_path = os.path.join(folder, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No metrics.json in {folder}")

    with open(metrics_path) as f:
        metrics = json.load(f)

    rollouts = {}
    for fname in os.listdir(folder):
        if fname.startswith("problem_") and fname.endswith(".json"):
            pid = int(fname.replace("problem_", "").replace(".json", ""))
            with open(os.path.join(folder, fname)) as f:
                rollouts[pid] = json.load(f)

    return metrics, rollouts


def diagnose_problem(problem_metrics: dict, rollout: list[dict] | None) -> dict:
    """Analyze a single problem's rollout to diagnose poor performance."""
    pid = problem_metrics["id"]
    score = problem_metrics["score"]
    baseline = problem_metrics.get("baseline")
    phases = problem_metrics.get("phases", {})
    correct_variants = problem_metrics.get("correct_variants", 0)

    diagnosis = {
        "id": pid,
        "score": score,
        "baseline": baseline,
        "correct_variants": correct_variants,
        "phases": phases,
        "issues": [],
    }

    # Check if no correct solution was ever found
    p1_score = phases.get("p1", {}).get("score", 0)
    p1b_score = phases.get("p1b", {}).get("score", 0)

    if score == 0:
        if p1_score == 0 and p1b_score == 0:
            diagnosis["issues"].append("NEVER_CORRECT: No correct solution found in any phase")
        elif p1_score == 0 and p1b_score > 0:
            diagnosis["issues"].append("RETRY_SAVED: Only found correct solution via retry phase")

    # Check if shortening/evolve helped
    p2_score = phases.get("p2", {}).get("score", 0)
    p3_score = phases.get("p3", {}).get("score", 0)
    p1b_len = phases.get("p1b", {}).get("len")
    p2_len = phases.get("p2", {}).get("len")
    p3_len = phases.get("p3", {}).get("len")

    if p1b_score > 0 and p2_score == p1b_score and p1b_len and p2_len and p2_len >= p1b_len:
        diagnosis["issues"].append("SHORTEN_NO_GAIN: Shortening phase produced no improvement")

    if p2_score > 0 and p3_score == p2_score and p2_len and p3_len and p3_len >= p2_len:
        diagnosis["issues"].append("EVOLVE_NO_GAIN: Evolve phase produced no improvement")

    if correct_variants < 2:
        diagnosis["issues"].append(f"LOW_DIVERSITY: Only {correct_variants} correct variant(s) — evolve needs >= 2")

    # Analyze rollout details if available
    if rollout:
        gen_entries = [e for e in rollout if e.get("phase") == "generate"]
        retry_entries = [e for e in rollout if e.get("phase") == "retry"]
        shorten_entries = [e for e in rollout if e.get("phase") == "shorten"]

        # Count extraction failures
        no_code = sum(1 for e in gen_entries if e.get("extracted_code") is None and "error" not in e)
        if no_code > 0:
            diagnosis["issues"].append(f"EXTRACT_FAIL: {no_code}/{len(gen_entries)} generation attempts failed code extraction")

        # Count errors
        errors = [e for e in rollout if "error" in e]
        if errors:
            error_msgs = list({e["error"] for e in errors})[:3]
            diagnosis["issues"].append(f"ERRORS: {len(errors)} errors — {'; '.join(error_msgs)}")

        # Check if submissions were incorrect
        gen_submitted = [e for e in gen_entries if e.get("submit_result")]
        gen_incorrect = [e for e in gen_submitted if not e["submit_result"].get("correct")]
        if gen_submitted and len(gen_incorrect) == len(gen_submitted):
            diagnosis["issues"].append(f"ALL_WRONG: All {len(gen_submitted)} submitted generations were incorrect")

        # Check shorten correctness rate
        shorten_submitted = [e for e in shorten_entries if e.get("submit_result")]
        shorten_correct = [e for e in shorten_submitted if e["submit_result"].get("correct")]
        if shorten_submitted:
            diagnosis["issues"].append(
                f"SHORTEN_RATE: {len(shorten_correct)}/{len(shorten_submitted)} shorten attempts correct"
            )

        # Best and worst generation lengths
        gen_lens = [e["code_len"] for e in gen_entries if e.get("code_len")]
        if gen_lens:
            diagnosis["gen_len_range"] = [min(gen_lens), max(gen_lens)]

        # Final best code snippet (first 200 chars)
        final = [e for e in rollout if e.get("phase") == "final_best"]
        if final and final[0].get("code"):
            diagnosis["final_code_preview"] = final[0]["code"][:200]

    if not diagnosis["issues"]:
        diagnosis["issues"].append("NO_OBVIOUS_ISSUE: Score is low but no clear failure pattern")

    return diagnosis


def print_aggregate_report(label: str, problems: list[dict], analyses: list[dict]):
    """Print phase-on-phase improvement and issue frequency for a set of problems."""
    n = len(problems)
    phase_transitions = [("p1", "p1b"), ("p1b", "p2"), ("p2", "p3")]
    print("\n" + "=" * 80)
    print(f"Phase-on-phase improvement ({label}, n={n}):")
    for prev_phase, next_phase in phase_transitions:
        deltas = []
        improved_count = 0
        for p in problems:
            phases = p.get("phases", {})
            prev_score = phases.get(prev_phase, {}).get("score")
            next_score = phases.get(next_phase, {}).get("score")
            if prev_score is not None and next_score is not None:
                delta = next_score - prev_score
                deltas.append(delta)
                if delta > 0:
                    improved_count += 1
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            print(f"  {prev_phase} -> {next_phase}: mean improvement = {mean_delta:+.4f}, "
                  f"improved {improved_count}/{len(deltas)} problems")
        else:
            print(f"  {prev_phase} -> {next_phase}: no data")

    print(f"\nIssue frequency ({label}):")
    issue_counts: dict[str, int] = {}
    for a in analyses:
        for issue in a["issues"]:
            tag = issue.split(":")[0]
            issue_counts[tag] = issue_counts.get(tag, 0) + 1
    for tag, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}/{n}")


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout folder for worst-scoring problems")
    parser.add_argument("folder", help="Path to rollout run folder (e.g. rollouts/20260309_143000)")
    parser.add_argument("-n", "--top", type=int, default=10,
                        help="Number of worst problems to analyze (default: 10)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save analysis to JSON file")
    args = parser.parse_args()

    metrics, rollouts = load_run(args.folder)

    problems = metrics.get("problems", [])
    if not problems:
        print("No problems found in metrics.json")
        return

    # Sort by score ascending (worst first)
    problems.sort(key=lambda p: p.get("score", 0))
    worst = problems[:args.top]

    print(f"Analyzing {len(worst)} worst-scoring problems out of {len(problems)} total")
    print(f"Total score: {metrics.get('total_score', 'N/A')}")
    print(f"Problems solved: {metrics.get('problems_solved', 'N/A')}/{metrics.get('problems_total', 'N/A')}")
    print("=" * 80)

    analyses = []
    for p in worst:
        pid = p["id"]
        rollout = rollouts.get(pid)
        diag = diagnose_problem(p, rollout)
        analyses.append(diag)

        # Print summary
        print(f"\nProblem {pid}: score={p['score']}, baseline={p.get('baseline')}")
        phases = p.get("phases", {})
        for phase_name in ["p1", "p1b", "p2", "p3"]:
            ph = phases.get(phase_name, {})
            print(f"  {phase_name}: score={ph.get('score', 'N/A')}, len={ph.get('len', 'N/A')}")
        for issue in diag["issues"]:
            print(f"  >> {issue}")
        if "gen_len_range" in diag:
            print(f"  Gen lengths: {diag['gen_len_range'][0]}-{diag['gen_len_range'][1]}")
        if not rollout:
            print("  (no rollout data available)")

    # --- Reports for worst N and all problems ---
    all_analyses = []
    for p in problems:
        pid = p["id"]
        rollout = rollouts.get(pid)
        all_analyses.append(diagnose_problem(p, rollout))

    print_aggregate_report("Worst problems", worst, analyses)
    print_aggregate_report("All problems", problems, all_analyses)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(analyses, f, indent=2)
        print(f"\nAnalysis saved to {args.output}")


if __name__ == "__main__":
    main()
