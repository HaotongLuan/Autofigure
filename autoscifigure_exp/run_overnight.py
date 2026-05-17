"""
Overnight experiment runner for AutoSciFigure pipeline.
Runs all 10 test cases through Plan-Critique-Refine pipeline,
evaluates with VLM-as-Judge, and generates comparison report.

Usage:
  python run_overnight.py           # Run all 10 cases
  python run_overnight.py --cases 5 # Run first 5 cases only
  python run_overnight.py --skip-eval  # Skip VLM evaluation
"""

import json
import os
import sys
import time
from datetime import datetime

# --- API credentials from Claude Code config ---
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-5832884ed915490988b3e88ad4a45bc9")
os.environ.setdefault("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import SciFigurePipeline
from renderer import SVGRenderer
from evaluator import FigureEvaluator
from test_cases import get_test_cases, get_test_case


def run_overnight(num_cases: int = 10, skip_eval: bool = False, max_rounds: int = 2):
    """Run the full experiment pipeline on all test cases."""
    run_id = f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join("results", run_id)
    os.makedirs(out_dir, exist_ok=True)

    all_cases = get_test_cases()
    test_cases = all_cases[:num_cases]

    print("=" * 70)
    print(f"AutoSciFigure Overnight Experiment — {len(test_cases)} test cases")
    print(f"Run ID: {run_id}")
    print(f"Max refine rounds: {max_rounds}")
    print("=" * 70)

    pipeline = SciFigurePipeline()
    evaluator = FigureEvaluator() if not skip_eval else None
    renderer = SVGRenderer()

    results = {}
    all_stats = []

    for i, tc in enumerate(test_cases):
        tc_id = tc["id"]
        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(test_cases)}] {tc_id}: {tc['title']}")
        print(f"  Type: {tc['figure_type']}")
        print(f"  Text: {len(tc['text'])} chars")
        print(f"  Ground truth entities: {tc['ground_truth']['key_entities']}")

        t0 = time.time()
        try:
            result = pipeline.generate(tc["text"], tc["figure_type"], max_rounds=max_rounds)
            elapsed = time.time() - t0

            # Save SVG
            svg_path = os.path.join(out_dir, f"{tc_id}.svg")
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(result["svg"])

            # Save plan JSON
            plan_path = os.path.join(out_dir, f"{tc_id}_plan.json")
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(result["plan"], f, indent=2, ensure_ascii=False)

            # Save critique
            crit_path = os.path.join(out_dir, f"{tc_id}_critique.json")
            with open(crit_path, "w", encoding="utf-8") as f:
                json.dump(result.get("critique", {}), f, indent=2, ensure_ascii=False)

            cq = result.get("critique", {})
            stats = {
                "id": tc_id,
                "title": tc["title"],
                "type": tc["figure_type"],
                "nodes": len(result["plan"].get("nodes", [])),
                "edges": len(result["plan"].get("edges", [])),
                "groups": len(result["plan"].get("groups", [])),
                "svg_chars": len(result["svg"]),
                "rounds": result["rounds"],
                "time_seconds": round(elapsed, 1),
                "overall_quality": cq.get("overall_quality"),
                "structure_correctness": cq.get("structure_correctness"),
                "text_accuracy": cq.get("text_accuracy"),
                "visual_clarity": cq.get("visual_clarity"),
            }
            all_stats.append(stats)
            results[tc_id] = result

            print(f"  Plan: {stats['nodes']} nodes, {stats['edges']} edges, {stats['groups']} groups")
            print(f"  Critique: overall={cq.get('overall_quality')}, "
                  f"structure={cq.get('structure_correctness')}, "
                  f"text={cq.get('text_accuracy')}")
            print(f"  Rounds: {result['rounds']}, Time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({
                "id": tc_id, "title": tc["title"], "type": tc["figure_type"],
                "error": str(e), "time_seconds": round(elapsed, 1),
            })

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    succeeded = [s for s in all_stats if "error" not in s]
    failed = [s for s in all_stats if "error" in s]
    print(f"Succeeded: {len(succeeded)}/{len(all_stats)}")
    print(f"Failed: {len(failed)}/{len(all_stats)}")

    if succeeded:
        avg_quality = sum(s.get("overall_quality", 0) or 0 for s in succeeded) / len(succeeded)
        avg_rounds = sum(s.get("rounds", 1) for s in succeeded) / len(succeeded)
        avg_time = sum(s.get("time_seconds", 0) for s in succeeded) / len(succeeded)
        avg_nodes = sum(s.get("nodes", 0) for s in succeeded) / len(succeeded)
        print(f"Avg Overall Quality: {avg_quality:.1f}/10")
        print(f"Avg Refine Rounds: {avg_rounds:.1f}")
        print(f"Avg Time per Case: {avg_time:.1f}s")
        print(f"Avg Nodes per Figure: {avg_nodes:.1f}")

        print("\n--- Per-Case Scores ---")
        print(f"{'ID':6s} {'Quality':>8s} {'Structure':>10s} {'Text':>6s} {'Clarity':>8s} {'Rounds':>7s} {'Time':>8s}")
        for s in succeeded:
            print(f"{s['id']:6s} {str(s.get('overall_quality','-')):>8s} "
                  f"{str(s.get('structure_correctness','-')):>10s} "
                  f"{str(s.get('text_accuracy','-')):>6s} "
                  f"{str(s.get('visual_clarity','-')):>8s} "
                  f"{s.get('rounds',0):>7d} {str(s.get('time_seconds','-')):>8s}")

    if failed:
        print("\n--- Failed Cases ---")
        for s in failed:
            print(f"  {s['id']}: {s['error'][:120]}")

    # Save summary
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "timestamp": datetime.now().isoformat(),
                    "config": {"max_rounds": max_rounds, "num_cases": num_cases},
                    "stats": all_stats, "succeeded": len(succeeded), "failed": len(failed)},
                  f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to: results/{run_id}/")
    return results, all_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10, help="Number of test cases to run")
    parser.add_argument("--skip-eval", action="store_true", help="Skip VLM evaluation")
    parser.add_argument("--rounds", type=int, default=2, help="Max refine rounds")
    args = parser.parse_args()
    run_overnight(num_cases=args.cases, skip_eval=args.skip_eval, max_rounds=args.rounds)
