"""
Centralized Experiment Runner

Provides functions to run experiments by ID or run all experiments.

Usage:
  python -m utils.experiments --exp exp_001                    # runs exp_001, auto run-id
  python -m utils.experiments --exp exp_001 --run-id r005      # runs exp_001, specified run-id
  python -m utils.experiments --all --max-queries 5            # runs all experiments
"""

import argparse
import importlib
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def get_experiment_ids() -> list[str]:
    """Discover all experiment directories (exp_XXX pattern).

    Returns:
        Sorted list of experiment IDs (e.g., ["exp_001", "exp_002"])
    """
    exp_dirs = sorted(EXPERIMENTS_DIR.glob("exp_*"))
    return [d.name for d in exp_dirs if d.is_dir()]


def get_next_run_id(runs_dir: Path | str) -> str:
    """Get next run ID by counting existing runs.

    Scans the runs directory for existing run folders (r001, r002, etc.)
    and returns the next sequential ID.

    Args:
        runs_dir: Path to the runs directory

    Returns:
        Next run ID (e.g., "r001", "r002", ...)
    """
    runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        return "r001"

    existing = sorted(runs_dir.glob("r*"))
    if not existing:
        return "r001"

    last = existing[-1].name  # e.g., "r005"
    num = int(last[1:]) + 1
    return f"r{num:03d}"


def run_experiment(
    exp_id: str,
    run_id: str | None = None,
    max_queries: int | None = None,
) -> None:
    """Run a single experiment by ID.

    Args:
        exp_id: Experiment identifier (e.g., "exp_001")
        run_id: Run identifier (e.g., "r001"). Auto-generated if not provided.
        max_queries: Maximum number of queries to process per split (default: all)

    Raises:
        ValueError: If experiment directory or main.py not found
    """
    exp_dir = EXPERIMENTS_DIR / exp_id
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory not found: {exp_dir}")

    main_path = exp_dir / "src" / "main.py"
    if not main_path.exists():
        raise ValueError(f"Experiment main.py not found: {main_path}")

    runs_dir = exp_dir / "runs"

    # Auto-generate run ID if not provided
    if run_id is None:
        run_id = get_next_run_id(runs_dir)

    # Add experiment src directory to path for local imports
    src_dir = exp_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Dynamic import of experiment's main module
    spec = importlib.util.spec_from_file_location(f"{exp_id}.main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Run the experiment
    module.run_experiment(run_id, max_queries)


def run_all(
    run_id: str | None = None,
    max_queries: int | None = None,
) -> None:
    """Run all experiments with the same arguments.

    Args:
        run_id: Run identifier (e.g., "r001"). Auto-generated per experiment if not provided.
        max_queries: Maximum number of queries to process per split (default: all)
    """
    exp_ids = get_experiment_ids()

    if not exp_ids:
        print("No experiments found in experiments/ directory")
        return

    print(f"Found {len(exp_ids)} experiment(s): {', '.join(exp_ids)}")
    print("=" * 72)

    for exp_id in exp_ids:
        print(f"\n>>> Running {exp_id}...")
        try:
            run_experiment(exp_id, run_id, max_queries)
        except Exception as e:
            print(f"ERROR running {exp_id}: {e}")
            continue

    print("\n" + "=" * 72)
    print("All experiments complete!")


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Experiment ID to run (e.g., exp_001)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all experiments",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (e.g., r001). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to process per split (default: all)",
    )

    args = parser.parse_args()

    if args.run_all:
        run_all(args.run_id, args.max_queries)
    elif args.exp:
        run_experiment(args.exp, args.run_id, args.max_queries)
    else:
        parser.print_help()
        print("\nError: Must specify --exp EXP_ID or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()

