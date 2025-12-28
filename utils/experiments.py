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
import json
import sys
from pathlib import Path

import yaml

from evals.metrics import compute_retrieval_metrics, compute_response_metrics, compute_manual_label_rates

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


def get_latest_run_id(runs_dir: Path | str) -> str | None:
    """Get the latest run ID from existing runs.

    Scans the runs directory for existing run folders (r001, r002, etc.)
    and returns the most recent one.

    Args:
        runs_dir: Path to the runs directory

    Returns:
        Latest run ID (e.g., "r005") or None if no runs exist
    """
    runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        return None

    existing = sorted(runs_dir.glob("r*"))
    if not existing:
        return None

    return existing[-1].name  # e.g., "r005"


def compute_and_save_metrics(
    exp_id: str,
    run_id: str,
    run_dir: Path,
) -> None:
    """Compute standard metrics from retrieval.jsonl and write metrics.json.

    Groups results by split (dev/test) and computes metrics per split.
    Reads extra.jsonl for experiment-specific metrics if present.

    Args:
        exp_id: Experiment identifier
        run_id: Run identifier
        run_dir: Path to run directory containing retrieval.jsonl
    """
    retrieval_path = run_dir / "retrieval.jsonl"
    if not retrieval_path.exists():
        print(f"Warning: {retrieval_path} not found, skipping metrics computation")
        return

    # Load available tags for manual label computation
    tags_path = PROJECT_ROOT / "evals" / "tags.json"
    available_tags = []
    if tags_path.exists():
        with open(tags_path) as f:
            available_tags = json.load(f)

    # Read retrieval results and group by split
    results_by_split: dict[str, list[dict]] = {}
    with open(retrieval_path) as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                split = result.get("split", "unknown")
                if split not in results_by_split:
                    results_by_split[split] = []
                results_by_split[split].append(result)

    # Read extra.jsonl for experiment-specific metrics (keyed by split)
    extra_evals: dict[str, dict] = {}
    extra_path = run_dir / "extra.jsonl"
    if extra_path.exists():
        with open(extra_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    split = entry.pop("split", "unknown")
                    extra_evals[split] = entry

    # Build query_id -> split mapping from retrieval results
    query_id_to_split = {}
    for split, results in results_by_split.items():
        for result in results:
            query_id_to_split[result["query_id"]] = split

    # Read responses.jsonl and group by split using the mapping
    responses_by_split: dict[str, list[dict]] = {}
    responses_path = run_dir / "responses.jsonl"
    if responses_path.exists():
        with open(responses_path) as f:
            for line in f:
                if line.strip():
                    response = json.loads(line)
                    # Get split from query_id mapping
                    query_id = response.get("query_id")
                    split = query_id_to_split.get(query_id, "unknown")
                    if split not in responses_by_split:
                        responses_by_split[split] = []
                    responses_by_split[split].append(response)

    # Read run_receipt for metadata
    receipt_path = run_dir / "run_receipt.json"
    receipt = {}
    if receipt_path.exists():
        with open(receipt_path) as f:
            receipt = json.load(f)

    # Build metrics.json with per-split structure
    metrics = {
        "exp_id": exp_id,
        "run_id": run_id,
        "query_set_id": receipt.get("query_set_id", ""),
    }

    # Compute metrics for each split
    for split, results in results_by_split.items():
        split_metrics = {
            "query_count": len(results),
            "retrieval": compute_retrieval_metrics(results),
            "response_auto": compute_response_metrics(responses_by_split.get(split, [])),
            "response_manual": compute_manual_label_rates(
                responses_by_split.get(split, []),
                available_tags
            ),
        }
        # Add extra evals for this split if present
        if split in extra_evals:
            split_metrics["extra"] = extra_evals[split]
        metrics[split] = split_metrics

    # Write metrics.json
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics: {metrics_path}")


def update_registry(exp_id: str, run_id: str) -> None:
    """Append or overwrite run in registry.yaml.

    Reads experiment metadata and run metrics, then updates the registry.
    Only includes TEST split metrics (not dev).
    Overwrites if (exp_id, run_id) already exists.

    Args:
        exp_id: Experiment identifier (e.g., "exp_001")
        run_id: Run identifier (e.g., "r001")
    """
    exp_dir = EXPERIMENTS_DIR / exp_id
    run_dir = exp_dir / "runs" / run_id
    registry_path = EXPERIMENTS_DIR / "registry.yaml"

    # Read experiment.yaml for name/description
    exp_yaml_path = exp_dir / "experiment.yaml"
    if not exp_yaml_path.exists():
        print(f"Warning: {exp_yaml_path} not found, skipping registry update")
        return

    with open(exp_yaml_path) as f:
        exp_config = yaml.safe_load(f)

    # Read run_receipt.json for timestamp, query_set_id
    receipt_path = run_dir / "run_receipt.json"
    if not receipt_path.exists():
        print(f"Warning: {receipt_path} not found, skipping registry update")
        return

    with open(receipt_path) as f:
        receipt = json.load(f)

    # Read metrics.json for eval metrics
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} not found, skipping registry update")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Get test split metrics (registry only shows test, not dev)
    test_metrics = metrics.get("test", {})
    if not test_metrics:
        print(f"Warning: No test metrics found in {metrics_path}, skipping registry update")
        return

    # Build registry entry
    description = exp_config.get("description", "")
    if isinstance(description, str):
        # Collapse multi-line to single line
        description = " ".join(description.split())

    run_data = {
        "timestamp": receipt.get("test", {}).get("timestamp", ""),
        "exp_id": exp_id,
        "name": exp_config.get("name", ""),
        "description": description,
        "run_id": run_id,
        "query_set_id": metrics.get("query_set_id", ""),
        "retrieval": test_metrics.get("retrieval", {}),
        "response": test_metrics.get("response_auto", {}),  # Only auto labels in registry
    }

    # Read existing entries, filter out duplicate
    entries = []
    if registry_path.exists():
        with open(registry_path) as f:
            data = yaml.safe_load(f)
            if data is None:
                data = []
            entries = data if isinstance(data, list) else []
    
    # Filter out duplicate entry if it exists
    entries = [
        entry for entry in entries
        if (entry["exp_id"], entry["run_id"]) != (exp_id, run_id)
    ]

    entries.append(run_data)

    # Sort by timestamp
    entries.sort(key=lambda x: x.get("timestamp", ""))

    # Write back
    with open(registry_path, "w") as f:
        yaml.dump(entries, f, default_flow_style=False, sort_keys=False)

    print(f"Updated registry: {exp_id}/{run_id}")


def run_experiment(
    exp_id: str,
    run_id: str | None = None,
    max_queries: int | None = None,
    no_outputs: bool = False,
) -> None:
    """Run a single experiment by ID.

    Args:
        exp_id: Experiment identifier (e.g., "exp_001")
        run_id: Run identifier (e.g., "r001"). Auto-generated if not provided.
        max_queries: Maximum number of queries to process per split (default: all)
        no_outputs: Skip output generation, only recompute metrics (default: False)

    Raises:
        ValueError: If experiment directory or generate.py not found (unless no_outputs=True)
    """
    exp_dir = EXPERIMENTS_DIR / exp_id
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory not found: {exp_dir}")

    # Only check for generate.py if we're actually going to generate outputs
    if not no_outputs:
        generate_path = exp_dir / "src" / "generate.py"
        if not generate_path.exists():
            raise ValueError(f"Experiment generate.py not found: {generate_path}")

    runs_dir = exp_dir / "runs"

    # Auto-generate run ID if not provided
    if run_id is None:
        if no_outputs:
            # In no-outputs mode, use the latest run
            run_id = get_latest_run_id(runs_dir)
            if run_id is None:
                raise ValueError(f"No existing runs found in {runs_dir}. Cannot use --no-outputs without existing data.")
        else:
            run_id = get_next_run_id(runs_dir)

    # Verify run directory exists when in no-outputs mode
    if no_outputs:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}. Cannot use --no-outputs without existing data.")

    # Skip generation if --no-outputs flag is set
    if not no_outputs:
        # Add experiment src directory to path for local imports
        src_dir = exp_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        # Dynamic import of experiment's generate module
        spec = importlib.util.spec_from_file_location(f"{exp_id}.generate", generate_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Generate outputs (retrieval, responses, extra metrics)
        module.generate(run_id, max_queries)
    else:
        print(f"Skipping output generation (--no-outputs mode)")

    # Compute standard metrics from output files and save metrics.json
    run_dir = runs_dir / run_id
    compute_and_save_metrics(exp_id, run_id, run_dir)

    # Update registry with this run's results
    update_registry(exp_id, run_id)


def run_all(
    run_id: str | None = None,
    max_queries: int | None = None,
    no_outputs: bool = False,
) -> None:
    """Run all experiments with the same arguments.

    Args:
        run_id: Run identifier (e.g., "r001"). Auto-generated per experiment if not provided.
        max_queries: Maximum number of queries to process per split (default: all)
        no_outputs: Skip output generation, only recompute metrics (default: False)
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
            run_experiment(exp_id, run_id, max_queries, no_outputs)
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
    parser.add_argument(
        "--no-outputs",
        action="store_true",
        help="Skip output generation, only recompute metrics from existing files",
    )

    args = parser.parse_args()

    if args.run_all:
        run_all(args.run_id, args.max_queries, args.no_outputs)
    elif args.exp:
        run_experiment(args.exp, args.run_id, args.max_queries, args.no_outputs)
    else:
        parser.print_help()
        print("\nError: Must specify --exp EXP_ID or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()

