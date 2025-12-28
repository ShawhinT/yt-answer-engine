"""Streamlit dashboard for viewing and comparing experiment results."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_registry() -> list[dict]:
    """Load all experiment runs from registry.yaml.

    Returns:
        List of registry entries with test metrics, or empty list if not found.
    """
    registry_path = EXPERIMENTS_DIR / "registry.yaml"

    if not registry_path.exists():
        return []

    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, list) else []
    except (yaml.YAMLError, IOError):
        return []


def load_detailed_metrics(exp_id: str, run_id: str) -> dict | None:
    """Load detailed metrics.json for a specific run.

    Args:
        exp_id: Experiment ID (e.g., 'exp_001')
        run_id: Run ID (e.g., 'r001')

    Returns:
        Dict with dev and test metrics, or None if not found.
    """
    metrics_path = EXPERIMENTS_DIR / exp_id / "runs" / run_id / "metrics.json"

    if not metrics_path.exists():
        return None

    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_unique_exp_ids(registry: list[dict]) -> list[str]:
    """Get all unique experiment IDs from registry.

    Args:
        registry: List of registry entries

    Returns:
        Sorted list of unique experiment IDs.
    """
    exp_ids = set(entry.get('exp_id', '') for entry in registry)
    return sorted(exp_ids)


def get_runs_for_experiments(registry: list[dict], exp_ids: list[str]) -> list[dict]:
    """Get all runs for given experiment IDs.

    Args:
        registry: List of registry entries
        exp_ids: List of experiment IDs to filter by

    Returns:
        Filtered list of registry entries.
    """
    return [entry for entry in registry if entry.get('exp_id') in exp_ids]


# ============================================================================
# Data Processing Functions
# ============================================================================

def registry_to_dataframe(registry: list[dict]) -> pd.DataFrame:
    """Convert registry entries to displayable DataFrame.

    Args:
        registry: List of registry entries

    Returns:
        DataFrame with formatted columns for display.
    """
    rows = []
    for entry in registry:
        retrieval = entry.get('retrieval', {})
        response = entry.get('response', {})

        # Truncate long descriptions for table display
        description = entry.get('description', '')
        if len(description) > 65:
            description = description[:65] + "..."

        rows.append({
            "Timestamp": entry.get("timestamp", ""),
            "Exp ID": entry.get("exp_id", ""),
            "Run ID": entry.get("run_id", ""),
            "Name": entry.get("name", ""),
            "Description": description,
            "Query Set": entry.get("query_set_id", ""),
            "Recall@1": retrieval.get('recall@1', 0),
            "Recall@3": retrieval.get('recall@3', 0),
            "MRR": retrieval.get('mrr', 0),
            "Bad Framing": response.get('bad_framing_rate', 0),
        })

    return pd.DataFrame(rows)


# ============================================================================
# Rendering Functions
# ============================================================================

def render_default_table(registry: list[dict]):
    """Render default table showing all runs from registry.

    Args:
        registry: List of all registry entries
    """
    st.header("All Experiment Runs")
    st.caption("Showing test split metrics from registry")

    if not registry:
        st.info("No experiments found in registry")
        st.markdown("""
        **Get started:**
        1. Run an experiment to populate the registry
        2. Metrics will appear here automatically
        """)
        return

    # Convert to DataFrame
    df = registry_to_dataframe(registry)

    # Format percentage and float columns
    styled_df = df.copy()
    styled_df["Recall@1"] = styled_df["Recall@1"].apply(lambda x: f"{x:.2%}")
    styled_df["Recall@3"] = styled_df["Recall@3"].apply(lambda x: f"{x:.2%}")
    styled_df["MRR"] = styled_df["MRR"].apply(lambda x: f"{x:.3f}")
    styled_df["Bad Framing"] = styled_df["Bad Framing"].apply(lambda x: f"{x:.2%}")

    # Display table
    st.dataframe(
        styled_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Description": st.column_config.TextColumn(
                "Description",
                width=400,
            ),
        },
    )

    # Summary metrics
    st.divider()
    col1, col2, col3 = st.columns(3)

    unique_exps = set(entry['exp_id'] for entry in registry)
    latest_run = f"{registry[-1]['exp_id']}/{registry[-1]['run_id']}" if registry else "N/A"

    col1.metric("Total Experiments", len(unique_exps))
    col2.metric("Total Runs", len(registry))
    col3.metric("Latest Run", latest_run)

    # Full Descriptions section
    st.divider()
    st.subheader("📋 Full Descriptions")

    for entry in registry:
        exp_id = entry['exp_id']
        run_id = entry['run_id']
        name = entry.get('name', 'Unnamed')
        full_desc = entry.get('description', 'No description provided')

        with st.expander(f"{exp_id}/{run_id} - {name}"):
            st.write(full_desc)


def render_filtered_table(registry: list[dict], selected_exps: list[str]):
    """Render filtered table based on selected experiments.

    Args:
        registry: All registry entries
        selected_exps: List of exp_ids to show
    """
    st.header("Filtered Experiment View")

    # Filter registry
    filtered = get_runs_for_experiments(registry, selected_exps)

    if not filtered:
        st.info("No runs match the selected filters")
        return

    # Reuse default table rendering
    st.caption(f"Showing {len(filtered)} runs from selected experiments")

    # Convert to DataFrame
    df = registry_to_dataframe(filtered)

    # Format percentage and float columns
    styled_df = df.copy()
    styled_df["Recall@1"] = styled_df["Recall@1"].apply(lambda x: f"{x:.2%}")
    styled_df["Recall@3"] = styled_df["Recall@3"].apply(lambda x: f"{x:.2%}")
    styled_df["MRR"] = styled_df["MRR"].apply(lambda x: f"{x:.3f}")
    styled_df["Bad Framing"] = styled_df["Bad Framing"].apply(lambda x: f"{x:.2%}")

    # Display table
    st.dataframe(
        styled_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Description": st.column_config.TextColumn(
                "Description",
                width=400,
            ),
        },
    )

    # Full Descriptions section
    st.divider()
    st.subheader("📋 Full Descriptions")

    for entry in filtered:
        exp_id = entry['exp_id']
        run_id = entry['run_id']
        name = entry.get('name', 'Unnamed')
        full_desc = entry.get('description', 'No description provided')

        with st.expander(f"{exp_id}/{run_id} - {name}"):
            st.write(full_desc)


def render_split_analysis(registry: list[dict], selected_exps: list[str] | None = None):
    """Show dev vs test comparison for detecting overfitting.

    Args:
        registry: All registry entries
        selected_exps: Optional filter for specific experiments
    """
    st.header("Dev vs Test Split Analysis")
    st.caption("Compare performance across splits to detect overfitting")

    # Filter if needed
    entries = registry if not selected_exps else get_runs_for_experiments(registry, selected_exps)

    if not entries:
        st.info("No runs match the selected filters")
        return

    # Build comparison table
    rows = []
    skipped = []

    for entry in entries:
        exp_id = entry['exp_id']
        run_id = entry['run_id']

        # Load detailed metrics
        metrics = load_detailed_metrics(exp_id, run_id)

        if not metrics:
            skipped.append(f"{exp_id}/{run_id}: no detailed metrics")
            continue

        if 'dev' not in metrics or 'test' not in metrics:
            skipped.append(f"{exp_id}/{run_id}: missing dev or test split")
            continue

        dev = metrics['dev']
        test = metrics['test']

        # Extract metrics
        dev_recall1 = dev.get('retrieval', {}).get('recall@1', 0)
        test_recall1 = test.get('retrieval', {}).get('recall@1', 0)
        recall1_delta = test_recall1 - dev_recall1

        dev_recall3 = dev.get('retrieval', {}).get('recall@3', 0)
        test_recall3 = test.get('retrieval', {}).get('recall@3', 0)
        recall3_delta = test_recall3 - dev_recall3

        dev_mrr = dev.get('retrieval', {}).get('mrr', 0)
        test_mrr = test.get('retrieval', {}).get('mrr', 0)
        mrr_delta = test_mrr - dev_mrr

        rows.append({
            "Exp/Run": f"{exp_id}/{run_id}",
            "Dev Recall@1": f"{dev_recall1:.2%}",
            "Test Recall@1": f"{test_recall1:.2%}",
            "Δ Recall@1": f"{recall1_delta:+.2%}",
            "Dev Recall@3": f"{dev_recall3:.2%}",
            "Test Recall@3": f"{test_recall3:.2%}",
            "Δ Recall@3": f"{recall3_delta:+.2%}",
            "Dev MRR": f"{dev_mrr:.3f}",
            "Test MRR": f"{test_mrr:.3f}",
            "Δ MRR": f"{mrr_delta:+.3f}",
            "Dev Queries": dev.get('query_count', 0),
            "Test Queries": test.get('query_count', 0),
        })

    # Show skipped runs
    if skipped:
        with st.expander(f"Skipped {len(skipped)} runs"):
            for msg in skipped:
                st.caption(f"⚠️ {msg}")

    if not rows:
        st.warning("No runs with both dev and test metrics found")
        st.info("Ensure experiments have been run on both splits")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', hide_index=True)

    # Helpful tip
    st.info("💡 Large negative deltas (test << dev) may indicate overfitting")


def render_run_comparison(registry: list[dict], selected_runs: list[str], split: str):
    """Compare multiple runs side-by-side.

    Args:
        registry: All registry entries
        selected_runs: List of "exp_id/run_id" strings
        split: "test" or "dev"
    """
    st.header("Run Comparison")
    st.caption(f"Showing {split} split metrics")

    if len(selected_runs) < 2:
        st.info("Select at least 2 runs to compare")
        return

    # Parse selected runs
    run_tuples = []
    for run_str in selected_runs:
        parts = run_str.split('/')
        if len(parts) == 2:
            run_tuples.append((parts[0], parts[1]))

    if not run_tuples:
        st.warning("Invalid run selection")
        return

    # Load metrics for each run
    run_data = []
    for exp_id, run_id in run_tuples:
        # For test split, can use registry
        if split == "test":
            entry = next((e for e in registry if e['exp_id'] == exp_id and e['run_id'] == run_id), None)
            if entry:
                run_data.append({
                    'label': f"{exp_id}/{run_id}",
                    'name': entry.get('name', ''),
                    'retrieval': entry.get('retrieval', {}),
                    'response': entry.get('response', {}),
                })
        else:
            # For dev split, need detailed metrics
            metrics = load_detailed_metrics(exp_id, run_id)
            if metrics and 'dev' in metrics:
                dev = metrics['dev']
                run_data.append({
                    'label': f"{exp_id}/{run_id}",
                    'name': '',
                    'retrieval': dev.get('retrieval', {}),
                    'response': dev.get('response_auto', {}),
                })

    if not run_data:
        st.warning(f"No {split} metrics found for selected runs")
        return

    # Create comparison columns
    st.subheader("Retrieval Metrics")
    cols = st.columns(len(run_data))

    for col, data in zip(cols, run_data):
        with col:
            st.markdown(f"**{data['label']}**")
            if data['name']:
                st.caption(data['name'])

            retrieval = data['retrieval']
            st.metric("Recall@1", f"{retrieval.get('recall@1', 0):.2%}")
            st.metric("Recall@3", f"{retrieval.get('recall@3', 0):.2%}")
            st.metric("MRR", f"{retrieval.get('mrr', 0):.3f}")

    st.divider()
    st.subheader("Response Metrics")
    cols = st.columns(len(run_data))

    for col, data in zip(cols, run_data):
        with col:
            st.markdown(f"**{data['label']}**")
            response = data['response']
            st.metric("Bad Framing", f"{response.get('bad_framing_rate', 0):.2%}")


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Experiment Viewer",
        page_icon="📊",
        layout="wide"
    )

    # Load data
    registry = load_registry()

    # Check if registry exists
    if not registry:
        st.error("Registry file not found at experiments/registry.yaml")
        st.info("Run an experiment to populate the registry")
        return

    # Sidebar
    with st.sidebar:
        st.title("Experiment Viewer")
        st.caption("📊 View and compare experiment results")

        st.divider()

        # View mode selector
        view_mode = st.radio(
            "View Mode",
            options=["All Runs", "Filter by Experiment", "Compare Runs", "Dev vs Test Analysis"],
            index=0
        )

        st.divider()

        # Conditional filters
        selected_exps = None
        selected_runs = None
        split_view = "test"

        if view_mode in ["Filter by Experiment", "Dev vs Test Analysis"]:
            st.subheader("Filters")

            # Get available experiments
            experiments = get_unique_exp_ids(registry)
            selected_exps = st.multiselect(
                "Experiments",
                options=experiments,
                default=experiments if view_mode == "Filter by Experiment" else None
            )

        if view_mode == "Compare Runs":
            st.subheader("Select Runs")

            # Get all available runs
            run_options = [f"{e['exp_id']}/{e['run_id']}" for e in registry]
            selected_runs = st.multiselect(
                "Runs to Compare",
                options=run_options,
                default=run_options[:2] if len(run_options) >= 2 else run_options
            )

            st.divider()
            split_view = st.radio(
                "Data Split",
                options=["test", "dev"],
                index=0
            )

    # Main area - route to appropriate view
    if view_mode == "All Runs":
        render_default_table(registry)
    elif view_mode == "Filter by Experiment":
        if selected_exps:
            render_filtered_table(registry, selected_exps)
        else:
            st.info("Select one or more experiments from the sidebar")
    elif view_mode == "Compare Runs":
        render_run_comparison(registry, selected_runs or [], split_view)
    elif view_mode == "Dev vs Test Analysis":
        render_split_analysis(registry, selected_exps)


if __name__ == "__main__":
    main()
