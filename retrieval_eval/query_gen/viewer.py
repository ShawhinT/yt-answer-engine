"""Streamlit app for reviewing, editing, and curating synthetic queries."""

import json
import csv
import random
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

import streamlit as st
import pandas as pd

# Import enums from functions.py
from functions import QueryType, DifficultyLevel

# ============================================================================
# Configuration
# ============================================================================

JSONL_PATH = Path(__file__).parent / "data" / "raw_queries.jsonl"
CSV_OUTPUT_PATH = Path(__file__).parent / "data" / "queries.csv"

QUERY_TYPE_OPTIONS = [qt.value for qt in QueryType]
DIFFICULTY_OPTIONS = [dl.value for dl in DifficultyLevel]


# ============================================================================
# Core I/O Functions
# ============================================================================

def get_all_video_ids(jsonl_path: Path) -> list[tuple[str, str, int]]:
    """
    Get all unique video IDs with their titles and query counts.
    
    Returns:
        List of (video_id, title, query_count) tuples sorted by video_id.
    """
    video_data: dict[str, dict] = {}
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                video_id = record["video_id"]
                if video_id not in video_data:
                    video_data[video_id] = {
                        "title": record["video_title"],
                        "count": 0
                    }
                video_data[video_id]["count"] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    
    return sorted([
        (vid, data["title"], data["count"])
        for vid, data in video_data.items()
    ], key=lambda x: x[0])


def load_queries_for_video(video_id: str, jsonl_path: Path) -> list[dict]:
    """
    Load only queries for the specified video (just-in-time loading).
    
    Args:
        video_id: YouTube video ID to filter by.
        jsonl_path: Path to the JSONL file.
    
    Returns:
        List of query dictionaries for this video.
    """
    queries = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record["video_id"] == video_id:
                    # Ensure optional fields exist with defaults
                    record.setdefault("notes", "")
                    record.setdefault("golden", False)
                    queries.append(record)
            except (json.JSONDecodeError, KeyError):
                continue
    
    return queries


def update_query_in_jsonl(
    video_id: str,
    query_index: int,
    updated_query: dict,
    jsonl_path: Path
) -> bool:
    """
    Replace the Nth occurrence of video_id with the updated query.
    Uses atomic write (temp file + rename) for safety.
    
    Args:
        video_id: Video ID to match.
        query_index: Which occurrence of this video_id to update (0-indexed).
        updated_query: The new query dictionary.
        jsonl_path: Path to the JSONL file.
    
    Returns:
        True if update was successful, False otherwise.
    """
    # Create backup before modifying
    backup_path = jsonl_path.with_suffix('.jsonl.bak')
    shutil.copy2(jsonl_path, backup_path)
    
    lines = []
    current_index = 0
    updated = False
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            
            try:
                record = json.loads(stripped)
                if record["video_id"] == video_id:
                    if current_index == query_index:
                        # This is the line to update
                        lines.append(json.dumps(updated_query) + '\n')
                        updated = True
                    else:
                        lines.append(line)
                    current_index += 1
                else:
                    lines.append(line)
            except (json.JSONDecodeError, KeyError):
                lines.append(line)
    
    if not updated:
        return False
    
    # Atomic write: write to temp file, then rename
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=jsonl_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp:
        tmp.writelines(lines)
        tmp_path = Path(tmp.name)
    
    tmp_path.replace(jsonl_path)
    return True


def append_query_to_jsonl(query_dict: dict, jsonl_path: Path) -> None:
    """
    Append a new query to the end of the JSONL file.
    
    Args:
        query_dict: The query dictionary to append.
        jsonl_path: Path to the JSONL file.
    """
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(query_dict) + '\n')


def count_golden_queries(jsonl_path: Path) -> int:
    """Count total queries marked as golden."""
    count = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("golden", False):
                    count += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return count


def count_golden_videos(jsonl_path: Path) -> int:
    """Count videos that have at least one golden query."""
    golden_videos: set[str] = set()
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("golden", False):
                    golden_videos.add(record["video_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return len(golden_videos)


def export_golden_queries_csv(jsonl_path: Path, output_path: Path) -> dict:
    """
    Export golden queries to CSV with 80/20 video-level split.
    
    Args:
        jsonl_path: Path to source JSONL file.
        output_path: Path to output CSV file.
    
    Returns:
        Stats dictionary with counts and video IDs.
    """
    # Load all golden queries grouped by video
    queries_by_video: dict[str, list[dict]] = defaultdict(list)
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("golden", False):
                    queries_by_video[record["video_id"]].append(record)
            except (json.JSONDecodeError, KeyError):
                continue
    
    if not queries_by_video:
        return {
            "golden_count": 0,
            "val_videos": 0,
            "val_queries": 0,
            "test_videos": 0,
            "test_queries": 0,
            "val_video_ids": [],
            "test_video_ids": []
        }
    
    # Split videos 80/20 (seed=42 for reproducibility)
    video_ids = sorted(queries_by_video.keys())
    random.seed(42)
    random.shuffle(video_ids)
    
    split_point = int(len(video_ids) * 0.8)
    val_video_ids = video_ids[:split_point]
    test_video_ids = video_ids[split_point:]
    
    # Assign splits and generate query IDs
    rows = []
    for video_id in video_ids:
        split = "validation" if video_id in val_video_ids else "test"
        for idx, query in enumerate(queries_by_video[video_id]):
            query_id = f"{video_id}_{idx}"
            row = {
                "query_id": query_id,
                "video_id": query["video_id"],
                "video_title": query["video_title"],
                "query": query["query"],
                "query_type": query["query_type"],
                "difficulty": query["difficulty"],
                "grounding": query["grounding"],
                "source": query["source"],
                "notes": query.get("notes", ""),
                "split": split
            }
            rows.append(row)
    
    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id", "video_id", "video_title", "query", "query_type",
        "difficulty", "grounding", "source", "notes", "split"
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    # Calculate stats
    val_queries = sum(len(queries_by_video[vid]) for vid in val_video_ids)
    test_queries = sum(len(queries_by_video[vid]) for vid in test_video_ids)
    
    return {
        "golden_count": len(rows),
        "val_videos": len(val_video_ids),
        "val_queries": val_queries,
        "test_videos": len(test_video_ids),
        "test_queries": test_queries,
        "val_video_ids": val_video_ids,
        "test_video_ids": test_video_ids
    }


# ============================================================================
# Streamlit App
# ============================================================================

st.set_page_config(
    page_title="Query Curation",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_video_id = None
    st.session_state.video_list = get_all_video_ids(JSONL_PATH)
    st.session_state.search_filter = ""


# ============================================================================
# Sidebar - Navigation & Progress
# ============================================================================

with st.sidebar:
    st.title("Query Curation")
    
    # Progress stats
    video_list = st.session_state.video_list
    total_videos = len(video_list)
    total_queries = sum(count for _, _, count in video_list)
    golden_count = count_golden_queries(JSONL_PATH)
    golden_videos = count_golden_videos(JSONL_PATH)
    
    col1, col2 = st.columns(2)
    col1.metric("Videos", total_videos)
    col2.metric("Golden Videos", golden_videos)
    col1, col2 = st.columns(2)
    col1.metric("Queries", total_queries)
    col2.metric("Golden Queries", golden_count)
    
    st.divider()
    
    # Video search
    search = st.text_input("Search videos", value=st.session_state.search_filter)
    st.session_state.search_filter = search
    
    # Filter video list by search term
    if search:
        filtered_videos = [
            (vid, title, count) for vid, title, count in video_list
            if search.lower() in title.lower() or search.lower() in vid.lower()
        ]
    else:
        filtered_videos = video_list
    
    # Video selector
    if filtered_videos:
        options = [
            f"{vid} - {title[:40]}{'...' if len(title) > 40 else ''} ({count})"
            for vid, title, count in filtered_videos
        ]
        video_ids = [vid for vid, _, _ in filtered_videos]
        
        # Find current selection index
        current_idx = 0
        if st.session_state.current_video_id in video_ids:
            current_idx = video_ids.index(st.session_state.current_video_id)
        
        def on_video_select():
            sel = st.session_state.video_selector
            idx = options.index(sel)
            st.session_state.current_video_id = video_ids[idx]
        
        st.selectbox(
            "Select video",
            options=options,
            index=current_idx,
            key="video_selector",
            on_change=on_video_select
        )
        
        # Navigation buttons
        st.caption(f"Video {current_idx + 1} / {len(filtered_videos)}")
        
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
        
        with nav_col1:
            if st.button("⏮", help="First", key="nav_first"):
                st.session_state.current_video_id = video_ids[0]
                st.rerun()
        
        with nav_col2:
            if st.button("◀", help="Previous", key="nav_prev"):
                new_idx = (current_idx - 1) % len(video_ids)
                st.session_state.current_video_id = video_ids[new_idx]
                st.rerun()
        
        with nav_col3:
            if st.button("▶", help="Next", key="nav_next"):
                new_idx = (current_idx + 1) % len(video_ids)
                st.session_state.current_video_id = video_ids[new_idx]
                st.rerun()
        
        with nav_col4:
            if st.button("⏭", help="Last", key="nav_last"):
                st.session_state.current_video_id = video_ids[-1]
                st.rerun()
    else:
        st.warning("No videos match your search.")
    
    st.divider()
    
    # Export section
    st.subheader("Export")
    if st.button("📤 Export Golden Queries", width="stretch"):
        stats = export_golden_queries_csv(JSONL_PATH, CSV_OUTPUT_PATH)
        
        if stats["golden_count"] == 0:
            st.warning("No golden queries selected. Mark queries with the 'Golden' checkbox first.")
        else:
            st.success(f"""
            Export complete!
            - Total: {stats['golden_count']} queries
            - Validation: {stats['val_videos']} videos, {stats['val_queries']} queries
            - Test: {stats['test_videos']} videos, {stats['test_queries']} queries
            """)
            
            with st.expander("View split details"):
                st.write("**Validation videos:**", stats["val_video_ids"])
                st.write("**Test videos:**", stats["test_video_ids"])
    
    # Show last export info
    if CSV_OUTPUT_PATH.exists():
        import datetime
        mtime = CSV_OUTPUT_PATH.stat().st_mtime
        last_export = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        st.caption(f"Last export: {last_export}")
    
    st.divider()
    
    # Help section
    with st.expander("Help"):
        st.markdown("""
        **Query Types:**
        - `exact`: Factual lookup
        - `conceptual`: Understanding why/how
        - `procedural`: Step-by-step how-to
        
        **Difficulty:**
        - `grounded`: Directly in transcript
        - `medium`: Requires inference
        - `hard`: Requires synthesis
        
        **Tips:**
        - Edit cells directly in the table
        - Check 'Golden' to include in export
        - Tab to navigate, Enter to confirm
        """)


# ============================================================================
# Main Panel - Video Content
# ============================================================================

if st.session_state.current_video_id is None:
    st.info("Select a video from the sidebar to begin.")
else:
    video_id = st.session_state.current_video_id
    queries = load_queries_for_video(video_id, JSONL_PATH)
    
    if not queries:
        st.warning("No queries found for this video.")
    else:
        # Video header
        st.header(queries[0]["video_title"])
        
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.caption(f"Video ID: `{video_id}`")
            golden_in_video = sum(1 for q in queries if q.get("golden"))
            st.caption(f"{len(queries)} queries ({golden_in_video} golden)")
        with header_col2:
            st.link_button(
                "Watch on YouTube",
                f"https://youtube.com/watch?v={video_id}",
                width="stretch"
            )
        
        st.divider()
        
        # Select all / Deselect all buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
        with btn_col1:
            if st.button("✅ Select All"):
                for idx, query in enumerate(queries):
                    if not query.get("golden", False):
                        updated_query = query.copy()
                        updated_query["golden"] = True
                        update_query_in_jsonl(video_id, idx, updated_query, JSONL_PATH)
                st.session_state.video_list = get_all_video_ids(JSONL_PATH)
                st.rerun()
        with btn_col2:
            if st.button("❌ Deselect All"):
                for idx, query in enumerate(queries):
                    if query.get("golden", False):
                        updated_query = query.copy()
                        updated_query["golden"] = False
                        update_query_in_jsonl(video_id, idx, updated_query, JSONL_PATH)
                st.session_state.video_list = get_all_video_ids(JSONL_PATH)
                st.rerun()
        
        # Convert queries to DataFrame for editing
        df = pd.DataFrame(queries)
        
        # Table shows: golden, query, query_type, difficulty, notes (notes last)
        display_columns = ["golden", "query", "query_type", "difficulty", "notes"]
        # Keep only columns that exist
        display_columns = [c for c in display_columns if c in df.columns]
        table_df = df[display_columns].copy()
        
        # Store original for change detection
        original_table_df = table_df.copy()
        
        # Configure and display data editor
        edited_table_df = st.data_editor(
            table_df,
            column_config={
                "golden": st.column_config.CheckboxColumn(
                    "Golden",
                    default=False,
                    help="Mark for export to golden dataset"
                ),
                "query": st.column_config.TextColumn(
                    "Query",
                    width=500
                ),
                "query_type": st.column_config.SelectboxColumn(
                    "Type",
                    options=QUERY_TYPE_OPTIONS,
                    width="small"
                ),
                "difficulty": st.column_config.SelectboxColumn(
                    "Difficulty",
                    options=DIFFICULTY_OPTIONS,
                    width="small"
                ),
                "notes": st.column_config.TextColumn(
                    "Notes",
                    width="medium"
                ),
            },
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            key=f"editor_{video_id}"
        )
        
        # Detect and persist changes from table
        if not edited_table_df.equals(original_table_df):
            # Find changed rows
            for idx in range(len(edited_table_df)):
                if not edited_table_df.iloc[idx].equals(original_table_df.iloc[idx]):
                    # Merge table edits with full query data
                    updated_query = queries[idx].copy()
                    updated_query["golden"] = bool(edited_table_df.iloc[idx]["golden"])
                    updated_query["query"] = str(edited_table_df.iloc[idx]["query"])
                    updated_query["query_type"] = str(edited_table_df.iloc[idx]["query_type"])
                    updated_query["difficulty"] = str(edited_table_df.iloc[idx]["difficulty"])
                    updated_query["notes"] = str(edited_table_df.iloc[idx]["notes"])
                    
                    # Persist immediately
                    success = update_query_in_jsonl(
                        video_id,
                        idx,
                        updated_query,
                        JSONL_PATH
                    )
                    
                    if success:
                        st.toast(f"Saved changes to query {idx + 1}")
                    else:
                        st.error(f"Failed to save query {idx + 1}")
            
            # Refresh video list to update golden counts
            st.session_state.video_list = get_all_video_ids(JSONL_PATH)
            st.rerun()
        
        # Query details section below the table (grounding only)
        st.divider()
        st.subheader("Grounding")
        
        for idx, query in enumerate(queries):
            with st.expander(f"Query {idx + 1}: {query['query'][:60]}{'...' if len(query['query']) > 60 else ''}", expanded=False):
                st.text(query.get("grounding", ""))
        
        st.divider()
        
        # Add new query button
        if st.button("➕ Add New Query"):
            new_query = {
                "video_id": video_id,
                "video_title": queries[0]["video_title"],
                "query": "",
                "query_type": "exact",
                "difficulty": "grounded",
                "grounding": "",
                "source": "manual",
                "notes": "",
                "golden": False
            }
            append_query_to_jsonl(new_query, JSONL_PATH)
            st.session_state.video_list = get_all_video_ids(JSONL_PATH)
            st.toast("New query added!")
            st.rerun()

