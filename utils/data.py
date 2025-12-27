"""Data loading utilities for experiments."""

import csv
from pathlib import Path


def load_queries(path: Path | str, split: str) -> list[dict]:
    """Load queries from CSV, filtered by split.

    Args:
        path: Path to queries CSV file
        split: Split to filter by (e.g., 'dev', 'test')

    Returns:
        List of query dicts matching the specified split
    """
    queries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                queries.append(row)
    return queries


def load_prompt(prompts_dir: Path | str, name: str) -> str:
    """Load a prompt template from a prompts directory.

    Args:
        prompts_dir: Path to prompts directory
        name: Prompt name (without 'answer_' prefix or '.md' suffix)

    Returns:
        Prompt template content as string
    """
    prompts_dir = Path(prompts_dir)
    prompt_path = prompts_dir / f"answer_{name}.md"
    return prompt_path.read_text()

