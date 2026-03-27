from pathlib import Path
import json
from typing import Any

import pandas as pd


def load_text(path: Path, encoding: str = "utf-8") -> str:
    """Load text from a file.

    Args:
        path: Path to the text file
        encoding: File encoding (default: "utf-8")

    Returns:
        Text content of the file
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding=encoding)


def load_json(path: Path, encoding: str = "utf-8") -> Any:
    """
    Load JSON data from a file.

    Args:
        path: Path to the JSON file
        encoding: File encoding (default: "utf-8")
        
    Returns:
        Parsed JSON data
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding=encoding) as f:
        return json.load(f)
    

def load_csv_corpus(
    csv_path: Path,
    chunk_size: int = 100_000,
    max_rows: int | None = None
) -> list[dict]:
    """Load CSV corpus into list of dicts.
    
    Args:
        csv_path: Path to CSV file with 'citation' and 'text' columns
        chunk_size: Rows to process per chunk (for memory efficiency)
        max_rows: Optional limit on rows (for testing with smaller corpus)
    
    Returns:
        List of {"citation": str, "text": str} dicts
    """
    documents = []
    
    with open(csv_path, encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1
    
    if max_rows:
        total_rows = min(total_rows, max_rows)
    print(f"Total rows to load: {total_rows:,}")
    
    rows_loaded = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            if max_rows and rows_loaded >= max_rows:
                break
            documents.append({
                "citation": str(row["citation"]),
                "text": str(row["text"]) if pd.notna(row["text"]) else ""
            })
            rows_loaded += 1
        if max_rows and rows_loaded >= max_rows:
            break
    
    return documents
