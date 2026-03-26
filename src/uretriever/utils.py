from pathlib import Path
import json
from typing import Any


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