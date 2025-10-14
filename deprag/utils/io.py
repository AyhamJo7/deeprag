import json
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
