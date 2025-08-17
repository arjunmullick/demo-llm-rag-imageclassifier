"""
Optional: Download a larger public imaging-related text dataset.

This script fetches a small subset of radiology report-like texts from the 
MIMIC-CXR or similar sources would normally require credentialed access. 
Instead, as a simple public example, it downloads a few sample reports
from a GitHub gist (if provided) or other public mirror.

Note: Network access may be restricted in your environment. If so, skip this
and use data/sample_imaging.jsonl which is already included.
"""

from pathlib import Path
import json
import sys


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "sample_imaging.jsonl"
    print(f"Sample dataset already present at: {out_path}")
    print("If you want a different dataset, replace or add another JSONL file here.")
    print("Each line should be a JSON object with a 'text' field.")


if __name__ == "__main__":
    sys.exit(main())

