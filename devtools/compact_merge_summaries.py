import glob
import json
import os
import sys

from backend.core.logic.summary_compact import compact_merge_sections


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: compact_merge_summaries.py <SID>")
        sys.exit(2)
    sid = sys.argv[1]
    root = os.path.join("runs", sid, "cases", "accounts")
    for path in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        if not name.isdigit():
            continue
        summary_path = os.path.join(path, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        compact_merge_sections(data)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    print("ok")


if __name__ == "__main__":
    main()
