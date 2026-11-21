"""Let a runner pick up where an interrupted run stopped."""

import json
import os


def finished(path):
    """Drop a half-written last line from OUT and return the ids it already holds."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.endswith("\n")]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return {json.loads(line)["id"] for line in lines}
