import argparse
import json
from pathlib import Path


def main(path: str):
    """
    Read a run log (JSONL) and summarize ε-spending statistics.
    """
    p = Path(path)
    if not p.exists():
        print("No run log found:", path)
        return

    spent = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            if "eps_spent" in obj:
                spent.append(obj["eps_spent"])
        except Exception:
            # Ignore malformed lines
            pass

    if spent:
        print(f"Runs: {len(spent)} | mean ε: {sum(spent) / len(spent):.2f} | max ε: {max(spent):.2f}")
    else:
        print("No ε entries found.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summarize epsilon spending from a run log.")
    ap.add_argument("--log", default="runs/last_run.jsonl", help="Path to run log JSONL file")
    args = ap.parse_args()
    main(args.log)
