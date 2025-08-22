import os
import re
from typing import List
from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DEFAULT_INPUT = os.path.join(REPO_ROOT, "data", "bible_fulfillment.md")
OUT_PATH = os.path.join(REPO_ROOT, "data", "bible_fulfillment_structured.txt")


def extract_pairs(md_text: str) -> List[str]:
    lines: List[str] = []
    # Accept many arrow variants and separators; capture two scripture refs
    ref = r"[A-Za-z1-3 ]+\d+:\d+(?:-\d+)?"
    pattern = re.compile(rf"({ref})\s*(?:->|=>|→|⇒|—>|-\>|\t)\s*({ref})")
    for raw in md_text.splitlines():
        m = pattern.search(raw)
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
            # Keep as-is; downstream builder assumes NT -> OT direction
            # We'll auto-swap in the builder if needed
            lines.append(f"{left} -> {right}")
    return lines


def main(inp: str = DEFAULT_INPUT, out: str = OUT_PATH) -> None:
    if not os.path.exists(inp):
        raise FileNotFoundError(inp)
    with open(inp, "r") as f:
        md = f.read()
    lines = extract_pairs(md)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Wrote {len(lines)} lines to {out}")


if __name__ == "__main__":
    main()


