from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.plots.config import RESULTS_DIR


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    commands = [
        [sys.executable, str(REPO_ROOT / "scripts" / "train_models.py")],
        [sys.executable, str(REPO_ROOT / "scripts" / "feature_selection.py")],
        [sys.executable, str(REPO_ROOT / "scripts" / "train_state_of_the_art_models.py")],
        [sys.executable, str(REPO_ROOT / "scripts" / "train_mixture_of_experts.py")],
    ]

    for command in commands:
        run_command(command)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands": [" ".join(command) for command in commands],
    }
    (RESULTS_DIR / "result_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
