"""
explain_results.py — Invoke OpenCode CLI to generate a plain-English explanation
of evaluation results. OpenCode is strictly limited to interpretation only.

OpenCode MUST NOT:
  • Train agents or choose driving actions
  • Modify reward functions or project code
  • Invent or fabricate metrics not in the provided summary
  • Claim statistical significance without evidence

All numerical differences are pre-computed in Python (metrics.py) and passed to
OpenCode so it interprets computed values rather than inventing them.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from typing import Tuple

import config


PROMPT_TEMPLATE = """
You are an AI assistant analysing reinforcement-learning evaluation results.
The attached file (summary.json) contains the complete measured data from an experiment
comparing two PPO driving agents on the highway-v0 environment:

  • Agent A — Baseline PPO, trained WITHOUT any sensor failures.
  • Agent B — PPO trained WITH simulated sensor failures (10% per timestep dropout).

Both agents were evaluated under four sensor-failure rates: 0%, 5%, 10%, and 20%.
At each rate, 30 episodes were run with deterministic policies.

Pre-computed numerical differences (Agent B minus Agent A) are in the "comparisons" key.
Percentage changes are also pre-computed. Use ONLY these values — do not invent numbers.

Please answer the following questions based solely on the provided data:

1. What do the results show overall? Give a concise summary of each agent's performance.
2. Which agent performed better under each tested failure rate (0%, 5%, 10%, 20%)
   in terms of collision rate, average reward, and average speed?
3. Is there a safety vs speed trade-off? Does reducing collision risk come at the cost
   of lower speed or reward?
4. What do the results suggest about the hypothesis that sensor-failure training improves
   robustness? Is the evidence strong, weak, or inconclusive? Say so honestly.
5. What are the main limitations of this experiment (e.g. simulation vs real world,
   number of episodes, single traffic configuration)?
6. What further experiments or modifications would provide stronger evidence?

Rules:
  - Use ONLY the measurements in the provided JSON. Do not invent any numbers.
  - If the results are inconclusive or show no meaningful difference, say so clearly.
  - Do not claim the simulation results demonstrate real-world vehicle safety.
  - Do not compute your own statistics — use the "comparisons" values provided.
  - Respond in clear, plain English suitable for a project report. Use headers and
    bullet points where helpful.
""".strip()


def check_opencode() -> Tuple[bool, str]:
    """Return (available, path_or_error)."""
    path = shutil.which("opencode")
    if path:
        return True, path
    return False, "opencode binary not found on PATH"


def explain_results(
    summary_path: str = None,
    progress_callback=None,
) -> str:
    """
    Run OpenCode on the evaluation summary and return explanation text.

    Args:
        summary_path: Path to summary.json (defaults to config.SUMMARY_JSON).
        progress_callback: Optional callable(message: str) for UI updates.

    Returns:
        Explanation text as a string.

    Raises:
        FileNotFoundError: If summary.json does not exist.
        RuntimeError: If OpenCode is not available or fails.
    """
    if summary_path is None:
        summary_path = config.SUMMARY_JSON

    # ── Validate summary exists ───────────────────────────────────────────────
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Evaluation summary not found at {summary_path}. "
            "Please run the full evaluation first:\n"
            "  python evaluate.py --episodes 30 --failure-rates 0 0.05 0.10 0.20"
        )

    # ── Validate it is parseable JSON ─────────────────────────────────────────
    try:
        with open(summary_path) as f:
            summary_data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"summary.json is not valid JSON: {e}")

    if "results" not in summary_data:
        raise RuntimeError("summary.json is missing the 'results' key. "
                           "Re-run evaluate.py to regenerate it.")

    # ── Check OpenCode is installed ───────────────────────────────────────────
    avail, path_or_err = check_opencode()
    if not avail:
        raise RuntimeError(
            f"OpenCode is not available: {path_or_err}\n"
            "Install it from https://opencode.ai and ensure it is on your PATH."
        )

    if progress_callback:
        progress_callback("Invoking OpenCode for result interpretation...")

    # ── Build command ─────────────────────────────────────────────────────────
    # opencode run -m <provider/model> -f <file> <prompt>
    cmd = [
        "opencode", "run",
        "-m", "opencode/mimo-v2.6-flash-free",
        "-f", summary_path,
        PROMPT_TEMPLATE,
    ]

    # ── Execute ───────────────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,   # 5 minutes max
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("OpenCode timed out after 5 minutes.")
    except FileNotFoundError:
        raise RuntimeError(
            "Could not execute 'opencode'. Make sure it is installed and on PATH."
        )

    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr)"
        raise RuntimeError(
            f"OpenCode exited with code {result.returncode}.\n"
            f"Stderr: {stderr}\n\n"
            "Possible causes:\n"
            "  • No AI provider configured (run: opencode)\n"
            "  • Invalid API key\n"
            "  • Network connectivity issue"
        )

    explanation = result.stdout.strip()
    if not explanation:
        raise RuntimeError(
            "OpenCode returned an empty response. "
            "Check your provider configuration with: opencode"
        )

    # ── Save to disk ──────────────────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.EXPLANATION_TXT, "w") as f:
        f.write(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n")
        f.write(explanation)
    print(f"[explain] Explanation saved → {config.EXPLANATION_TXT}")

    return explanation


if __name__ == "__main__":
    try:
        text = explain_results()
        print("\n" + "="*60)
        print(text)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[explain] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
