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
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Tuple

# Strip ANSI escape codes from OpenCode's terminal output
_ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

import config


PROMPT_TEMPLATE = """
Analyse these PPO driving agent results. For each question below, use it as a heading and answer in 2-3 sentences max. Use only the numbers in the data. Do not invent figures.

Agent A = baseline PPO, no sensor failures during training.
Agent B = PPO with curriculum sensor dropout + safety reward (collision penalty x5, halved speed reward).
Evaluated at 0%, 5%, 10%, 20% sensor failure rates. "comparisons" key has pre-computed deltas.

Q1: Which agent is safer and by how much?
Q2: What does Agent B give up to achieve lower collision rates?
Q3: Is the improvement from dropout training, reward shaping, or impossible to tell from this data alone?
Q4: Why does Agent B sometimes score lower reward than A even at 0% failure — is that a flaw or expected?
Q5: Can these results be trusted, or are the episode counts too small to draw conclusions?
Q6: What single experiment would most strengthen or disprove this study's main claim?
Q7: If this policy were deployed on a real vehicle, what is the one result here that should concern an engineer most?
Q8: What does the speed-reward pattern across all four failure rates reveal about what Agent B actually learned?
Q9: Does the data support the claim that sensor-failure training produces a more robust policy, or just a more cautious one?
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
    save_to: str = None,
) -> str:
    """
    Run OpenCode on the evaluation summary and return explanation text.

    Args:
        summary_path: Path to summary.json (defaults to config.SUMMARY_JSON).
        progress_callback: Optional callable(message: str) for UI updates.
        save_to: Optional path to persist the explanation. Use this ONLY for
            the canonical baseline (results/baseline_explanation.txt), which is
            generated once and served from disk thereafter. Experiment
            explanations are ephemeral and must NOT be saved.

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

    # ── Build prompt — inline the JSON data directly (no -f flag) ────────────
    with open(summary_path) as f:
        json_text = f.read()
    full_prompt = PROMPT_TEMPLATE + "\n\n---\nDATA (summary.json):\n" + json_text

    # ── Build command — matches the working MAD-2 pattern exactly ─────────────
    # opencode run --model <provider/model> <prompt>
    cmd = [
        "opencode", "run",
        "--model", "opencode/mimo-v2.6-flash-free",
        full_prompt,
    ]

    # ── Execute ───────────────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,   # 5 minutes max
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("OpenCode timed out after 5 minutes.")
    except FileNotFoundError:
        raise RuntimeError(
            "Could not execute 'opencode'. Make sure it is installed and on PATH."
        )

    if result.returncode != 0:
        stderr = _ANSI.sub('', result.stderr).strip() or "(no stderr)"
        raise RuntimeError(
            f"OpenCode exited with code {result.returncode}.\n"
            f"Stderr: {stderr}\n\n"
            "Possible causes:\n"
            "  • No AI provider configured (run: opencode)\n"
            "  • Invalid API key\n"
            "  • Network connectivity issue"
        )

    explanation = _ANSI.sub('', result.stdout).strip()
    if not explanation:
        raise RuntimeError(
            "OpenCode returned an empty response. "
            "Check your provider configuration with: opencode"
        )

    # Persist ONLY when explicitly requested (canonical baseline explanation).
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, "w") as f:
            f.write(f"""# OpenCode Interpretation — Canonical Baseline (30 Episodes)
<!-- Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')} | Model: mimo-v2.6-flash-free -->
<!-- Source: results/summary.json (frozen post-training evaluation) | Do not edit by hand; regenerate with: python explain_results.py -->

""")
            f.write(explanation)
        print(f"[explain] Baseline explanation saved → {save_to}")

    return explanation


if __name__ == "__main__":
    try:
        # CLI runs explain the canonical baseline and persist its explanation
        # to the repo so the dashboard can serve it instantly.
        text = explain_results(save_to=config.BASELINE_EXPLANATION_TXT)
        print("\n" + "="*60)
        print(text)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[explain] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
