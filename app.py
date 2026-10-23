"""
app.py — Flask backend for the PPO Autonomous Driving dashboard.

Serves the frontend and exposes REST endpoints for:
  • Model/result status checks
  • Triggering evaluation in a background thread
  • Fetching results and graph images
  • Live side-by-side simulation frames
  • OpenCode result explanation
"""

import json
import os
import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

import config
from live_simulation import LiveSimulationManager
from explain_results import explain_results, check_opencode


app = Flask(__name__)
app.config["SECRET_KEY"] = "ppo-driving-local-secret"

# Singleton simulation manager
sim_manager = LiveSimulationManager()

# Background evaluation state
_eval_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "error": None,
    "done": False,
}

# Background explanation state
_explain_state = {
    "running": False,
    "result": None,
    "error": None,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ok(data=None):
    return jsonify({"success": True, "data": data, "error": None})


def _err(msg: str, code: int = 400):
    return jsonify({"success": False, "data": None, "error": msg}), code


def _model_status():
    a = os.path.exists(config.AGENT_A_PATH + ".zip")
    b = os.path.exists(config.AGENT_B_PATH + ".zip")
    return {"agent_a": a, "agent_b": b, "both_ready": a and b}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    models = _model_status()
    has_results = os.path.exists(config.SUMMARY_JSON)
    has_explanation = os.path.exists(config.EXPLANATION_TXT)
    has_graphs = (
        os.path.isdir(config.GRAPHS_DIR) and
        bool(list(Path(config.GRAPHS_DIR).glob("*.png")))
    )
    opencode_ok, _ = check_opencode()
    return _ok({
        "models": models,
        "has_results": has_results,
        "has_graphs": has_graphs,
        "has_explanation": has_explanation,
        "opencode_available": opencode_ok,
        "demo_running": sim_manager.is_running,
        "eval_running": _eval_state["running"],
    })


# ─── Evaluation ───────────────────────────────────────────────────────────────

@app.route("/api/evaluate", methods=["POST"])
def start_evaluation():
    global _eval_state
    if _eval_state["running"]:
        return _err("Evaluation is already running.", 409)

    if not _model_status()["both_ready"]:
        return _err(
            "Both models must be trained before evaluation. "
            "Run: python train.py --agent A --timesteps 200000  &&  "
            "python train.py --agent B --timesteps 200000"
        )

    data = request.get_json(silent=True) or {}
    n_episodes = int(data.get("episodes", config.DEFAULT_EVAL_EPISODES))
    failure_rates = data.get("failure_rates", config.EVAL_FAILURE_RATES)

    _eval_state = {
        "running": True,
        "progress": 0,
        "total": len(failure_rates) * 2,  # 2 agents
        "message": "Starting evaluation...",
        "error": None,
        "done": False,
    }

    def _run():
        global _eval_state
        try:
            from evaluate import run_evaluation

            def _cb(current, total, msg):
                _eval_state["progress"] = current
                _eval_state["total"] = total
                _eval_state["message"] = msg

            run_evaluation(
                n_episodes=n_episodes,
                failure_rates=failure_rates,
                progress_callback=_cb,
            )
            _eval_state["done"] = True
            _eval_state["message"] = "Evaluation complete."
        except Exception as e:
            _eval_state["error"] = str(e)
            _eval_state["message"] = f"Error: {e}"
        finally:
            _eval_state["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return _ok({"message": "Evaluation started."})


@app.route("/api/evaluate/status")
def evaluation_status():
    return _ok(dict(_eval_state))


# ─── Results ──────────────────────────────────────────────────────────────────

@app.route("/api/results")
def get_results():
    if not os.path.exists(config.SUMMARY_JSON):
        return _err("No results found. Run the evaluation first.", 404)
    with open(config.SUMMARY_JSON) as f:
        data = json.load(f)
    return _ok(data)


@app.route("/api/results/graphs")
def list_graphs():
    if not os.path.isdir(config.GRAPHS_DIR):
        return _ok([])
    graphs = [f for f in os.listdir(config.GRAPHS_DIR) if f.endswith(".png")]
    return _ok(sorted(graphs))


@app.route("/api/results/graphs/<filename>")
def serve_graph(filename: str):
    return send_from_directory(config.GRAPHS_DIR, filename)


@app.route("/api/results/episodes")
def get_episodes():
    import pandas as pd
    if not os.path.exists(config.EPISODE_METRICS_CSV):
        return _err("No episode data found. Run evaluation first.", 404)
    df = pd.read_csv(config.EPISODE_METRICS_CSV)
    return _ok(df.to_dict(orient="records"))


# ─── Live Demo ────────────────────────────────────────────────────────────────

@app.route("/api/demo/start", methods=["POST"])
def demo_start():
    if not _model_status()["both_ready"]:
        return _err(
            "Both models must be trained first.\n"
            "  python train.py --agent A --timesteps 200000\n"
            "  python train.py --agent B --timesteps 200000"
        )

    data = request.get_json(silent=True) or {}
    failure_rate = float(data.get("failure_rate", 0.0))
    seed = int(data.get("seed", 42))

    try:
        sim_manager.start(failure_rate=failure_rate, seed=seed)
        return _ok({"message": f"Demo started (failure_rate={failure_rate}, seed={seed})."})
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        traceback.print_exc()
        return _err(f"Failed to start demo: {e}", 500)


@app.route("/api/demo/step")
def demo_step():
    if not sim_manager.is_running:
        return _err("Demo is not running. Call /api/demo/start first.", 400)
    try:
        frame_data = sim_manager.step()
        if frame_data is None:
            return _err("Demo stopped unexpectedly.", 500)
        return _ok(frame_data)
    except Exception as e:
        traceback.print_exc()
        return _err(f"Step failed: {e}", 500)


@app.route("/api/demo/stop", methods=["POST"])
def demo_stop():
    sim_manager.stop()
    return _ok({"message": "Demo stopped."})


@app.route("/api/demo/status")
def demo_status():
    return _ok({
        "running": sim_manager.is_running,
        "models_ready": _model_status()["both_ready"],
    })


# ─── OpenCode Explanation ─────────────────────────────────────────────────────

@app.route("/api/explain", methods=["POST"])
def start_explain():
    global _explain_state

    if _explain_state.get("running"):
        return _err("Explanation is already in progress.", 409)

    if not os.path.exists(config.SUMMARY_JSON):
        return _err(
            "No evaluation summary found. "
            "Run the full evaluation first via 'Run Full Evaluation'."
        )

    _explain_state = {"running": True, "result": None, "error": None}

    def _run():
        global _explain_state
        try:
            text = explain_results()
            _explain_state["result"] = text
        except Exception as e:
            _explain_state["error"] = str(e)
        finally:
            _explain_state["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return _ok({"message": "Explanation started. Poll /api/explain/result."})


@app.route("/api/explain/result")
def get_explanation():
    state = dict(_explain_state)
    # Also check if we have a saved explanation file
    if state.get("result") is None and os.path.exists(config.EXPLANATION_TXT):
        with open(config.EXPLANATION_TXT) as f:
            state["result"] = f.read()
    return _ok(state)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*20)
    print("  PPO Autonomous Driving — Local Dashboard")
    print("  http://127.0.0.1:5000")
    print("="*20 + "\n")

    # Check model status on startup
    ms = _model_status()
    if not ms["agent_a"]:
        print("⚠  Agent A model not found. Train with:")
        print("     python train.py --agent A --timesteps 200000\n")
    if not ms["agent_b"]:
        print("⚠  Agent B model not found. Train with:")
        print("     python train.py --agent B --timesteps 200000\n")
    if ms["both_ready"]:
        print("✓  Both models loaded and ready.")

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
