"""
HTTP bridge between the Node.js vision server and the triage logic.
Start with:  python triage_http.py
server.js will POST each frame result to http://localhost:8002/triage automatically.
"""
import os
import json

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
except ImportError:
    pass

from flask import Flask, request, jsonify
from triage_logic import adapt_frame, check_significant_change, run_triage

app    = Flask(__name__)
_state = {"last_scene": None, "last_triage": None}


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/triage", methods=["POST"])
def triage():
    rich = request.get_json(force=True, silent=True)
    if not rich:
        return jsonify({"error": "Missing or invalid JSON body"}), 400

    # Auto-detect schema: rich schema has "people" or "injuries"; flat has "people_count"
    frame = adapt_frame(rich) if ("people" in rich or "injuries" in rich) else rich

    last = _state["last_scene"]
    if last is None:
        should_alert = True
        reason       = "Initial scene assessment."
    else:
        result       = check_significant_change(last, frame)
        should_alert = result.get("significant_change", False)
        reason       = result.get("reason", "")

    _state["last_scene"] = frame

    if not should_alert:
        return jsonify({"alert": False, "reason": reason, "last_triage": _state["last_triage"]})

    triage_text = run_triage(json.dumps(frame))
    _state["last_triage"] = triage_text
    return jsonify({"alert": True, "triage": triage_text, "reason": reason})


if __name__ == "__main__":
    print("Triage HTTP server starting on http://localhost:8002")
    app.run(port=8002, debug=False)
