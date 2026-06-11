#!/usr/bin/env python3
"""Replay real run training.log into /training-status for authentic console screenshots."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reviewer_utils import (  # noqa: E402
    DEFAULT_RUN_ID,
    build_training_state_from_run,
    render_report_html_for_run,
)

PUBLIC = ROOT / "public"
OUTPUT = ROOT / "experiments" / "output" / "reviewer_submission"

_current_state: Dict[str, Any] = {}
_replay_done = threading.Event()


def _replay_worker(run_id: str, delay_ms: int) -> None:
    global _current_state
    events_path = ROOT / "runs" / run_id / "logs" / "training.log"
    from scripts.run_reviewer_utils import parse_training_log_events

    events = parse_training_log_events(events_path)
    _current_state = build_training_state_from_run(run_id, include_complete=False, epoch_limit=0)
    _current_state["status"] = "running"
    _current_state["metrics"] = []
    _current_state["logs"] = []
    _current_state["evaluation_protocol"] = {}

    for event in events:
        etype = event.get("type")
        if etype == "hpt_trial":
            _current_state["logs"].append(
                f"HPT trial {event.get('trial')}/{event.get('total')}: value={float(event.get('value', 0)):.4f}"
            )
        elif etype == "hpt_complete":
            _current_state["best_params"] = event.get("best_params") or {}
            _current_state["logs"].append("HPT complete — best params locked")
        elif etype == "epoch_metric":
            metric = {
                "epoch": int(event.get("epoch", 0)),
                "loss": float(event.get("loss", 0)),
                "val_loss": float(event.get("val_loss", 0)),
                "acc": float(event.get("acc", 0)),
                "val_acc": float(event.get("val_acc", 0)),
            }
            _current_state["metrics"].append(metric)
            _current_state["current_epoch"] = metric["epoch"]
            _current_state["logs"].append(f"Epoch {metric['epoch']}: val_acc={metric['val_acc']:.4f}")
        time.sleep(delay_ms / 1000.0)

    complete = build_training_state_from_run(run_id, include_complete=True)
    _current_state.update(complete)
    _replay_done.set()


class ReviewerConsoleHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, run_id: str = DEFAULT_RUN_ID, **kwargs):
        self.run_id = run_id
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/training-status":
            payload = json.dumps(_current_state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path in {"/report-preview", "/report-preview/"}:
            html = render_report_html_for_run(self.run_id).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if path in {"/training_console/index.html", "/public/training_console/index.html"}:
            self.path = "/training_console/index.html"
        return super().do_GET()

    def log_message(self, format, *args):
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--delay-ms", type=int, default=900, help="Delay between replay events")
    parser.add_argument("--no-replay", action="store_true", help="Serve completed state immediately")
    parser.add_argument("--freeze-epochs", type=int, default=None, help="Serve partial epoch state (mid-training snapshot)")
    args = parser.parse_args()

    global _current_state
    if args.freeze_epochs is not None:
        _current_state = build_training_state_from_run(
            args.run_id,
            include_complete=False,
            epoch_limit=args.freeze_epochs,
        )
        _current_state["status"] = "running"
        _replay_done.set()
    elif args.no_replay:
        _current_state = build_training_state_from_run(args.run_id, include_complete=True)
        _replay_done.set()
    else:
        threading.Thread(target=_replay_worker, args=(args.run_id, args.delay_ms), daemon=True).start()

    def handler_factory(*h_args, **h_kwargs):
        return ReviewerConsoleHandler(*h_args, run_id=args.run_id, **h_kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_factory)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Reviewer console server: http://127.0.0.1:{args.port}/training_console/index.html")
    print(f"Report preview: http://127.0.0.1:{args.port}/report-preview")
    print(f"Run ID: {args.run_id}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
