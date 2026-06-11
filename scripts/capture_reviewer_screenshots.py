#!/usr/bin/env python3
"""Capture Training Console and report screenshots via Playwright."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "experiments" / "output" / "reviewer_submission" / "screenshots"


def _ensure_playwright() -> None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)


def capture(
    port: int,
    out_dir: Path,
    *,
    run_id: str = "breast_cancer_biopsy_c795d4ae",
    mid_port: int = 8766,
) -> None:
    _ensure_playwright()
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    complete_base = f"http://127.0.0.1:{port}"
    mid_base = f"http://127.0.0.1:{mid_port}"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        page.goto(f"{mid_base}/training_console/index.html", wait_until="networkidle")
        page.wait_for_timeout(800)
        page.screenshot(path=str(out_dir / "training_console_mid.png"), full_page=True)

        page.goto(f"{complete_base}/training_console/index.html", wait_until="networkidle")
        page.wait_for_function(
            """() => {
                const panel = document.getElementById('eval-panel');
                return panel && panel.classList.contains('visible');
            }""",
            timeout=20000,
        )
        page.wait_for_timeout(400)
        page.screenshot(path=str(out_dir / "training_console_complete.png"), full_page=True)

        page.goto(f"{complete_base}/report-preview", wait_until="networkidle")
        page.wait_for_timeout(500)
        page.screenshot(path=str(out_dir / "final_report_ui.png"), full_page=True)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.35)")
        page.wait_for_timeout(300)
        page.screenshot(path=str(out_dir / "final_report_tables.png"), full_page=False)

        browser.close()

    print(f"Screenshots written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mid-port", type=int, default=8766)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    capture(args.port, args.out_dir, mid_port=args.mid_port)


if __name__ == "__main__":
    main()
