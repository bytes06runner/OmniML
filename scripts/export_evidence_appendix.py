#!/usr/bin/env python3
"""Export a markdown evidence appendix from a run manifest.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_appendix(manifest_path: Path, output_path: Path) -> None:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    lines = [
        "# Evidence Appendix",
        "",
        f"- Run ID: `{manifest.get('run_id', 'unknown')}`",
        f"- User query: {manifest.get('user_query', '')}",
        f"- Evidence version: {manifest.get('evidence_version', '')}",
        "",
        "## Task representation",
        "",
        "```json",
        json.dumps(manifest.get("metadata", {}).get("task_representation", {}), indent=2),
        "```",
        "",
        "## Artifact refs",
        "",
    ]
    for ref in manifest.get("artifact_refs", []):
        lines.append(f"- **{ref.get('name')}** ({ref.get('kind')}): `{ref.get('path')}`")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(manifest.get("metadata", {}), indent=2))
    lines.append("```")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="Path to runs/<id>/manifest.json")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.manifest.parent / "evidence_appendix.md"
    export_appendix(args.manifest, output)


if __name__ == "__main__":
    main()
