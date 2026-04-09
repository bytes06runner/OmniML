from __future__ import annotations

from anomallm.plugins import EvidenceProviderPlugin, PluginContext


class ExampleEvidencePlugin(EvidenceProviderPlugin):
    slot = "evidence_provider"

    def run(self, context: PluginContext, config=None):
        note = (config or {}).get("note", {}).get("default", "Example plugin evidence injected.")
        return {
            "evidence_block": {
                "title": "Example Plugin Evidence",
                "body": note,
            }
        }
