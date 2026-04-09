from __future__ import annotations

import importlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schemas import PluginArtifacts, PluginExecutionRecord, PluginManifest


PLUGIN_SLOTS = {
    "pre_run",
    "dataset_connector",
    "preprocessor",
    "feature_engineering",
    "post_training_evaluator",
    "evidence_provider",
}
PLUGIN_API_VERSION = "1.0"


class PluginContext:
    def __init__(self, state: Dict[str, Any]):
        self.state = state


class PluginNode:
    slot = "post_training_evaluator"
    required = False

    def run(self, context: PluginContext, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {}


class PreRunPlugin(PluginNode):
    slot = "pre_run"


class DatasetConnectorPlugin(PluginNode):
    slot = "dataset_connector"


class PreprocessorPlugin(PluginNode):
    slot = "preprocessor"


class FeatureEngineeringPlugin(PluginNode):
    slot = "feature_engineering"


class PostTrainingEvaluatorPlugin(PluginNode):
    slot = "post_training_evaluator"


class EvidenceProviderPlugin(PluginNode):
    slot = "evidence_provider"


class PluginRegistry:
    def __init__(self, plugin_dir: Optional[str] = None):
        self.plugin_dir = plugin_dir or os.path.join(os.getcwd(), "plugins")

    def discover(self) -> List[PluginManifest]:
        manifests: List[PluginManifest] = []
        if not os.path.isdir(self.plugin_dir):
            return manifests
        for root, _, files in os.walk(self.plugin_dir):
            if "plugin.json" not in files:
                continue
            path = os.path.join(root, "plugin.json")
            with open(path, "r", encoding="utf-8") as handle:
                manifest = PluginManifest.model_validate(json.load(handle))
            if manifest.slot in PLUGIN_SLOTS:
                manifests.append(manifest)
        return manifests

    def load(self, manifest: PluginManifest) -> PluginNode:
        if manifest.api_version != PLUGIN_API_VERSION:
            raise RuntimeError(
                f"Plugin '{manifest.name}' targets API version {manifest.api_version}, but OmniML expects {PLUGIN_API_VERSION}."
            )
        module = importlib.import_module(manifest.module)
        cls = getattr(module, manifest.class_name)
        plugin = cls()
        if getattr(plugin, "slot", manifest.slot) != manifest.slot:
            raise RuntimeError(f"Plugin '{manifest.name}' declared slot '{manifest.slot}' but implements '{getattr(plugin, 'slot', None)}'.")
        return plugin

    def execute_slot(
        self,
        slot: str,
        state: Dict[str, Any],
        plugin_artifacts: Optional[PluginArtifacts] = None,
        enabled_plugins: Optional[List[str]] = None,
    ) -> PluginArtifacts:
        artifacts = plugin_artifacts or PluginArtifacts()
        manifests = self.discover()
        artifacts.discovered = manifests
        artifacts.catalog = manifests
        allowlist = set(enabled_plugins or [])
        active = [manifest for manifest in manifests if not allowlist or manifest.name in allowlist]
        artifacts.enabled_plugins = [manifest.name for manifest in active]
        for manifest in active:
            if manifest.slot != slot:
                continue
            record = PluginExecutionRecord(plugin_name=manifest.name, slot=slot, status="running")
            plugin: Optional[PluginNode] = None
            try:
                plugin = self.load(manifest)
                result = plugin.run(PluginContext(state), manifest.config_schema)
                record.status = "completed"
                record.details = result or {}
            except Exception as exc:
                record.status = "failed"
                record.details = {"error": str(exc)}
                if getattr(plugin, "required", False):
                    raise
            record.finished_at = datetime.utcnow()
            artifacts.executions.append(record)
        return artifacts
