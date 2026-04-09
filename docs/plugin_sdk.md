# OmniML Plugin SDK

## Manifest

Each plugin package must include a `plugin.json` manifest with:

- `name`
- `version`
- `module`
- `class_name`
- `slot`
- `api_version`
- optional `description`, `task_types`, `compliance_impact`, `config_schema`

## Supported Slots

- `pre_run`
- `dataset_connector`
- `preprocessor`
- `feature_engineering`
- `post_training_evaluator`
- `evidence_provider`

## Python Contract

Plugins subclass one of the base classes from `anomallm.plugins`:

- `PreRunPlugin`
- `DatasetConnectorPlugin`
- `PreprocessorPlugin`
- `FeatureEngineeringPlugin`
- `PostTrainingEvaluatorPlugin`
- `EvidenceProviderPlugin`

Each plugin implements:

```python
def run(self, context: PluginContext, config: dict | None = None) -> dict:
    ...
```

## Lifecycle

1. OmniML discovers manifests under `plugins/**/plugin.json`.
2. UI lists installed plugins and allows enablement by name.
3. Enabled plugins execute only for their declared slot.
4. Execution results and failures are recorded per run.

## Evidence Contribution

`evidence_provider` plugins may return additional evidence payloads. These are recorded in plugin execution records and can be surfaced in compliance summaries.

## Failure Handling

- Plugin failures are isolated and recorded.
- Plugins do not mutate graph topology.
- A plugin only becomes blocking if its class sets `required = True`.
