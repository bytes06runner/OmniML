from openai import OpenAI
from typing import List, Optional

class LLMReporter:
    """
    Intelligence Layer for generating human-readable diagnostic reports using an LLM.
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "llama3"):
        """
        Initialize the reporter, defaulting to local, air-gapped Ollama execution.
        """
        # Using a dummy API key as it's typically required by the OpenAI client but ignored by local servers like Ollama
        self.client = OpenAI(base_url=base_url, api_key="sk-local-run-dummy")
        self.model = model

    def generate_incident_report(self, asset_id: str, mse_score: float, root_cause: str, cascade_path: List[str], mode: str = "Technical Diagnostics", feature_names: Optional[List[str]] = None) -> str:
        """
        Injects diagnostic variables into a strict system prompt and returns the LLM's diagnostic string.
        Domain-Blind Edition: Infers context structurally from telemetry feature maps.
        """
        
        features_context = ", ".join(feature_names) if feature_names else "Generic unknown variables"
        
        system_prompt = (
            "You are AnomaLLM, a universal diagnostic AI. Your objective is domain-blind operational diagnostics.\n"
            f"Review the following system variables detected in the telemetry stream: [{features_context}].\n"
            "Based on these names, deduce the industry context (e.g., Aerospace, IT Server, Biometrics, Finance) "
            "and adapt your tone, vocabulary, and analogies accordingly.\n"
            "CRITICAL DIRECTIVE: NEVER use words like 'machine', 'factory', 'turbine', or 'component' "
            "UNLESS the feature names explicitly suggest industrial physical hardware.\n\n"
            f"Your task is to analyze anomaly detection and causality outputs to generate a "
            f"{'highly technical' if mode == 'Technical Diagnostics' else 'simple, easy to understand'} incident report. "
            f"{'Explain it to me like Im 5.' if mode != 'Technical Diagnostics' else ''} Do not use filler dialogue.\n\n"
            "Format the output strictly as:\n"
            "**INCIDENT DETECTED**\n"
            "- Monitored Asset ID: [value]\n"
            "- Severity Score (MSE): [value]\n"
            "- Identified Root Cause Parameter: [value]\n"
            "- Causality Cascade Path: [value]\n\n"
            "**DIAGNOSTIC SUMMARY**\n"
            "[Provide a short, 2-3 sentence domain-specific interpretation of what this specific cascade likely means "
            "for the monitored entity's operational stability based on your inferred industry context.]"
        )
        
        cascade_str = " -> ".join(cascade_path) if cascade_path else "Isolated incident (No cascade detected)"
        
        user_prompt = (
            f"Monitored Asset ID: {asset_id}\n"
            f"MSE Score: {mse_score:.4f}\n"
            f"Root Cause Feature: {root_cause}\n"
            f"Cascade Path: {cascade_str}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Low temperature for analytical consistency
                max_tokens=250
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"**LLM Generation Failed (Air-Gapped connection timeout or model missing)**\nSystem Exception Details: {str(e)}\n\n" \
                   f"Raw Data:\nRoot Cause: {root_cause}, Cascade: {cascade_str}, MSE: {mse_score:.4f}"
