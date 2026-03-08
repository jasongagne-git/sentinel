"""SENTINEL Model Registry — discover, validate, and manage models for experiments.

Provides model discovery from Ollama, capability profiling, and model matrix
generation for running the same experiment across multiple model configurations.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from .ollama import OllamaClient

log = logging.getLogger("sentinel.models")


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    digest: str
    family: str
    parameter_size: str
    quantization: str
    size_bytes: int

    @property
    def short_digest(self) -> str:
        return self.digest[:12]

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024 ** 3), 1)


def discover_models(client: OllamaClient) -> list[ModelInfo]:
    """Discover all models available on the local Ollama instance."""
    raw = client.list_models()
    models = []
    for m in raw:
        details = m.get("details", {})
        models.append(ModelInfo(
            name=m["name"],
            digest=m.get("digest", "unknown"),
            family=details.get("family", "unknown"),
            parameter_size=details.get("parameter_size", "unknown"),
            quantization=details.get("quantization_level", "unknown"),
            size_bytes=m.get("size", 0),
        ))
    return models


def validate_model(client: OllamaClient, model_name: str) -> Optional[ModelInfo]:
    """Check if a specific model is available. Returns ModelInfo or None."""
    models = discover_models(client)
    for m in models:
        if m.name == model_name:
            return m
    return None


def generate_model_matrix(
    base_config: dict,
    models: list[str],
    mode: str = "homogeneous",
) -> list[dict]:
    """Generate experiment configs for a model matrix.

    Args:
        base_config: Base experiment config dict (with agents, topology, etc.)
        models: List of model names to test
        mode: Matrix mode:
            - "homogeneous": One experiment per model, all agents use that model
            - "mixed": All possible per-agent model combinations (combinatorial)
            - "asymmetric": One agent uses each model while others use the base model

    Returns:
        List of experiment config dicts, each a variant of the base config
    """
    configs = []

    if mode == "homogeneous":
        # Simple: run the same experiment with each model
        for model in models:
            config = json.loads(json.dumps(base_config))  # deep copy
            config["default_model"] = model
            config["name"] = f"{base_config['name']} [{model}]"
            config["description"] = (
                f"{base_config.get('description', '')} "
                f"Model variant: all agents using {model}."
            )
            # Remove any per-agent model overrides so default applies
            for agent in config["agents"]:
                agent.pop("model", None)
            configs.append(config)

    elif mode == "asymmetric":
        # Each agent gets a different model, one at a time
        base_model = base_config.get("default_model", models[0])
        for i, agent_def in enumerate(base_config["agents"]):
            for model in models:
                if model == base_model:
                    continue  # Skip if same as default
                config = json.loads(json.dumps(base_config))
                config["agents"][i]["model"] = model
                agent_name = agent_def["name"]
                config["name"] = f"{base_config['name']} [{agent_name}={model}]"
                config["description"] = (
                    f"{base_config.get('description', '')} "
                    f"Asymmetric variant: {agent_name} uses {model}, others use {base_model}."
                )
                configs.append(config)

    elif mode == "mixed":
        # Full combinatorial — all agents × all models
        import itertools
        agent_count = len(base_config["agents"])
        for combo in itertools.product(models, repeat=agent_count):
            config = json.loads(json.dumps(base_config))
            names = []
            for j, model in enumerate(combo):
                config["agents"][j]["model"] = model
                names.append(f"{config['agents'][j]['name']}={model}")
            config["name"] = f"{base_config['name']} [{', '.join(names)}]"
            config["description"] = (
                f"{base_config.get('description', '')} "
                f"Mixed variant: {', '.join(names)}."
            )
            configs.append(config)

    else:
        raise ValueError(f"Unknown matrix mode: {mode}. Use 'homogeneous', 'asymmetric', or 'mixed'.")

    return configs


def print_model_matrix_preview(configs: list[dict]):
    """Print a preview of what a model matrix will run."""
    print(f"\nModel Matrix: {len(configs)} experiment(s)")
    print("-" * 70)
    for i, config in enumerate(configs, 1):
        agents_desc = []
        default_model = config.get("default_model", "?")
        for agent in config["agents"]:
            model = agent.get("model", default_model)
            agents_desc.append(f"{agent['name']}→{model}")
        print(f"  {i}. {config['name']}")
        print(f"     {', '.join(agents_desc)}")
