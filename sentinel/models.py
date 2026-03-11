"""SENTINEL Model Registry — discover, validate, and manage models for experiments.

Provides model discovery from Ollama, system-aware capability profiling, and
model matrix generation for running experiments across multiple models.

System awareness: detects available memory, GPU type, and known hardware
constraints (e.g., Jetson NvMap single-allocation limits) to prevent users
from selecting models that will crash or leak GPU memory.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ollama import OllamaClient

log = logging.getLogger("sentinel.models")

# VRAM overhead multiplier: model file size * this = estimated VRAM at runtime.
# Accounts for KV cache, CUDA context, activations. Conservative for safety.
VRAM_OVERHEAD_FACTOR = 1.25

# Minimum headroom (bytes) to leave free after loading a model.
# Prevents OOM during inference (context growth, batch allocations).
MIN_HEADROOM_BYTES = 512 * 1024 * 1024  # 512 MB

# Known hardware constraints that can't be detected from /proc.
# Jetson NvMap can't do single CUDA allocations above this threshold.
# Models requiring larger single allocs will panic Ollama and leak NvMap state.
NVMAP_SINGLE_ALLOC_LIMIT = 3.5 * 1024 * 1024 * 1024  # ~3.5 GiB


# ── System specs ─────────────────────────────────────────────────

@dataclass
class SystemSpecs:
    """Hardware specs relevant to model selection."""
    total_ram_bytes: int = 0
    available_ram_bytes: int = 0
    gpu_name: str = "unknown"
    is_jetson: bool = False
    is_unified_memory: bool = False
    max_single_alloc_bytes: int = 0      # NvMap limit (Jetson) or 0 (no limit)
    max_model_vram_bytes: int = 0        # largest model we can safely load
    platform: str = "unknown"

    @property
    def total_ram_gb(self) -> float:
        return round(self.total_ram_bytes / (1024 ** 3), 1)

    @property
    def available_ram_gb(self) -> float:
        return round(self.available_ram_bytes / (1024 ** 3), 1)

    @property
    def max_model_vram_gb(self) -> float:
        return round(self.max_model_vram_bytes / (1024 ** 3), 1)


def get_system_specs() -> SystemSpecs:
    """Detect hardware specs from /proc, /sys, and device tree."""
    specs = SystemSpecs()

    # Memory from /proc/meminfo
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                specs.total_ram_bytes = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                specs.available_ram_bytes = int(line.split()[1]) * 1024
    except OSError:
        pass

    # Detect Jetson from device tree
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00\n")
        if "Jetson" in model:
            specs.is_jetson = True
            specs.is_unified_memory = True
            specs.gpu_name = model
            specs.platform = "jetson"
            # NvMap single-allocation limit — the real constraint on Jetson.
            # This is NOT configurable and cannot be detected at runtime.
            specs.max_single_alloc_bytes = int(NVMAP_SINGLE_ALLOC_LIMIT)
        else:
            specs.gpu_name = model
    except OSError:
        pass

    # If not Jetson, try to detect discrete GPU via /proc
    if not specs.is_jetson:
        try:
            # Check for NVIDIA discrete GPU
            with open("/proc/driver/nvidia/gpus/0000:01:00.0/information") as f:
                for line in f:
                    if "Model:" in line:
                        specs.gpu_name = line.split(":", 1)[1].strip()
                        specs.platform = "nvidia-discrete"
        except OSError:
            specs.platform = "cpu-only"

    # Compute max model size we can safely load
    if specs.is_jetson:
        # Jetson: unified memory, constrained by both total available AND single alloc limit
        usable = specs.available_ram_bytes - MIN_HEADROOM_BYTES
        alloc_limit = specs.max_single_alloc_bytes
        # Model VRAM ≈ file_size * overhead, so max file size = limit / overhead
        max_from_alloc = int(alloc_limit / VRAM_OVERHEAD_FACTOR)
        max_from_ram = int(usable / VRAM_OVERHEAD_FACTOR)
        specs.max_model_vram_bytes = min(max_from_alloc, max_from_ram)
    else:
        # Discrete GPU or CPU: just use available RAM as estimate
        usable = specs.available_ram_bytes - MIN_HEADROOM_BYTES
        specs.max_model_vram_bytes = max(usable, 0)

    return specs


# ── Model compatibility ──────────────────────────────────────────

@dataclass
class ModelCompatibility:
    """A model with its compatibility assessment."""
    model: 'ModelInfo'
    fits: bool
    reason: str = ""
    estimated_vram_gb: float = 0.0
    margin_gb: float = 0.0  # how much headroom remains (negative = won't fit)

    @property
    def status(self) -> str:
        if self.fits:
            if self.margin_gb < 0.5:
                return "TIGHT"
            return "OK"
        return "NO"

    @property
    def status_icon(self) -> str:
        s = self.status
        if s == "OK":
            return "[OK]"
        elif s == "TIGHT":
            return "[!!]"
        return "[NO]"


def estimate_model_vram(model: 'ModelInfo') -> int:
    """Estimate VRAM needed to load and run a model (bytes)."""
    return int(model.size_bytes * VRAM_OVERHEAD_FACTOR)


def check_model_compatibility(
    model: 'ModelInfo', specs: SystemSpecs,
) -> ModelCompatibility:
    """Check if a model can run on this system."""
    est_vram = estimate_model_vram(model)
    est_vram_gb = est_vram / (1024 ** 3)
    max_vram = specs.max_model_vram_bytes
    margin = (max_vram - model.size_bytes) / (1024 ** 3)

    # Check NvMap single-alloc limit (Jetson-specific)
    if specs.is_jetson and specs.max_single_alloc_bytes > 0:
        if model.size_bytes > specs.max_single_alloc_bytes:
            return ModelCompatibility(
                model=model, fits=False,
                reason=f"Exceeds NvMap single-alloc limit ({specs.max_single_alloc_bytes / (1024**3):.1f} GiB). "
                       f"Will panic Ollama and leak GPU memory.",
                estimated_vram_gb=round(est_vram_gb, 1),
                margin_gb=round(margin, 1),
            )

    # Check total available VRAM
    if model.size_bytes > max_vram:
        return ModelCompatibility(
            model=model, fits=False,
            reason=f"Model ({model.size_gb} GB) exceeds available capacity ({specs.max_model_vram_gb} GB with headroom)",
            estimated_vram_gb=round(est_vram_gb, 1),
            margin_gb=round(margin, 1),
        )

    reason = ""
    if margin < 0.5:
        reason = f"Tight fit — only {margin:.1f} GB headroom"

    return ModelCompatibility(
        model=model, fits=True, reason=reason,
        estimated_vram_gb=round(est_vram_gb, 1),
        margin_gb=round(margin, 1),
    )


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


def get_compatible_models(
    client: OllamaClient, specs: SystemSpecs = None,
) -> list[ModelCompatibility]:
    """Get all available models with compatibility assessments."""
    if specs is None:
        specs = get_system_specs()
    models = discover_models(client)
    return [check_model_compatibility(m, specs) for m in models]


def filter_compatible(compatibilities: list[ModelCompatibility]) -> list[ModelInfo]:
    """Return only models that fit on this system."""
    return [c.model for c in compatibilities if c.fits]


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
