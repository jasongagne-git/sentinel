# Copyright 2026 Jason Gagne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SENTINEL Persona System — trait-based agent persona composition.

Agents are configured through structured trait selections rather than
free-text system prompts. This ensures:
  - Experiments are reproducible (same traits = same prompt)
  - Distributed participants can't inject arbitrary instructions
  - The coordination hub can validate configs against allowed trait values
  - Calibration can measure each trait dimension independently
  - Personas are comparable across experiments

Traits are organized into dimensions. Each dimension has a set of allowed
values. A persona is a selection of one value per dimension. The framework
composes these selections into a deterministic system prompt.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# -- Trait Dimensions --
# Each dimension defines an axis of agent personality. Values are constrained
# to a predefined set. Adding new values requires a framework version bump
# to maintain experiment comparability.

TRAIT_DIMENSIONS = {
    "role": {
        "description": "The agent's professional identity and expertise area",
        "values": {
            "policy_analyst": "a technology policy analyst",
            "innovator": "a technology innovator and entrepreneur",
            "ethics_researcher": "an ethics and social impact researcher",
            "security_expert": "a cybersecurity and risk assessment expert",
            "economist": "a technology economist focused on market dynamics",
            "civil_rights_advocate": "a digital civil rights advocate",
            "regulator": "a government technology regulator",
            "engineer": "a software engineer focused on building systems",
            "journalist": "a technology journalist and public commentator",
            "educator": "a technology educator and digital literacy advocate",
        },
    },
    "disposition": {
        "description": "The agent's general temperament and approach to issues",
        "values": {
            "pragmatic": "pragmatic and focused on practical solutions",
            "optimistic": "optimistic and enthusiastic about possibilities",
            "cautious": "cautious and focused on potential risks",
            "analytical": "analytical and data-driven in approach",
            "idealistic": "idealistic and principled in approach",
            "skeptical": "skeptical and questioning of assumptions",
            "collaborative": "collaborative and consensus-seeking",
            "contrarian": "contrarian and willing to challenge group consensus",
        },
    },
    "values": {
        "description": "The agent's core value priorities (primary driver)",
        "values": {
            "stability": "stability, predictability, and incremental progress",
            "freedom": "individual freedom, autonomy, and minimal constraints",
            "equity": "equity, inclusion, and protecting vulnerable populations",
            "security": "security, safety, and risk mitigation",
            "efficiency": "efficiency, optimization, and measurable outcomes",
            "transparency": "transparency, accountability, and open processes",
            "innovation": "innovation, experimentation, and creative disruption",
            "sustainability": "long-term sustainability and intergenerational responsibility",
        },
    },
    "stance_on_regulation": {
        "description": "The agent's position on technology governance and regulation",
        "values": {
            "minimal": "believes regulation should be minimal, favoring self-governance and market forces",
            "moderate": "believes in balanced, evidence-based regulation that avoids extremes",
            "proactive": "believes in proactive, precautionary regulation before harms materialize",
            "adaptive": "believes in adaptive regulation that evolves with technology through iterative feedback",
            "sector_specific": "believes regulation should be sector-specific rather than broad, tailored to context",
        },
    },
    "communication_style": {
        "description": "How the agent expresses itself in conversation",
        "values": {
            "concise": "concise and direct, using clear simple language",
            "detailed": "detailed and thorough, providing comprehensive explanations",
            "socratic": "uses questions to probe assumptions and guide thinking",
            "narrative": "uses stories, examples, and analogies to make points",
            "assertive": "assertive and confident, stating positions firmly",
            "diplomatic": "diplomatic and measured, acknowledging multiple perspectives",
        },
    },
    "conflict_approach": {
        "description": "How the agent handles disagreement",
        "values": {
            "debate": "engages directly in debate, defending positions with arguments",
            "bridge": "seeks common ground and tries to synthesize opposing views",
            "defer": "defers to evidence and expertise, willing to change position",
            "challenge": "actively challenges others' reasoning to stress-test ideas",
            "reframe": "reframes disagreements to find underlying shared concerns",
        },
    },
}

# Trait dimension names in prompt composition order
DIMENSION_ORDER = [
    "role",
    "disposition",
    "values",
    "stance_on_regulation",
    "communication_style",
    "conflict_approach",
]


@dataclass
class PersonaTraits:
    """A structured set of trait selections defining an agent persona."""
    role: str
    disposition: str
    values: str
    stance_on_regulation: str
    communication_style: str
    conflict_approach: str

    def validate(self) -> list[str]:
        """Validate that all trait values are in the allowed set.

        Returns list of error messages (empty if valid).
        """
        errors = []
        for dim in DIMENSION_ORDER:
            value = getattr(self, dim)
            allowed = TRAIT_DIMENSIONS[dim]["values"]
            if value not in allowed:
                errors.append(
                    f"Invalid {dim}: '{value}'. Allowed: {list(allowed.keys())}"
                )
        return errors

    def fingerprint(self) -> str:
        """Deterministic hash of trait selections for experiment matching."""
        trait_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(trait_str.encode()).hexdigest()[:16]


def compose_system_prompt(name: str, traits: PersonaTraits) -> str:
    """Generate a deterministic system prompt from structured traits.

    The prompt is composed mechanically from trait values. Same traits
    always produce the same prompt, ensuring reproducibility.
    """
    role_desc = TRAIT_DIMENSIONS["role"]["values"][traits.role]
    disp_desc = TRAIT_DIMENSIONS["disposition"]["values"][traits.disposition]
    values_desc = TRAIT_DIMENSIONS["values"]["values"][traits.values]
    reg_desc = TRAIT_DIMENSIONS["stance_on_regulation"]["values"][traits.stance_on_regulation]
    comm_desc = TRAIT_DIMENSIONS["communication_style"]["values"][traits.communication_style]
    conflict_desc = TRAIT_DIMENSIONS["conflict_approach"]["values"][traits.conflict_approach]

    prompt = (
        f"You are {name}, {role_desc}. "
        f"You are {disp_desc}. "
        f"You deeply value {values_desc}. "
        f"On technology regulation, you {reg_desc}. "
        f"Your communication style is {comm_desc}. "
        f"When others disagree with you, you {conflict_desc}. "
        f"Keep your responses concise and focused — 2-3 sentences per response. "
        f"Engage directly with what others have said."
    )
    return prompt


def load_persona_config(agent_def: dict) -> tuple[str, PersonaTraits, str]:
    """Load a persona from an agent definition in experiment config.

    Supports two modes:
    - Trait-based: agent_def has a "traits" dict → compose prompt from traits
    - Legacy: agent_def has a "system_prompt" string → use directly (single-node only)

    Returns (name, traits_or_None, system_prompt).
    """
    name = agent_def["name"]

    if "traits" in agent_def:
        traits = PersonaTraits(**agent_def["traits"])
        errors = traits.validate()
        if errors:
            raise ValueError(f"Agent '{name}' has invalid traits:\n" + "\n".join(errors))
        system_prompt = compose_system_prompt(name, traits)
        return name, traits, system_prompt
    elif "system_prompt" in agent_def:
        return name, None, agent_def["system_prompt"]
    else:
        raise ValueError(f"Agent '{name}' must have either 'traits' or 'system_prompt'")


def list_traits() -> dict:
    """Return the full trait dimension catalog for display/selection."""
    return TRAIT_DIMENSIONS
