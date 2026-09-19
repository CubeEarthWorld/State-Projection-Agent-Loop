"""Skills: progressive-disclosure instructions, expressed as capabilities.

A skill is a body of instructions the model should only read when it is
relevant. Making it a capability ``skill.<name>.load`` reuses every
discovery mechanism that already exists — it shows up in the TOC under
``skill``, in auto-selected candidates when the request matches its
summary, and in ``meta.tool.find`` — with no second index to maintain.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..capability import Capability


def skill_capability(name: str, text: str, *, summary: str) -> Capability:
    """Build the capability that loads one skill's instructions.

    ``name`` is a lowercase identifier (``[a-z][a-z0-9_]*``); ``summary`` is
    the one line the model sees before deciding to load the skill.
    """
    return Capability.from_dict({
        "name": f"skill.{name}.load",
        "category": "skill",
        "card": {"summary": summary, "tags": ["skill", name]},
        "spec": {
            "description": f'Load the instructions for the "{name}" skill: {summary}',
            "parameters": {"type": "object", "properties": {}},
        },
        "discovery": {"embedding_text": f"{name} {summary}"},
        "execution": {"timeout_s": 5, "retry_safety": "pure"},
        "effects": [{"kind": "none"}],
    }, handler=lambda: text)


_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def load_skills(directory: str | Path) -> list[Capability]:
    """Every ``<directory>/<skill>/SKILL.md`` as a skill capability.

    The file format is the one agent skill directories converge on: YAML
    front matter with ``name`` and ``description``, then the instructions.
    A skill's text is data the model reads on request, never part of the
    kernel; load only directories you trust.
    """
    skills: list[Capability] = []
    for path in sorted(Path(directory).glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        match = _FRONTMATTER.match(text)
        fields = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line) if match else {}
        name = fields.get("name", path.parent.name).strip().strip("\"'").lower().replace("-", "_")
        summary = fields.get("description", "").strip().strip("\"'") or f"The {name} skill."
        skills.append(skill_capability(name, text[match.end():] if match else text, summary=summary))
    return skills

