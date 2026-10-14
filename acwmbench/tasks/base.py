"""Task specifications.

A `TaskSpec` carries everything a metric needs to know about *what counts as
success* for a particular task. For the MVP we only consume the fields used
by the physical-status-judge VLM prompt, but the schema is intentionally
forward-compatible with future metrics (key-area masks, multi-view, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskSpec:
    task_id: str
    instruction: str = ""
    task_type: str = "generic"
    judging_schema: Dict[str, Any] = field(default_factory=dict)
    key_steps: List[Dict[str, Any]] = field(default_factory=list)
    keyarea_masks: List[str] = field(default_factory=list)
    primary_views: List[str] = field(default_factory=lambda: ["head"])
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_instruction: str = "") -> "TaskSpec":
        return cls(
            task_id=str(data.get("task_id", "unknown")),
            instruction=str(data.get("instruction", default_instruction or "")),
            task_type=str(data.get("type", data.get("task_type", "generic"))),
            judging_schema=dict(data.get("judging_schema", {})),
            key_steps=list(data.get("key_steps", [])),
            keyarea_masks=list(data.get("keyarea_masks", [])),
            primary_views=list(data.get("primary_views", ["head"])),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "task_id",
                    "instruction",
                    "type",
                    "task_type",
                    "judging_schema",
                    "key_steps",
                    "keyarea_masks",
                    "primary_views",
                }
            },
        )

    @classmethod
    def fallback(cls, task_id: str, instruction: str) -> "TaskSpec":
        """When no task_spec.json is provided we fall back to free-form judging."""
        return cls(task_id=str(task_id), instruction=instruction or "")

    def has_structured_schema(self) -> bool:
        return bool(self.judging_schema)


def load_task_spec(
    task_id: str,
    instruction: str,
    raw_spec: Optional[Dict[str, Any]] = None,
) -> TaskSpec:
    if raw_spec:
        spec = TaskSpec.from_dict(raw_spec, default_instruction=instruction)
        if not spec.task_id or spec.task_id == "unknown":
            spec.task_id = str(task_id)
        if not spec.instruction:
            spec.instruction = instruction
        return spec
    return TaskSpec.fallback(task_id, instruction)
