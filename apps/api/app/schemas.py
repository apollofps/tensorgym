"""Pydantic models for content and API payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Content(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate forward-compatible fields


class Check(_Content):
    name: str
    kind: str
    visibility: Literal["visible", "hidden"] = "visible"
    description: str | None = None
    input: Any | None = None
    params: dict[str, Any] | None = None


class Hint(_Content):
    level: int
    text: str


class Verification(_Content):
    status: str = "draft"
    verified_at: str | None = None


class Challenge(_Content):
    id: str
    title: str
    type: str = "implementation"
    difficulty: str = "easy"
    prompt: str
    starter_code: str = ""
    function_name: str
    constraints: list[str] = Field(default_factory=list)
    visible_tests: list[Check] = Field(default_factory=list)
    hidden_test_file: str | None = None
    hints: list[Hint] = Field(default_factory=list)
    solution_file: str | None = None
    estimated_minutes: int | None = None
    concept_capsules: list[str] = Field(default_factory=list)
    required_prerequisites: list[str] = Field(default_factory=list)
    optional_prerequisites: list[str] = Field(default_factory=list)
    concepts_taught: list[str] = Field(default_factory=list)
    concepts_tested: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    verification: Verification = Field(default_factory=Verification)


class Capsule(_Content):
    id: str
    title: str
    estimated_minutes: int | None = None
    supported_pytorch_version: str | None = None
    summary: str
    learning_objectives: list[str] = Field(default_factory=list)
    interactive_example: str | None = None
    warmup_challenge_id: str | None = None
    common_mistakes: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    verification: Verification = Field(default_factory=Verification)


class Warmup(_Content):
    id: str
    title: str
    prompt: str
    starter_code: str = ""
    function_name: str
    constraints: list[str] = Field(default_factory=list)
    visible_tests: list[Check] = Field(default_factory=list)
    solution_inline: str | None = None
    verification: Verification = Field(default_factory=Verification)


class Source(_Content):
    id: str
    title: str
    organization: str | None = None
    url: str
    source_type: str | None = None
    documentation_version: str | None = None
    accessed_at: str | None = None


# ----- API request bodies -----
class CodeBody(BaseModel):
    code: str


class RunBody(BaseModel):
    code: str
    hints_used: int = 0


class WorkspaceBody(BaseModel):
    code: str
    hint_level: int = -1


class ScriptBody(BaseModel):
    code: str | None = None  # optional: fall back to the capsule's stored example
