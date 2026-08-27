"""Pydantic response contracts for the internal converter API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CharacterOut(BaseModel):
    character_id: str
    display_name: str
    rig_profile: str
    revision: str


class CharactersResponse(BaseModel):
    characters: list[CharacterOut]


class HealthResponse(BaseModel):
    ok: bool
    solver_version: str
    checks: dict[str, Any]


class ErrorDetail(BaseModel):
    code: str
    message: str
    conversion_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ConvertOptions(BaseModel):
    character_id: str = "standin-master-v2"
    frame: int = 0
    mirror: bool = False
    output_mode: str = "rigged_rest"
    apply_root_translation: bool = False
