"""Execution and file system models."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class FileEntry(BaseModel):
    """File entry model for E2B sandbox files."""
    name: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)
    content: Optional[str] = None
    size: Optional[int] = None

    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('name cannot be empty')
        return v

    @field_validator('path')
    @classmethod
    def path_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('path cannot be empty')
        return v