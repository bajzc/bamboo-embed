"""Pydantic data models shared across parsers (spec §5)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Form = Literal["narrative", "dictionary", "poetry"]


class Book(BaseModel):
    book_id: str
    title: str
    dynasty: str = ""
    author: str = ""
    category: str
    category_dir: str
    source_files: list[str]
    char_count: int
    form: Form
    is_empty: bool = False
    is_complete: bool = True
    is_stub: bool = False          # e.g. 詩經_symbolic: a pointer to another file
    related_to: list[str] = Field(default_factory=list)
    notes: str = ""


class Manifest(BaseModel):
    books: list[Book]


class Passage(BaseModel):
    chunk_id: str
    book_id: str
    title: str
    dynasty: str = ""
    author: str = ""
    category: str
    juan: str
    juan_idx: int
    para_start: int
    para_end: int
    text_raw: str
    text_norm: str
    text_for_embed: str
    char_count: int
    prev_id: Optional[str] = None
    next_id: Optional[str] = None


class CharEntry(BaseModel):
    headword: str
    headword_norm: str
    source_book: str
    body_raw: str
    body_norm: str
    section: str = ""
