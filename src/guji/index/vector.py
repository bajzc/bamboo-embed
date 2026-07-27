"""LanceDB vector store (§4, dense path).

Holds one row per passage: the embedding + the metadata needed for dense
retrieval and Phase-3 metadata filtering (book/dynasty/category). Display text
lives in the FTS sidecar (:mod:`guji.index.fts`) to keep this table lean.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

TABLE = "passages"


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("book_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("dynasty", pa.string()),
            pa.field("category", pa.string()),
            pa.field("juan_idx", pa.int32()),
        ]
    )


def connect(lancedb_path: Path):
    import lancedb

    lancedb_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(lancedb_path))


def open_table(db, dim: int):
    if TABLE in db.table_names():
        return db.open_table(TABLE)
    return db.create_table(TABLE, schema=_schema(dim))


def existing_ids(table) -> set[str]:
    if table.count_rows() == 0:
        return set()
    return set(table.to_lance().to_table(columns=["chunk_id"]).column("chunk_id").to_pylist())


def add_rows(table, rows: list[dict]) -> None:
    if rows:
        table.add(rows)


def create_ann_index(table) -> str:
    """Build an ANN index when there are enough rows; else flat search is used."""
    n = table.count_rows()
    if n < 256:
        return f"flat (only {n} rows; ANN index not built)"
    try:
        table.create_index(metric="cosine", vector_column_name="vector")
        return f"ANN index built over {n} rows"
    except Exception as e:  # pragma: no cover - depends on lancedb/version
        return f"flat (ANN index skipped: {e})"


def search(table, qvec, k: int, where: str | None = None) -> list[dict]:
    q = table.search(qvec, vector_column_name="vector").metric("cosine").limit(k)
    if where:
        q = q.where(where, prefilter=True)
    out = []
    for r in q.to_list():
        out.append({"chunk_id": r["chunk_id"], "_distance": r.get("_distance")})
    return out
