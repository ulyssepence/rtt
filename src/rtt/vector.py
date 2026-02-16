import random

import numpy as np
import pyarrow as pa
import pyarrow.compute

from rtt import types as t


class Database:
    def __init__(self):
        self._tables: list[pa.Table] = []
        self._merged: pa.Table | None = None
        self._embeddings: np.ndarray | None = None
        self._norms: np.ndarray | None = None

    def _invalidate(self):
        self._merged = None
        self._embeddings = None
        self._norms = None

    def _ensure_merged(self) -> pa.Table | None:
        if self._merged is not None:
            return self._merged
        if not self._tables:
            return None
        tables = list(self._tables)
        random.shuffle(tables)
        self._merged = pa.concat_tables(tables)
        emb_col = self._merged.column("text_embedding")
        self._embeddings = np.array(emb_col.to_pylist(), dtype=np.float32)
        self._norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._norms = np.where(self._norms == 0, 1, self._norms)
        return self._merged

    @classmethod
    def memory(cls) -> "Database":
        return cls()

    def add(self, segments: list[t.Segment]) -> None:
        if not segments:
            return
        table = pa.table({
            "segment_id": [s.segment_id for s in segments],
            "video_id": [s.video_id for s in segments],
            "start_seconds": [s.start_seconds for s in segments],
            "end_seconds": [s.end_seconds for s in segments],
            "transcript_raw": [s.transcript_raw for s in segments],
            "transcript_enriched": [s.transcript_enriched for s in segments],
            "text_embedding": [s.text_embedding for s in segments],
            "frame_path": [s.frame_path for s in segments],
            "has_speech": [s.has_speech for s in segments],
            "source": [s.source for s in segments],
            "collection": [s.collection for s in segments],
        })
        self._tables.append(table)
        self._invalidate()

    def add_table(self, table: pa.Table) -> None:
        if len(table) == 0:
            return
        self._tables.append(table)
        self._invalidate()

    def merge(self, other: "Database") -> None:
        other_table = other._ensure_merged()
        if other_table is not None:
            self.add_table(other_table)

    def closest(self, query_embedding: list[float], n: int = 10, collections: list[str] | None = None) -> list[dict]:
        table = self._ensure_merged()
        if table is None:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        scores = (self._embeddings / self._norms) @ q

        if collections:
            col = table.column("collection")
            mask = None
            for c in collections:
                m = pyarrow.compute.equal(col, c)
                mask = m if mask is None else pyarrow.compute.or_(mask, m)
            filter_mask = mask.to_pylist()
            scores = np.where(filter_mask, scores, -np.inf)

        n = min(n, len(scores))
        if n >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, n)[:n]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        names = [f.name for f in table.schema if f.name != "text_embedding"]
        results = []
        for idx in top_idx:
            if scores[idx] == -np.inf:
                break
            row = {name: table.column(name)[int(idx)].as_py() for name in names}
            row["text_embedding"] = self._embeddings[int(idx)].tolist()
            row["_distance"] = 1.0 - float(scores[idx])
            results.append(row)
        return results

    def get_segment(self, segment_id: str) -> dict | None:
        table = self._ensure_merged()
        if table is None:
            return None
        col = table.column("segment_id")
        mask = pyarrow.compute.equal(col, segment_id)
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            return None
        names = filtered.schema.names
        return {name: filtered.column(name)[0].as_py() for name in names}

    def list_segments(self, offset: int = 0, limit: int = 50, collections: list[str] | None = None) -> list[dict]:
        table = self._ensure_merged()
        if table is None:
            return []
        if collections:
            col = table.column("collection")
            mask = None
            for c in collections:
                m = pyarrow.compute.equal(col, c)
                mask = m if mask is None else pyarrow.compute.or_(mask, m)
            table = table.filter(mask)
        table = table.slice(offset, limit)
        names = table.schema.names
        return [
            {name: table.column(name)[i].as_py() for name in names}
            for i in range(table.num_rows)
        ]

    def video_segments(self, video_id: str) -> list[dict]:
        table = self._ensure_merged()
        if table is None:
            return []
        mask = pyarrow.compute.equal(table.column("video_id"), video_id)
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            return []
        indices = pyarrow.compute.sort_indices(filtered.column("start_seconds"))
        filtered = filtered.take(indices)
        names = [f.name for f in filtered.schema if f.name != "text_embedding"]
        return [
            {name: filtered.column(name)[i].as_py() for name in names}
            for i in range(filtered.num_rows)
        ]

    def count(self, collections: list[str] | None = None) -> int:
        table = self._ensure_merged()
        if table is None:
            return 0
        if collections:
            col = table.column("collection")
            mask = None
            for c in collections:
                m = pyarrow.compute.equal(col, c)
                mask = m if mask is None else pyarrow.compute.or_(mask, m)
            table = table.filter(mask)
        return table.num_rows
