import os
import random

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pyarrow as pa
import pyarrow.compute

from rtt import types as t


class Database:
    def __init__(self):
        self._tables: list[pa.Table] = []
        self._embedding_chunks: list[np.ndarray] = []
        self._merged: pa.Table | None = None
        self._embeddings: np.ndarray | None = None

    def _invalidate(self):
        self._merged = None
        self._embeddings = None

    def _ensure_merged(self) -> pa.Table | None:
        if self._merged is not None:
            return self._merged
        if not self._tables:
            return None
        paired = list(zip(self._tables, self._embedding_chunks))
        random.shuffle(paired)
        tables, chunks = zip(*paired)
        self._merged = pa.concat_tables(tables)
        self._embeddings = np.concatenate(chunks)
        emb32 = self._embeddings.astype(np.float32)
        norms = np.linalg.norm(emb32, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb32 /= norms
        self._embeddings = emb32.astype(np.float16)
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
        self.add_table(table)

    def add_table(self, table: pa.Table) -> None:
        if len(table) == 0:
            return
        emb_col = table.column("text_embedding")
        flat = emb_col.combine_chunks().values.to_numpy(zero_copy_only=False)
        self._embedding_chunks.append(flat.astype(np.float16).reshape(-1, 768))
        self._tables.append(table.drop("text_embedding"))
        self._invalidate()

    def merge(self, other: "Database") -> None:
        other._ensure_merged()
        if other._merged is not None and other._embeddings is not None:
            self._tables.append(other._merged)
            self._embedding_chunks.append(other._embeddings)
            self._invalidate()

    def closest(self, query_embedding: list[float], n: int = 10, collections: list[str] | None = None) -> list[dict]:
        table = self._ensure_merged()
        if table is None:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        scores = np.empty(len(self._embeddings), dtype=np.float32)
        CHUNK = 20_000
        for i in range(0, len(self._embeddings), CHUNK):
            chunk = self._embeddings[i:i + CHUNK].astype(np.float32)
            scores[i:i + CHUNK] = chunk @ q

        if collections:
            col = table.column("collection")
            mask = None
            for c in collections:
                m = pyarrow.compute.equal(col, c)
                mask = m if mask is None else pyarrow.compute.or_(mask, m)
            filter_np = mask.combine_chunks().to_numpy(zero_copy_only=False)
            scores[~filter_np] = -np.inf

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
            row["_distance"] = 1.0 - float(scores[idx])
            results.append(row)
        return results

    def compact(self):
        self._tables.clear()
        self._embedding_chunks.clear()

    def get_segment(self, segment_id: str) -> dict | None:
        table = self._ensure_merged()
        if table is None:
            return None
        col = table.column("segment_id")
        idx = pyarrow.compute.index(col, segment_id).as_py()
        if idx < 0:
            return None
        row = {name: table.column(name)[idx].as_py() for name in table.schema.names}
        if self._embeddings is not None:
            row["text_embedding"] = self._embeddings[idx].tolist()
        return row

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
