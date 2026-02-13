import pathlib

import lancedb
import pyarrow as pa
import pyarrow.compute

from rtt import types as t

SCHEMA = pa.schema([
    pa.field("segment_id", pa.string()),
    pa.field("video_id", pa.string()),
    pa.field("start_seconds", pa.float64()),
    pa.field("end_seconds", pa.float64()),
    pa.field("transcript_raw", pa.string()),
    pa.field("transcript_enriched", pa.string()),
    pa.field("text_embedding", pa.list_(pa.float32(), 768)),
    pa.field("frame_path", pa.string()),
    pa.field("has_speech", pa.bool_()),
    pa.field("source", pa.string()),
    pa.field("collection", pa.string()),
])


class Database:
    def __init__(self, db: lancedb.DBConnection, table: lancedb.table.Table):
        self._db = db
        self._table = table

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Database":
        db = lancedb.connect(str(path))
        if "segments" in db.table_names():
            table = db.open_table("segments")
        else:
            table = db.create_table("segments", schema=SCHEMA)
        return cls(db, table)

    @classmethod
    def memory(cls) -> "Database":
        db = lancedb.connect("memory://")
        table = db.create_table("segments", schema=SCHEMA)
        return cls(db, table)

    def add(self, segments: list[t.Segment]) -> None:
        if not segments:
            return
        rows = [
            {
                "segment_id": s.segment_id,
                "video_id": s.video_id,
                "start_seconds": s.start_seconds,
                "end_seconds": s.end_seconds,
                "transcript_raw": s.transcript_raw,
                "transcript_enriched": s.transcript_enriched,
                "text_embedding": s.text_embedding,
                "frame_path": s.frame_path,
                "has_speech": s.has_speech,
                "source": s.source,
                "collection": s.collection,
            }
            for s in segments
        ]
        self._table.add(rows)

    def merge(self, other: "Database") -> None:
        data = other._table.to_arrow()
        if len(data) > 0:
            self._table.add(data)

    def closest(self, query_embedding: list[float], n: int = 10, collections: list[str] | None = None) -> list[dict]:
        q = self._table.search(query_embedding).limit(n)
        if collections:
            filter_expr = " OR ".join(f"collection = '{c}'" for c in collections)
            q = q.where(f"({filter_expr})")
        return q.to_list()

    def get_segment(self, segment_id: str) -> dict | None:
        rows = self._table.search().where(f"segment_id = '{segment_id}'").limit(1).to_list()
        return rows[0] if rows else None

    def list_segments(self, offset: int = 0, limit: int = 50, collections: list[str] | None = None) -> list[dict]:
        table = self._table.to_arrow()
        if collections:
            col = table.column("collection")
            mask = None
            for c in collections:
                m = pa.compute.equal(col, c)
                mask = m if mask is None else pa.compute.or_(mask, m)
            table = table.filter(mask)
        table = table.slice(offset, limit)
        names = table.schema.names
        return [
            {name: table.column(name)[i].as_py() for name in names}
            for i in range(table.num_rows)
        ]

    def count(self, collections: list[str] | None = None) -> int:
        table = self._table.to_arrow()
        if collections:
            col = table.column("collection")
            mask = None
            for c in collections:
                m = pa.compute.equal(col, c)
                mask = m if mask is None else pa.compute.or_(mask, m)
            table = table.filter(mask)
        return table.num_rows
