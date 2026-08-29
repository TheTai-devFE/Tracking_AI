"""Database repository shared by PostgreSQL and the local SQLite fallback."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from audience.contracts import AudienceSession


metadata = MetaData()

audience_sessions = Table(
    "audience_sessions",
    metadata,
    Column("session_id", String, primary_key=True),
    Column("schema_version", Integer, nullable=False),
    Column("standee_id", String, nullable=False),
    Column("campaign_id", String, nullable=False),
    Column("creative_id", String, nullable=False),
    Column("provider", String, nullable=False),
    Column("provider_track_id", Integer, nullable=False),
    Column("started_at", Float, nullable=False),
    Column("ended_at", Float, nullable=False),
    Column("presence_seconds", Float, nullable=False),
    Column("attention_seconds", Float, nullable=False),
    Column("attention_ratio", Float, nullable=False),
    Column("look_away_count", Integer, nullable=False),
    Column("is_impression", Boolean, nullable=False),
    Column("is_viewer", Boolean, nullable=False),
    Column("is_engaged", Boolean, nullable=False),
    Column("age_group", String, nullable=False),
    Column("age_confidence", Float),
    Column("estimated_age", Float, nullable=True),
    Column("gender", String, nullable=False),
    Column("gender_confidence", Float),
    Column("dominant_emotion", String, nullable=True),
    Column("average_distance_m", Float, nullable=True),
    Column("gaze_direction", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
)
Index(
    "idx_sessions_scope_time",
    audience_sessions.c.standee_id,
    audience_sessions.c.campaign_id,
    audience_sessions.c.creative_id,
    audience_sessions.c.started_at,
)

device_telemetry = Table(
    "device_telemetry",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("standee_id", String, nullable=False),
    Column("recorded_at", Float, nullable=False),
    Column("payload", JSON, nullable=False),
)
Index("idx_telemetry_standee_time", device_telemetry.c.standee_id, device_telemetry.c.recorded_at)

creatives = Table(
    "creatives",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("file_name", String, nullable=False),
    Column("file_url", String, nullable=False),
    Column("media_type", String, nullable=False),  # "video" | "image"
    Column("duration_seconds", Float, nullable=False, default=10.0),
    Column("campaign_id", String, nullable=False, default="CMP-LOCAL"),
    Column("is_active", Boolean, nullable=False, default=True),
    Column("sort_order", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
)
Index("idx_creatives_active_order", creatives.c.is_active, creatives.c.sort_order)


def database_url_from_environment(root: Path) -> str:
    """Resolve PostgreSQL URL first, retaining the old local database setting."""
    value = os.getenv("DATABASE_URL")
    if value:
        # Make the Psycopg 3 driver explicit; some hosting providers return postgres://.
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value.removeprefix("postgres://")
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value.removeprefix("postgresql://")
        return value

    legacy_path = Path(os.getenv("TRACKING_DATABASE", root / "data" / "tracking_poc.db"))
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{legacy_path.resolve().as_posix()}"


class TrackingRepository:
    def __init__(self, database: str | Path) -> None:
        if isinstance(database, Path) or "://" not in database:
            path = Path(database)
            path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{path.resolve().as_posix()}"
        else:
            database_url = database

        engine_options: dict[str, Any] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            engine_options["connect_args"] = {"timeout": 30, "check_same_thread": False}
            engine_options["poolclass"] = NullPool
        else:
            engine_options.update(pool_size=5, max_overflow=10, pool_recycle=1800)
        self.engine: Engine = create_engine(database_url, **engine_options)

    @property
    def backend(self) -> str:
        return self.engine.dialect.name

    @property
    def display_url(self) -> str:
        return self.engine.url.render_as_string(hide_password=True)

    def initialize(self) -> None:
        metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            for col, pg_type, sqlite_type in [
                ("estimated_age", "DOUBLE PRECISION", "REAL"),
                ("dominant_emotion", "VARCHAR(32)", "TEXT"),
                ("average_distance_m", "DOUBLE PRECISION", "REAL"),
                ("gaze_direction", "VARCHAR(32)", "TEXT"),
            ]:
                try:
                    if self.backend == "postgresql":
                        connection.execute(text(f"ALTER TABLE audience_sessions ADD COLUMN IF NOT EXISTS {col} {pg_type};"))
                    elif self.backend == "sqlite":
                        connection.execute(text(f"ALTER TABLE audience_sessions ADD COLUMN {col} {sqlite_type};"))
                except Exception:
                    pass

    def healthcheck(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(select(1))

    def save_sessions(self, sessions: list[AudienceSession]) -> int:
        if not sessions:
            return 0
        rows = [session.as_dict() for session in sessions]
        if self.backend == "postgresql":
            statement = postgresql_insert(audience_sessions).values(rows).on_conflict_do_nothing(
                index_elements=[audience_sessions.c.session_id]
            )
        elif self.backend == "sqlite":
            statement = sqlite_insert(audience_sessions).values(rows).on_conflict_do_nothing(
                index_elements=[audience_sessions.c.session_id]
            )
        else:
            raise RuntimeError(f"Unsupported database backend: {self.backend}")
        with self.engine.begin() as connection:
            result = connection.execute(statement)
            return max(result.rowcount or 0, 0)

    def save_telemetry(self, standee_id: str, recorded_at: float, payload: dict[str, Any]) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                device_telemetry.insert().values(
                    standee_id=standee_id,
                    recorded_at=recorded_at,
                    payload=payload,
                )
            )

    @staticmethod
    def _filters(
        standee_id: str | None,
        campaign_id: str | None,
        creative_id: str | None,
        date_from: float | None,
        date_to: float | None,
    ) -> list[Any]:
        filters: list[Any] = []
        for column, value in (
            (audience_sessions.c.standee_id, standee_id),
            (audience_sessions.c.campaign_id, campaign_id),
            (audience_sessions.c.creative_id, creative_id),
        ):
            if value:
                filters.append(column == value)
        if date_from is not None:
            filters.append(audience_sessions.c.started_at >= date_from)
        if date_to is not None:
            filters.append(audience_sessions.c.started_at < date_to)
        return filters

    def overview(
        self,
        standee_id: str | None = None,
        campaign_id: str | None = None,
        creative_id: str | None = None,
        date_from: float | None = None,
        date_to: float | None = None,
    ) -> dict[str, Any]:
        statement = select(
            func.count().label("sessions"),
            func.coalesce(func.sum(audience_sessions.c.is_impression.cast(Integer)), 0).label("impressions"),
            func.coalesce(func.sum(audience_sessions.c.is_viewer.cast(Integer)), 0).label("viewers"),
            func.coalesce(func.sum(audience_sessions.c.is_engaged.cast(Integer)), 0).label("engaged_viewers"),
            func.coalesce(func.sum(audience_sessions.c.presence_seconds), 0).label("total_presence_seconds"),
            func.coalesce(func.sum(audience_sessions.c.attention_seconds), 0).label("total_attention_seconds"),
            func.coalesce(func.avg(audience_sessions.c.presence_seconds), 0).label("average_presence_seconds"),
            func.coalesce(func.avg(audience_sessions.c.attention_seconds), 0).label("average_attention_seconds"),
            func.coalesce(func.sum(audience_sessions.c.look_away_count), 0).label("look_away_count"),
        ).where(*self._filters(standee_id, campaign_id, creative_id, date_from, date_to))
        with self.engine.connect() as connection:
            row = dict(connection.execute(statement).mappings().one())
        presence = float(row["total_presence_seconds"])
        attention = float(row["total_attention_seconds"])
        row["attention_rate"] = round(attention / presence, 4) if presence else 0.0
        for field in (
            "total_presence_seconds",
            "total_attention_seconds",
            "average_presence_seconds",
            "average_attention_seconds",
        ):
            row[field] = round(float(row[field]), 3)
        return row

    def recent_sessions(self, standee_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        statement = select(audience_sessions)
        if standee_id:
            statement = statement.where(audience_sessions.c.standee_id == standee_id)
        statement = statement.order_by(audience_sessions.c.started_at.desc()).limit(max(1, min(limit, 500)))
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def timeline(self, standee_id: str | None = None, hours: int = 24) -> list[dict[str, Any]]:
        filters: list[Any] = [audience_sessions.c.started_at >= time.time() - hours * 3600]
        if standee_id:
            filters.append(audience_sessions.c.standee_id == standee_id)

        if self.backend == "postgresql":
            bucket = func.date_trunc("hour", func.to_timestamp(audience_sessions.c.started_at)).label("bucket")
        else:
            bucket = func.strftime(
                "%Y-%m-%dT%H:00:00Z",
                audience_sessions.c.started_at,
                "unixepoch",
            ).label("bucket")
        statement = select(
            bucket,
            func.sum(audience_sessions.c.is_impression.cast(Integer)).label("impressions"),
            func.sum(audience_sessions.c.is_viewer.cast(Integer)).label("viewers"),
            func.sum(audience_sessions.c.is_engaged.cast(Integer)).label("engaged_viewers"),
            func.sum(audience_sessions.c.attention_seconds).label("attention_seconds"),
        ).where(*filters).group_by(bucket).order_by(bucket)
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if hasattr(item["bucket"], "isoformat"):
                item["bucket"] = item["bucket"].isoformat().replace("+00:00", "Z")
            item["attention_seconds"] = round(float(item["attention_seconds"] or 0), 3)
            result.append(item)
        return result

    def list_creatives(self, only_active: bool = False) -> list[dict[str, Any]]:
        statement = select(creatives)
        if only_active:
            statement = statement.where(creatives.c.is_active.is_(True))
        statement = statement.order_by(creatives.c.sort_order.asc(), creatives.c.created_at.asc())
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def create_creative(self, data: dict[str, Any]) -> dict[str, Any]:
        with self.engine.begin() as connection:
            connection.execute(creatives.insert().values(**data))
        with self.engine.connect() as connection:
            row = connection.execute(select(creatives).where(creatives.c.id == data["id"])).mappings().one()
        return dict(row)

    def delete_creative(self, creative_id: str) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(delete(creatives).where(creatives.c.id == creative_id))
            return bool(result.rowcount and result.rowcount > 0)

    def toggle_creative(self, creative_id: str, is_active: bool) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(creatives).where(creatives.c.id == creative_id).values(is_active=is_active)
            )
            return bool(result.rowcount and result.rowcount > 0)

    def update_creative(self, creative_id: str, data: dict[str, Any]) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(creatives).where(creatives.c.id == creative_id).values(**data)
            )
            return bool(result.rowcount and result.rowcount > 0)

    def reorder_creatives(self, ordered_ids: list[str]) -> bool:
        with self.engine.begin() as connection:
            for index, cid in enumerate(ordered_ids):
                connection.execute(
                    update(creatives).where(creatives.c.id == cid).values(sort_order=index)
                )
        return True

    def creatives_report(
        self, standee_id: str | None = None, campaign_id: str | None = None
    ) -> list[dict[str, Any]]:
        filters = []
        if standee_id:
            filters.append(audience_sessions.c.standee_id == standee_id)
        if campaign_id:
            filters.append(audience_sessions.c.campaign_id == campaign_id)

        session_stats = select(
            audience_sessions.c.creative_id,
            func.count().label("sessions"),
            func.coalesce(func.sum(audience_sessions.c.is_impression.cast(Integer)), 0).label("impressions"),
            func.coalesce(func.sum(audience_sessions.c.is_viewer.cast(Integer)), 0).label("viewers"),
            func.coalesce(func.sum(audience_sessions.c.is_engaged.cast(Integer)), 0).label("engaged_viewers"),
            func.coalesce(func.sum(audience_sessions.c.presence_seconds), 0).label("total_presence_seconds"),
            func.coalesce(func.sum(audience_sessions.c.attention_seconds), 0).label("total_attention_seconds"),
            func.coalesce(func.avg(audience_sessions.c.presence_seconds), 0).label("average_presence_seconds"),
            func.coalesce(func.avg(audience_sessions.c.attention_seconds), 0).label("average_attention_seconds"),
            func.coalesce(func.sum(audience_sessions.c.look_away_count), 0).label("look_away_count"),
        ).where(*filters).group_by(audience_sessions.c.creative_id)

        with self.engine.connect() as connection:
            session_rows = {row["creative_id"]: dict(row) for row in connection.execute(session_stats).mappings().all()}
            registered_creatives = {row["id"]: dict(row) for row in connection.execute(select(creatives)).mappings().all()}

        all_creative_ids = set(session_rows.keys()) | set(registered_creatives.keys())
        results = []
        for cid in sorted(all_creative_ids):
            stats = session_rows.get(cid, {
                "creative_id": cid,
                "sessions": 0,
                "impressions": 0,
                "viewers": 0,
                "engaged_viewers": 0,
                "total_presence_seconds": 0.0,
                "total_attention_seconds": 0.0,
                "average_presence_seconds": 0.0,
                "average_attention_seconds": 0.0,
                "look_away_count": 0,
            })
            reg = registered_creatives.get(cid, {})
            presence = float(stats["total_presence_seconds"])
            attention = float(stats["total_attention_seconds"])
            rate = round(attention / presence, 4) if presence else 0.0

            results.append({
                "creative_id": cid,
                "name": reg.get("name", cid),
                "media_type": reg.get("media_type", "unknown"),
                "file_url": reg.get("file_url", ""),
                "duration_seconds": reg.get("duration_seconds", 10.0),
                "is_active": reg.get("is_active", True),
                "sessions": stats["sessions"],
                "impressions": stats["impressions"],
                "viewers": stats["viewers"],
                "engaged_viewers": stats["engaged_viewers"],
                "total_presence_seconds": round(presence, 2),
                "total_attention_seconds": round(attention, 2),
                "average_presence_seconds": round(float(stats["average_presence_seconds"]), 2),
                "average_attention_seconds": round(float(stats["average_attention_seconds"]), 2),
                "attention_rate": rate,
                "look_away_count": stats["look_away_count"],
            })
        return results
