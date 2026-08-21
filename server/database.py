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
    func,
    select,
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
    Column("gender", String, nullable=False),
    Column("gender_confidence", Float),
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
