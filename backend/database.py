from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

DATABASE_URL = "sqlite:///./traceai.db"


class Base(DeclarativeBase):
    pass


class Requirement(Base):
    __tablename__ = "requirements"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    generation_runs: Mapped[List["GenerationRun"]] = relationship(back_populates="requirement")
    test_cases: Mapped[List["TestCase"]] = relationship(back_populates="requirement")
    evaluation_results: Mapped[List["EvaluationResult"]] = relationship(back_populates="requirement")
    traceability_records: Mapped[List["TraceabilityRecord"]] = relationship(back_populates="requirement")


class GenerationRun(Base):
    __tablename__ = "generation_runs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    requirement_id: Mapped[int] = mapped_column(
        ForeignKey("requirements.id"),
        nullable=False,
        index=True,
    )
    request_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    raw_ai_output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="completed", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    requirement: Mapped["Requirement"] = relationship(back_populates="generation_runs")
    test_cases: Mapped[List["TestCase"]] = relationship(back_populates="generation_run")
    evaluation_result: Mapped[Optional["EvaluationResult"]] = relationship(
        back_populates="generation_run",
        uselist=False,
    )
    traceability_records: Mapped[List["TraceabilityRecord"]] = relationship(back_populates="generation_run")


class TestCase(Base):
    __tablename__ = "test_cases"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    generation_run_id: Mapped[int] = mapped_column(
        ForeignKey("generation_runs.id"),
        nullable=False,
        index=True,
    )
    requirement_id: Mapped[int] = mapped_column(
        ForeignKey("requirements.id"),
        nullable=False,
        index=True,
    )

    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    external_case_id: Mapped[str] = mapped_column(String(32), nullable=False)
    case_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scenario: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    class_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    input_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    values: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expected: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    requirement_id_ref: Mapped[str] = mapped_column(String(32), nullable=False)

    generation_run: Mapped["GenerationRun"] = relationship(back_populates="test_cases")
    requirement: Mapped["Requirement"] = relationship(back_populates="test_cases")
    traceability_records: Mapped[List["TraceabilityRecord"]] = relationship(back_populates="test_case")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    generation_run_id: Mapped[int] = mapped_column(
        ForeignKey("generation_runs.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    requirement_id: Mapped[int] = mapped_column(
        ForeignKey("requirements.id"),
        nullable=False,
        index=True,
    )

    test_design_coverage: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    requirement_traceability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ai_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_breakdown: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    requirement_coverage: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    structural_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duplicate_free: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bva_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ep_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    negative_applicable: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    bva_applicable: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    ep_applicable: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    duplicates_removed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    negative_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bva_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ep_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    generation_run: Mapped["GenerationRun"] = relationship(back_populates="evaluation_result")
    requirement: Mapped["Requirement"] = relationship(back_populates="evaluation_results")


class TraceabilityRecord(Base):
    __tablename__ = "traceability_records"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    generation_run_id: Mapped[int] = mapped_column(
        ForeignKey("generation_runs.id"),
        nullable=False,
        index=True,
    )
    requirement_id: Mapped[int] = mapped_column(
        ForeignKey("requirements.id"),
        nullable=False,
        index=True,
    )
    test_case_id: Mapped[int] = mapped_column(
        ForeignKey("test_cases.id"),
        nullable=False,
        index=True,
    )

    requirement_id_ref: Mapped[str] = mapped_column(String(32), nullable=False)
    test_case_id_ref: Mapped[str] = mapped_column(String(32), nullable=False)
    record_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="Covered", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    generation_run: Mapped["GenerationRun"] = relationship(back_populates="traceability_records")
    requirement: Mapped["Requirement"] = relationship(back_populates="traceability_records")
    test_case: Mapped["TestCase"] = relationship(back_populates="traceability_records")


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
