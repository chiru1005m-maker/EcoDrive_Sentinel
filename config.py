"""
EcoDrive-Sentinel: Shared Configuration & Pydantic Data Models
==============================================================
Central configuration hub following EU Battery Passport 2026 standards.
All Pydantic v2 models for inter-module data contracts are defined here.

Author: EcoDrive-Sentinel Team
Standard: EU Battery Regulation 2023/1542 / Battery Passport Annex XIII
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
ONNX_DIR = PROJECT_ROOT / "onnx"

# Dataset-specific source directories
NASA_DIR = PROJECT_ROOT / "NASA_PCoE_dataset" / "data"
CALCE_DIR = PROJECT_ROOT / "CALCE_dataset" / "Train"

for _d in [DATA_DIR, MODEL_DIR, ONNX_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Application Settings (env-driven)
# ─────────────────────────────────────────────
class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables or .env file.
    Follows 12-factor app methodology for production deployability.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # MongoDB
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db: str = Field(default="ecodrive_sentinel", alias="MONGO_DB")
    mongo_collection: str = Field(default="repair_protocols", alias="MONGO_COLLECTION")

    # LLM
    openai_api_key: str = Field(default="sk-placeholder", alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # Model
    rul_threshold: int = Field(default=20, alias="RUL_THRESHOLD")
    model_path: Path = Field(default=MODEL_DIR / "cnn_lstm.pt", alias="MODEL_PATH")
    onnx_path: Path = Field(default=ONNX_DIR / "cnn_lstm.onnx", alias="ONNX_PATH")

    # Training
    batch_size: int = Field(default=32, alias="BATCH_SIZE")
    learning_rate: float = Field(default=1e-3, alias="LEARNING_RATE")
    epochs: int = Field(default=50, alias="EPOCHS")
    sequence_length: int = Field(default=30, alias="SEQUENCE_LENGTH")

    # Edge-AI / NPU
    npu_target: str = Field(default="RYZEN_AI_HAWK_POINT", alias="NPU_TARGET")
    max_latency_ms: int = Field(default=50, alias="MAX_LATENCY_MS")


# Singleton settings instance
settings = Settings()


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────
class DatasetSource(str, Enum):
    """Supported battery dataset sources."""
    NASA = "NASA_PCoE"
    CALCE = "CALCE"
    SYNTHETIC = "SYNTHETIC"


class MaintenanceStatus(str, Enum):
    """Battery health status per EU Battery Passport Annex XIII."""
    NORMAL = "NORMAL_OPERATION"
    WARNING = "DEGRADATION_WARNING"
    CRITICAL = "MAINTENANCE_REQUIRED"
    FAULT = "FAULT_DETECTED"


class ChemistryType(str, Enum):
    """Battery chemistry types per EU Battery Regulation 2023/1542."""
    LCO = "LiCoO2"
    LFP = "LiFePO4"
    NMC = "LiNiMnCoO2"
    NCA = "LiNiCoAlO2"


# ─────────────────────────────────────────────
# Data Contract Models (Pydantic v2)
# ─────────────────────────────────────────────
class RawCycleRecord(BaseModel):
    """
    Validated raw record from battery cycling data.
    Handles heterogeneous NASA/CALCE column schemas.
    """

    model_config = {"arbitrary_types_allowed": True}

    cycle_number: int = Field(..., ge=0, description="Monotonic cycle index")
    voltage_measured: float = Field(..., description="Terminal voltage [V]")
    current_measured: float = Field(..., description="Current [A], negative=discharge")
    temperature_measured: float = Field(..., description="Cell temperature [°C]")
    capacity: Optional[float] = Field(None, description="Measured capacity [Ah]")
    source: DatasetSource = Field(..., description="Origin dataset")
    battery_id: str = Field(..., description="Unique battery cell identifier")

    @field_validator("temperature_measured")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (-40.0 <= v <= 85.0):
            raise ValueError(f"Temperature {v}°C outside operational range [-40, 85]°C")
        return v

    @field_validator("voltage_measured")
    @classmethod
    def validate_voltage(cls, v: float) -> float:
        if not (0.0 <= v <= 5.0):
            raise ValueError(f"Voltage {v}V outside plausible range [0, 5]V")
        return v


class HealthIndicators(BaseModel):
    """
    Extracted Health Indicators (HIs) — the feature vector for the model.
    Derived from raw cycling data per state-of-the-art BMS literature.
    """

    battery_id: str
    cycle_number: int = Field(..., ge=0)
    source: DatasetSource

    # Core Health Indicators
    voltage_drop: float = Field(..., description="ΔV = V_nominal - V_eod [V]")
    avg_temperature: float = Field(..., description="Mean cycle temperature [°C]")
    capacity_fade: float = Field(..., description="C_fade = 1 - (C_n / C_0), dimensionless [0,1]")
    internal_resistance_proxy: float = Field(
        default=0.0, description="ΔV/ΔI approximation [Ω]"
    )
    charge_time_delta: float = Field(
        default=0.0, description="Normalized change in charge duration"
    )

    # Target
    rul: Optional[float] = Field(None, ge=0, description="Remaining Useful Life [cycles]")

    @field_validator("capacity_fade")
    @classmethod
    def validate_fade(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Capacity fade {v} must be in [0, 1]")
        return v


class SensorReading(BaseModel):
    """
    Real-time sensor payload arriving at the Agentic inference endpoint.
    Validated before entering the LangGraph state machine.
    """

    battery_id: str = Field(..., min_length=1)
    timestamp: float = Field(..., description="Unix epoch timestamp")
    voltage: float = Field(..., ge=0.0, le=5.0)
    current: float = Field(..., description="Signed current [A]")
    temperature: float = Field(..., ge=-40.0, le=85.0)
    cycle_count: int = Field(..., ge=0)
    chemistry: ChemistryType = Field(default=ChemistryType.NMC)

    @model_validator(mode="after")
    def check_discharge_current(self) -> "SensorReading":
        # Warn if suspiciously high discharge rate (>5C for typical cells)
        if self.current < -200.0:
            raise ValueError(f"Discharge current {self.current}A exceeds safe threshold")
        return self


class InferenceResult(BaseModel):
    """Output contract from the Inference Node."""

    battery_id: str
    predicted_rul: float = Field(..., ge=0, description="Predicted RUL in cycles")
    confidence_interval: tuple[float, float] = Field(
        default=(0.0, 0.0), description="95% CI [lower, upper]"
    )
    maintenance_status: MaintenanceStatus
    inference_latency_ms: float = Field(..., description="End-to-end inference time")
    model_version: str = Field(default="1.0.0")


class DiagnosticReport(BaseModel):
    """Final output of the Agentic pipeline — EU Battery Passport compliant."""

    battery_id: str
    rul_cycles: float
    maintenance_status: MaintenanceStatus
    retrieved_protocols: list[str] = Field(default_factory=list)
    llm_summary: str = Field(default="", description="LLM-generated diagnostic narrative")
    recommended_actions: list[str] = Field(default_factory=list)
    passport_compliant: bool = Field(default=True)
    report_version: str = Field(default="EU-BP-2026-v1")
