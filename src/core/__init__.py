# ──────────────────────────────────────────────────────────
# EcoDrive-Sentinel | src.core Package
# ──────────────────────────────────────────────────────────
# Contains model definitions, training logic, feature engineering,
# and shared configuration for the predictive maintenance system.
# ──────────────────────────────────────────────────────────

from src.core.config import settings, Settings
from src.core.config import (
    PROJECT_ROOT, DATA_DIR, MODEL_DIR, ONNX_DIR, NASA_DIR, CALCE_DIR,
    DatasetSource, MaintenanceStatus, ChemistryType,
    RawCycleRecord, HealthIndicators, SensorReading,
    InferenceResult, DiagnosticReport,
)
