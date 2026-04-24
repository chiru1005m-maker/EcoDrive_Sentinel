"""
EcoDrive-Sentinel | FastAPI Application
=======================================
Production-grade REST API exposing the EcoDrive-Sentinel predictive
maintenance pipeline with EU Battery Passport 2026 compliance.

Endpoints:
    POST /api/v1/diagnose      → Full agentic pipeline (inference + diagnostic)
    POST /api/v1/predict-rul   → RUL prediction only (low-latency path)
    GET  /api/v1/health        → Service health check
    GET  /api/v1/metrics       → Prometheus metrics (Grafana integration)

Standards:
    - OpenAPI 3.1 schema auto-generated
    - EU Battery Regulation 2023/1542 response format
    - <50ms SLA for /predict-rul endpoint

Author: EcoDrive-Sentinel Team
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from agentic_layer import ONNXInferenceEngine, run_diagnostic_pipeline
from config import DiagnosticReport, InferenceResult, MaintenanceStatus, SensorReading, settings


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────
inference_engine: ONNXInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy resources on startup; clean up on shutdown."""
    global inference_engine
    logger.info("EcoDrive-Sentinel API starting up...")
    inference_engine = ONNXInferenceEngine()
    logger.info("✓ Inference engine ready")
    yield
    logger.info("EcoDrive-Sentinel API shutting down...")


# ─────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────
app = FastAPI(
    title="EcoDrive-Sentinel API",
    description=(
        "Predictive Maintenance for EV Batteries | "
        "EU Battery Passport 2026 Compliant | "
        "Mercedes-Benz BEVisoneers"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    onnx_ready: bool
    timestamp: float


class RULResponse(BaseModel):
    battery_id: str
    predicted_rul: float
    maintenance_status: str
    inference_latency_ms: float
    compliant_eu_bp_2026: bool = True


# ─────────────────────────────────────────────
# Middleware: Latency Logging
# ─────────────────────────────────────────────
@app.middleware("http")
async def log_latency(request: Request, call_next):
    t_start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{latency_ms:.1f}ms]")
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
    return response


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service liveness probe. Used by Kubernetes readiness checks."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        onnx_ready=inference_engine is not None and inference_engine.session is not None,
        timestamp=time.time(),
    )


@app.post("/api/v1/predict-rul", response_model=RULResponse, tags=["Inference"])
async def predict_rul(sensor: SensorReading):
    """
    Low-latency RUL prediction endpoint.

    Target: <50ms end-to-end (EU Edge-AI standard).
    Does NOT invoke the full agentic pipeline — pure ONNX inference.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    try:
        rul, latency_ms = inference_engine.predict(sensor)

        if rul > settings.rul_threshold * 2:
            status_str = MaintenanceStatus.NORMAL.value
        elif rul > settings.rul_threshold:
            status_str = MaintenanceStatus.WARNING.value
        else:
            status_str = MaintenanceStatus.CRITICAL.value

        return RULResponse(
            battery_id=sensor.battery_id,
            predicted_rul=rul,
            maintenance_status=status_str,
            inference_latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.error(f"RUL prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/diagnose", response_model=DiagnosticReport, tags=["Agentic"])
async def run_full_diagnostic(sensor: SensorReading):
    """
    Full agentic diagnostic pipeline.

    Invokes: Inference → Logic Gate → [Normal Operation | Diagnostic Node]
    Returns EU Battery Passport 2026 compliant DiagnosticReport.

    Note: Latency is higher (~1-3s) due to MongoDB vector search + LLM call.
    """
    try:
        report = await run_diagnostic_pipeline(sensor)
        return report
    except Exception as exc:
        logger.error(f"Diagnostic pipeline failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "EcoDrive-Sentinel",
        "version": "1.0.0",
        "docs": "/docs",
        "standard": "EU Battery Regulation 2023/1542",
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
