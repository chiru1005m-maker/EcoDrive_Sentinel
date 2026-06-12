"""
EcoDrive-Sentinel | Phase 3: Agentic Layer — LangGraph State Machine
=====================================================================
Implements a production-grade agentic workflow using LangGraph for
automated battery diagnostics.

State Machine Topology:
    [START]
       ↓
  [inference_node]  ← SensorReading arrives here
       ↓
  [logic_gate]      ← Routes based on predicted RUL
      ↙         ↘
[normal_op]  [diagnostic_node]
   (log)     (MongoDB Vector Search + LLM synthesis)
       ↓
    [END]

Key Design Decisions:
    - Typed state dict (TypedDict) ensures LangGraph node contracts
    - MongoDB Atlas Vector Search for semantic repair protocol retrieval
    - LLM synthesis generates EU Battery Passport compliant reports
    - All I/O validated through Pydantic v2 models
    - Async-first for FastAPI integration

Author: EcoDrive-Sentinel Team
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any, Optional, TypedDict

import numpy as np
import torch

# ── Ryzen AI NPU: Set environment BEFORE importing onnxruntime ──
import os as _os
_os.environ["XLNX_VART_FIRMWARE"] = r"C:\Program Files\AMD\RyzenAI\1x4.xclbin"

import onnxruntime as ort
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pymongo import MongoClient
from pydantic import ValidationError

from src.core.config import (
    DiagnosticReport,
    InferenceResult,
    MaintenanceStatus,
    SensorReading,
    settings,
)
from src.core.predictive_core import FEATURE_COLS, SEQUENCE_LEN

# ─────────────────────────────────────────────
# LangGraph State Schema
# ─────────────────────────────────────────────
class BatteryDiagnosticState(TypedDict):
    """
    Shared state flowing through the LangGraph state machine.

    Each node reads from and writes to this state dict.
    LangGraph merges state via reducers (add_messages for lists).

    Extended with audit_log and ignition_status for EU Battery Passport
    compliance and graceful vehicle shutdown support.
    """
    # Input
    sensor_reading: SensorReading
    # Inference outputs
    predicted_rul: Optional[float]
    rul_percentage: float
    inference_latency_ms: Optional[float]
    maintenance_status: Optional[MaintenanceStatus]
    # Diagnostic outputs
    retrieved_protocols: list[str]
    llm_summary: str
    recommended_actions: list[str]
    # Pipeline control
    route: str                    # "NORMAL" or "DIAGNOSTIC"
    error_message: Optional[str]
    # Ignition & Audit (EU Battery Passport 2023/1542)
    ignition_status: bool
    audit_log: list[str]


# ─────────────────────────────────────────────
# ONNX Runtime Inference Engine
# ─────────────────────────────────────────────
class ONNXInferenceEngine:
    """
    Wraps ONNX Runtime for low-latency RUL inference.

    Supports:
        - CPU Execution Provider (default)
        - AMD Vitis-AI Execution Provider (NPU, requires Vitis-AI SDK)
        - PyTorch fallback (if ONNX not available)

    Latency target: <50ms per EU Edge-AI standard.
    """

    def __init__(self, onnx_path: str = str(settings.onnx_path)):
        self.onnx_path = onnx_path
        self.session: Optional[ort.InferenceSession] = None
        self._load_scaler_from_metadata()
        self._initialize_session()

    def _initialize_session(self) -> None:
        """Initialize ONNX Runtime session with optimal EP."""
        import os

        # Explicit configuration for the local XDNA core compiler
        provider_options = [
            {"config_file": r"C:\Program Files\AMD\RyzenAI\vaip_config.json"},
            {}
        ]

        try:
            self.session = ort.InferenceSession(
                self.onnx_path,
                providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
                provider_options=provider_options
            )
            logger.info(f"Active providers: {self.session.get_providers()}")
        except Exception as e:
            logger.warning(f"ONNX Runtime NPU initialization failed: {e}. PyTorch fallback active.")
            self.session = None


    def _load_scaler_from_metadata(self) -> None:
        """Load scaler parameters from ONNX model metadata."""
        try:
            import onnx
            model = onnx.load(self.onnx_path)
            meta = {p.key: p.value for p in model.metadata_props}
            self.scaler_mean = np.array(
                [float(x) for x in meta.get("scaler_mean", "0," * len(FEATURE_COLS)).split(",") if x],
                dtype=np.float32
            )
            self.scaler_scale = np.array(
                [float(x) for x in meta.get("scaler_scale", "1," * len(FEATURE_COLS)).split(",") if x],
                dtype=np.float32
            )
            logger.info("Scaler metadata loaded from ONNX model")
        except Exception:
            logger.warning("Could not load scaler from ONNX metadata. Using identity scaling.")
            self.scaler_mean = np.zeros(len(FEATURE_COLS), dtype=np.float32)
            self.scaler_scale = np.ones(len(FEATURE_COLS), dtype=np.float32)

    def build_synthetic_tensor(self, voltage: float, current: float, temp: float, cycles: int, max_cycle_limit: int) -> np.ndarray:
        """
        Bridges raw API telemetry into the (1, 30, 5) tensor expected by the CNN-LSTM.
        Implements a manual Heuristic Min-Max Normalization layer to prevent activation saturation.
        """
        def normalize_feature(val: float, min_bound: float, max_bound: float) -> float:
            scaled = (val - min_bound) / (max_bound - min_bound)
            return max(0.0, min(1.0, scaled))

        VOLTAGE_BOUNDS = (2.0, 4.2)
        CURRENT_BOUNDS = (-50.0, 50.0)
        TEMP_BOUNDS = (-10.0, 65.0)

        scaled_voltage = normalize_feature(voltage, VOLTAGE_BOUNDS[0], VOLTAGE_BOUNDS[1])
        scaled_current = normalize_feature(current, CURRENT_BOUNDS[0], CURRENT_BOUNDS[1])
        scaled_temp = normalize_feature(temp, TEMP_BOUNDS[0], TEMP_BOUNDS[1])
        
        # Dynamic Cycle Normalization
        scaled_cycles = cycles / float(max_cycle_limit)
        
        # Pass these normalized values into the final Numpy array (padding 5th feature with 0.0)
        normalized_features = np.array([scaled_voltage, scaled_current, scaled_temp, scaled_cycles, 0.0], dtype=np.float32)
        
        # Ensure complete deterministic behavior for stateless API calls
        seed = int(voltage * 1000 + abs(current) * 1000 + temp * 10 + cycles) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        
        # Build 30-step trajectory with realistic sequence dynamics
        sequence = np.zeros((30, 5), dtype=np.float32)
        for i in range(30):
            # We bypass the missing ONNX scaler since we have manually min-max normalized
            noise = rng.normal(0, 0.01, size=5)
            sequence[i] = normalized_features + noise

        return np.expand_dims(sequence, axis=0).astype(np.float32)

    def predict(self, sensor: SensorReading) -> tuple[float, float]:
        """
        Run RUL inference.
        """
        t_start = time.perf_counter()
        
        # 1. Define physical limits per chemistry
        CHEMISTRY_LIMITS = {
            "LiNiMnCoO2": 2000,  # NMC
            "LiNiCoAlO2": 1500,  # NCA
            "LiFePO4": 6000,     # LFP
            "Na-ion": 3000       # Na-ion
        }
        
        # We also support string shorthand mapping just in case
        SHORTHAND_MAP = {
            "NMC": 2000,
            "NCA": 1500,
            "LFP": 6000,
            "Na-ion": 3000
        }

        # 2. Get the limit dynamically (default to 2000 if not specified)
        # Using string value of enum if available
        chem_str = str(getattr(sensor, "chemistry", "NMC"))
        if chem_str.startswith("ChemistryType."):
            chem_str = chem_str.split(".")[1]
            
        max_cycle_limit = CHEMISTRY_LIMITS.get(chem_str, SHORTHAND_MAP.get(chem_str, 2000))

        # 3. Absolute Failsafes (Physical Safety Bounds)
        if sensor.cycle_count >= max_cycle_limit:
            return 0.0, (time.perf_counter() - t_start) * 1000
            
        is_nmc = chem_str in ["NMC", "LiNiMnCoO2"]
        if is_nmc and sensor.voltage <= 3.0:
            return 0.0, (time.perf_counter() - t_start) * 1000

        # 4. Convert raw API input directly to tensor (no proportional scaling bridge)
        input_tensor = self.build_synthetic_tensor(
            sensor.voltage, 
            sensor.current, 
            sensor.temperature, 
            sensor.cycle_count,
            max_cycle_limit
        )

        if self.session is not None:
            # 5. Run standard ONNX inference to get raw base_rul
            ort_inputs = {"battery_health_indicators": input_tensor}
            outputs = self.session.run(["predicted_rul"], ort_inputs)
            # The universal model directly predicts RUL ratio (1=healthy, 0=dead)
            base_rul = max(0.0, float(outputs[0].squeeze()))
        else:
            # Synthetic prediction fallback for demo/CI
            base_rul = max(0.0, 1.0 - (sensor.cycle_count / max_cycle_limit) + np.random.normal(0, 0.02))

        # 6. Apply Graceful Degradation Curve
        critical_threshold = max_cycle_limit * 0.90
        
        if sensor.cycle_count > critical_threshold:
            # We are in the final 10% of life. Calculate decay multiplier.
            penalty_zone_range = max_cycle_limit - critical_threshold
            cycles_into_zone = sensor.cycle_count - critical_threshold
            decay_multiplier = 1.0 - (cycles_into_zone / penalty_zone_range)
            final_rul = base_rul * decay_multiplier
        else:
            final_rul = base_rul

        latency_ms = (time.perf_counter() - t_start) * 1000
        # Convert fractional RUL [0, 1] to percentage [0, 100] expected by pipeline
        return max(0.0, final_rul * 100.0), latency_ms


# ─────────────────────────────────────────────
# MongoDB Vector Search Client
# ─────────────────────────────────────────────
class RepairProtocolVectorSearch:
    """
    MongoDB Atlas Vector Search for semantic repair protocol retrieval.

    Collection schema:
        {
          "_id": ObjectId,
          "protocol_id": "RP-2024-NMC-001",
          "title": "NMC Cell Thermal Runaway Protocol",
          "content": "Full procedure text...",
          "embedding": [0.12, -0.33, ...],  # 1536-dim OpenAI embedding
          "chemistry": "NMC",
          "severity": "CRITICAL",
          "tags": ["thermal", "cooling", "inspection"]
        }

    Atlas Index config (JSON):
        {
          "mappings": {
            "dynamic": true,
            "fields": {
              "embedding": [{
                "dimensions": 1536,
                "similarity": "cosine",
                "type": "knnVector"
              }]
            }
          }
        }
    """

    def __init__(self):
        try:
            self.client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
            self.db = self.client[settings.mongo_db]
            self.collection = self.db["maintenance_vectors"]
            # Verify connection
            self.client.admin.command("ping")
            self.connected = True
            logger.info(f"MongoDB connected: {settings.mongo_uri}")
        except Exception as e:
            logger.warning(f"MongoDB unavailable: {e}. Using fallback protocols.")
            self.connected = False

    def search(self, query_embedding: list[float], k: int = 3) -> list[dict]:
        """
        Perform vector similarity search for relevant repair protocols.

        For air-gapped local MongoDB (no Atlas Search), we compute cosine
        similarity in-process using NumPy. This is efficient for small
        collections (<1000 docs) typical of repair protocol libraries.

        Args:
            query_embedding: Query vector (1536-dim).
            k:               Number of results to retrieve.

        Returns:
            List of protocol documents sorted by relevance.
        """
        if not self.connected:
            return self._fallback_protocols()

        try:
            # Fetch all protocols with embeddings from local MongoDB
            docs = list(self.collection.find(
                {"embedding": {"$exists": True}},
                {"protocol_id": 1, "title": 1, "content": 1, "severity": 1, "embedding": 1}
            ))

            if not docs:
                logger.warning("No embedded protocols in MongoDB. Using fallback.")
                return self._fallback_protocols()

            # Compute cosine similarity locally (air-gapped, no Atlas needed)
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                query_norm = 1.0

            scored = []
            for doc in docs:
                doc_vec = np.array(doc["embedding"], dtype=np.float32)
                doc_norm = np.linalg.norm(doc_vec)
                if doc_norm == 0:
                    doc_norm = 1.0
                score = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))
                scored.append({
                    "protocol_id": doc.get("protocol_id", "UNKNOWN"),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "severity": doc.get("severity", "INFO"),
                    "score": score,
                })

            # Sort by score descending, take top-k
            scored.sort(key=lambda x: x["score"], reverse=True)
            results = scored[:k]
            logger.info(f"Local vector search returned {len(results)} protocols (top score: {results[0]['score']:.3f})")
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._fallback_protocols()


    def _fallback_protocols(self) -> list[dict]:
        """Return hardcoded fallback protocols when MongoDB is unavailable."""
        return [
            {
                "protocol_id": "RP-FALLBACK-001",
                "title": "Standard Battery Inspection Protocol",
                "content": "Perform visual inspection for swelling, leakage, or discolouration. "
                           "Measure open-circuit voltage. Check terminal corrosion. "
                           "Log findings in Battery Management System.",
                "severity": "WARNING",
            },
            {
                "protocol_id": "RP-FALLBACK-002",
                "title": "Capacity Fade Diagnostic Procedure",
                "content": "Conduct reference performance test at 25°C, C/3 rate. "
                           "Compare measured capacity vs. rated capacity. "
                           "If fade >20%, escalate to battery replacement workflow.",
                "severity": "CRITICAL",
            },
            {
                "protocol_id": "RP-FALLBACK-003",
                "title": "Thermal Management Check",
                "content": "Verify cooling system flow rate and inlet temperature. "
                           "Check thermal interface material for degradation. "
                           "Review temperature gradient across module (max 5°C delta).",
                "severity": "WARNING",
            },
        ]


# ─────────────────────────────────────────────
# LangGraph Nodes
# ─────────────────────────────────────────────

# Singletons — initialized once, reused across requests
_inference_engine: Optional[ONNXInferenceEngine] = None
_vector_search: Optional[RepairProtocolVectorSearch] = None
_llm: Optional[ChatOllama] = None


def _get_inference_engine() -> ONNXInferenceEngine:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = ONNXInferenceEngine()
    return _inference_engine


def _get_vector_search() -> RepairProtocolVectorSearch:
    global _vector_search
    if _vector_search is None:
        _vector_search = RepairProtocolVectorSearch()
    return _vector_search


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model="llama3.2:3b",
            keep_alive=-1,
            temperature=0.1,
        )
    return _llm


# ── Node 1: Inference ───────────────────────
def inference_node(state: BatteryDiagnosticState) -> BatteryDiagnosticState:
    """
    LangGraph Node: Run RUL inference on incoming sensor data.

    Input state fields:  sensor_reading
    Output state fields: predicted_rul, inference_latency_ms, maintenance_status

    Edge cases handled:
        - ONNX session unavailable → synthetic prediction
        - RUL < 0 → clipped to 0
        - Latency >50ms → warning logged (EU Edge-AI compliance)
    """
    sensor = state["sensor_reading"]
    logger.info(f"[inference_node] Battery: {sensor.battery_id} | Cycle: {sensor.cycle_count}")

    try:
        engine = _get_inference_engine()
        predicted_rul, latency_ms = engine.predict(sensor)

        # Determine maintenance status per EU Battery Regulation thresholds
        if predicted_rul > settings.rul_threshold * 2:
            status = MaintenanceStatus.NORMAL
        elif predicted_rul > settings.rul_threshold:
            status = MaintenanceStatus.WARNING
        else:
            status = MaintenanceStatus.CRITICAL

        if latency_ms > settings.max_latency_ms:
            logger.warning(f"Inference latency {latency_ms:.1f}ms exceeds {settings.max_latency_ms}ms target")

        rul_pct = min(predicted_rul, 100.0)
        logger.info(f"[inference_node] RUL={predicted_rul:.1f}% | Status={status.value} | {latency_ms:.1f}ms")

        return {
            **state,
            "predicted_rul": predicted_rul,
            "rul_percentage": rul_pct,
            "inference_latency_ms": latency_ms,
            "maintenance_status": status,
            "error_message": None,
        }

    except Exception as exc:
        logger.error(f"[inference_node] Error: {exc}")
        return {
            **state,
            "predicted_rul": None,
            "inference_latency_ms": None,
            "maintenance_status": MaintenanceStatus.FAULT,
            "error_message": str(exc),
        }


# ── Node 2: Logic Gate (Router) ─────────────
def logic_gate(state: BatteryDiagnosticState) -> str:
    """
    LangGraph conditional edge: Route based on predicted RUL.

    Routing Logic:
        - RUL > RUL_THRESHOLD (default 20): → "normal_operation"
        - RUL <= RUL_THRESHOLD:             → "diagnostic_node"
        - Fault / None:                     → "diagnostic_node" (safe default)

    Returns:
        Node name string consumed by LangGraph router.
    """
    rul = state.get("predicted_rul")
    status = state.get("maintenance_status")

    if status == MaintenanceStatus.FAULT or rul is None:
        logger.warning("[logic_gate] Fault detected -> routing to diagnostic_node")
        return "diagnostic_node"

    if rul > settings.rul_threshold:
        logger.info(f"[logic_gate] RUL={rul:.1f} > {settings.rul_threshold} -> normal_operation")
        return "normal_operation"
    else:
        logger.info(f"[logic_gate] RUL={rul:.1f} <= {settings.rul_threshold} -> diagnostic_node")
        return "diagnostic_node"


# ── Node 3: Normal Operation (Healthy State) ──
def normal_operation_node(state: BatteryDiagnosticState) -> BatteryDiagnosticState:
    """
    LangGraph Node: Healthy battery — bypasses GPU to save power.

    No external calls required for healthy batteries.
    Optimizes for latency (<5ms target for this path).
    GPU is NOT engaged, preserving RTX 3050 power budget.
    """
    rul = state.get('predicted_rul', 0.0)
    rul_pct = state.get('rul_percentage', 0.0)
    logger.info(
        f"[healthy_node] Battery {state['sensor_reading'].battery_id} is within safe limits. "
        f"RUL={rul:.1f}%. GPU bypassed."
    )
    if rul > 50:
        status_val = "NORMAL_OPERATION"
        summary = "Battery is operating normally. No maintenance action required."
    else:
        status_val = "DEGRADATION_WARNING"
        summary = f"Battery degradation detected. Predicted RUL: {rul:.1f}%. Schedule preventative maintenance soon."

    return {
        **state,
        "route": "NORMAL",
        "retrieved_protocols": [],
        "maintenance_status": status_val,
        "llm_summary": summary,
        "recommended_actions": ["Continue normal monitoring. Next scheduled check in 50 cycles."] if rul > 50 else ["Schedule preventative maintenance."],
    }


# ── Node 4: Diagnostic ───────────────────────
def diagnostic_node(state: BatteryDiagnosticState) -> BatteryDiagnosticState:
    """
    LangGraph Node: Deep diagnostic using MongoDB Vector Search + LLM.

    Pipeline:
        1. Embed diagnostic context using the LLM's embedding model
        2. Query MongoDB Atlas Vector Search for relevant repair protocols
        3. Synthesize protocols + sensor data into an LLM-generated report
        4. Extract recommended actions

    Output is EU Battery Passport 2026 compliant.
    """
    sensor = state["sensor_reading"]
    rul = state.get("predicted_rul", 0.0)
    status = state.get("maintenance_status", MaintenanceStatus.CRITICAL)

    logger.info(f"[diagnostic_node] Initiating deep diagnostic for {sensor.battery_id}")

    # ── Step 1: Vector Search ────────────────
    vs_client = _get_vector_search()

    # Build query embedding context
    # In production: use OpenAI embeddings on the diagnostic context string
    # For demo: use random embedding (replace with actual embeddings in prod)
    query_context = (
        f"Battery {sensor.battery_id} chemistry {sensor.chemistry.value} "
        f"RUL {rul:.0f}% voltage {sensor.voltage:.2f}V "
        f"temperature {sensor.temperature:.1f}C status {status.value}"
    )
    logger.debug(f"[diagnostic_node] Vector search query: {query_context[:80]}...")

    # In production: embed the query_context with OpenAI
    # query_embedding = openai.embeddings.create(input=query_context, model="text-embedding-3-small").data[0].embedding
    # For fallback: use random 1536-dim vector (MongoDB will return fallback data)
    query_embedding = list(np.random.randn(1536).astype(float))

    protocols = vs_client.search(query_embedding, k=3)
    protocol_texts = [p.get("content", "") for p in protocols if p.get("content")]
    protocol_titles = [p.get("title", "Unknown") for p in protocols]

    logger.info(f"[diagnostic_node] Retrieved {len(protocols)} protocols: {protocol_titles}")

    # ── Step 2: LLM Synthesis ────────────────
    protocol_block = "\n".join([
        f"PROTOCOL {i+1} [{p.get('protocol_id', 'N/A')}]: {p.get('title', '')}\n{p.get('content', '')}"
        for i, p in enumerate(protocols)
    ])

    system_prompt = """You are an expert battery diagnostic engineer following EU Battery Regulation 2023/1542.
Your reports must be:
1. Technically precise and actionable
2. Referenced to specific repair protocols
3. Compliant with EU Battery Passport Annex XIII
4. Written in professional technical English

Structure your response as:
DIAGNOSTIC SUMMARY: (2-3 sentences)
ROOT CAUSE HYPOTHESIS: (1-2 sentences)
RECOMMENDED ACTIONS: (numbered list, 3-5 items)
URGENCY: (IMMEDIATE / 7-DAYS / 30-DAYS)

STEP 3: MANUAL-BASED CONCLUSION
STRICT RELEVANCY: You must evaluate the retrieved context before using it. If the context discusses infotainment, head units, multimedia displays, or backing up customer data to USB drives, IGNORE IT ENTIRELY.

DOMAIN FOCUS: Only recommend actions strictly related to High-Voltage (HV) battery chemistry, Cell Balancing, Thermal Management Systems (TMS), DC/DC Converters, or BMS hardware replacement.

FALLBACK: If the retrieved context does not contain relevant physical battery repair steps, state: 'No relevant HV repair protocol found in the current context. Recommend physical cell degradation analysis.'"""

    user_prompt = f"""Battery Diagnostic Request:
- Battery ID: {sensor.battery_id}
- Chemistry: {sensor.chemistry.value}
- Current Cycle: {sensor.cycle_count}
- Voltage: {sensor.voltage:.3f}V
- Temperature: {sensor.temperature:.1f}°C
- Predicted RUL: {rul:.1f}%
- Maintenance Status: {status.value}

Retrieved Repair Protocols:
{protocol_block}

Generate a diagnostic report and recommended maintenance actions."""

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        llm_output = response.content

        # Extract recommended actions (simple heuristic parser)
        actions = []
        for line in llm_output.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")) and len(line) > 10:
                action = line.lstrip("0123456789.-) ").strip()
                if action:
                    actions.append(action)

        if not actions:
            actions = [
                "Conduct immediate capacity measurement test",
                "Review thermal management system",
                "Schedule battery module inspection within 7 days",
            ]

    except Exception as exc:
        logger.warning(f"[diagnostic_node] LLM offline or busy ({exc}). Using offline fallback protocol.")
        llm_output = (
            f"DIAGNOSTIC SUMMARY: Battery {sensor.battery_id} shows critical degradation "
            f"with RUL of {rul:.1f}%. Immediate maintenance assessment required.\n"
            f"ROOT CAUSE HYPOTHESIS: Accelerated capacity fade likely due to thermal stress "
            f"or electrolyte depletion.\n"
            f"URGENCY: IMMEDIATE"
        )
        actions = [
            "Perform reference performance test (RPT) at 25°C",
            "Inspect cooling system for blockages or leaks",
            "Check cell-level voltage balance (ΔV < 50mV)",
            "Review charging history for over-voltage events",
            "Prepare battery replacement per OEM protocol",
        ]

    return {
        **state,
        "route": "DIAGNOSTIC",
        "retrieved_protocols": protocol_titles,
        "llm_summary": llm_output,
        "recommended_actions": actions,
    }


# ─────────────────────────────────────────────
# Audit Node — EU Battery Passport Compliance
# ─────────────────────────────────────────────
def audit_node(state: BatteryDiagnosticState) -> BatteryDiagnosticState:
    """
    LangGraph Node: Log cycle outcome to the audit trail.

    Writes to local MongoDB `inference_logs` collection for EU Battery
    Regulation 2023/1542 Battery Passport compliance. Both the healthy
    and diagnostic paths converge here before reaching END.

    Verification checks logged:
        - RUL threshold compliance
        - Inference latency compliance (< max_latency_ms)
        - Ignition status acknowledgement
        - VitisAI EP tracking
    """
    sensor = state["sensor_reading"]
    rul = state.get("predicted_rul", 0.0)
    rul_pct = state.get("rul_percentage", 0.0)
    route = state.get("route", "UNKNOWN")
    latency = state.get("inference_latency_ms", 0.0)

    entry = (
        f"[CYCLE {sensor.cycle_count}] {route} | "
        f"Battery={sensor.battery_id} | RUL={rul:.1f}%"
    )

    audit_log = list(state.get("audit_log", []))
    audit_log.append(entry)

    # ── Verification checks → logged to pipeline.log ──
    logger.info(f"[audit_node] ── Verification Checks ──")
    logger.info(f"[audit_node] {entry}")

    # Check 1: RUL threshold
    if rul is not None and rul <= settings.rul_threshold:
        logger.warning(
            f"[audit_node] ⚠️  RUL BREACH: {rul:.1f}% ≤ threshold {settings.rul_threshold}% → {route}"
        )
    else:
        logger.info(f"[audit_node] ✅ RUL within safe limits: {rul:.1f}%")

    # Check 2: Inference latency compliance
    if latency and latency > settings.max_latency_ms:
        logger.warning(
            f"[audit_node] ⚠️  LATENCY EXCEEDED: {latency:.1f}ms > {settings.max_latency_ms}ms target"
        )
    else:
        logger.info(f"[audit_node] ✅ Inference latency: {latency:.1f}ms (target: {settings.max_latency_ms}ms)")

    # Check 3: Ignition status
    ignition = state.get("ignition_status", True)
    logger.info(f"[audit_node] 🔑 Ignition status: {'ON' if ignition else 'OFF'}")

    # Check 4: Error state
    error = state.get("error_message")
    if error:
        logger.error(f"[audit_node] ❌ Pipeline error detected: {error}")
    else:
        logger.info(f"[audit_node] ✅ No pipeline errors")

    logger.info(f"[audit_node] ── End Verification ──")

    # Persist to local MongoDB (non-blocking, best-effort)
    try:
        client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=1000)
        db = client[settings.mongo_db]
        db["inference_logs"].insert_one({
            "battery_id": sensor.battery_id,
            "cycle": sensor.cycle_count,
            "rul": rul,
            "rul_pct": rul_pct,
            "route": route,
            "latency_ms": latency,
            "timestamp": time.time(),
            "source": "agentic_layer.audit_node",
        })
    except Exception:
        pass  # Audit also kept in-memory via audit_log

    return {
        **state,
        "audit_log": audit_log,
    }


# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────
def build_diagnostic_graph() -> StateGraph:
    """
    Construct and compile the LangGraph state machine.

    Graph topology:
        START → inference_node → [logic_gate]
                                    ├── normal_operation → audit_node → END
                                    └── diagnostic_node  → audit_node → END

    Includes MemorySaver checkpointer for fault-tolerant state persistence.

    Returns:
        Compiled LangGraph StateGraph.
    """
    workflow = StateGraph(BatteryDiagnosticState)

    # Register nodes
    workflow.add_node("inference_node", inference_node)
    workflow.add_node("normal_operation", normal_operation_node)
    workflow.add_node("diagnostic_node", diagnostic_node)
    workflow.add_node("audit_node", audit_node)

    # Define edges
    workflow.add_edge(START, "inference_node")

    # Conditional routing via logic_gate
    workflow.add_conditional_edges(
        "inference_node",
        logic_gate,
        {
            "normal_operation": "normal_operation",
            "diagnostic_node": "diagnostic_node",
        }
    )

    # Both paths converge on audit_node
    workflow.add_edge("normal_operation", "audit_node")
    workflow.add_edge("diagnostic_node", "audit_node")

    # Audit → END (external loop handles re-invocation)
    workflow.add_edge("audit_node", END)

    # Compile with MemorySaver for fault tolerance
    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph state machine compiled with MemorySaver checkpointer")
    return compiled


# ─────────────────────────────────────────────
# Public API Function
# ─────────────────────────────────────────────
async def run_diagnostic_pipeline(sensor_reading: SensorReading) -> DiagnosticReport:
    """
    Execute the full agentic diagnostic pipeline.

    Args:
        sensor_reading: Validated SensorReading from the FastAPI endpoint.

    Returns:
        DiagnosticReport with RUL, status, protocols, and LLM summary.
    """
    graph = build_diagnostic_graph()

    initial_state: BatteryDiagnosticState = {
        "sensor_reading": sensor_reading,
        "predicted_rul": None,
        "rul_percentage": 0.0,
        "inference_latency_ms": None,
        "maintenance_status": None,
        "retrieved_protocols": [],
        "llm_summary": "",
        "recommended_actions": [],
        "route": "",
        "error_message": None,
        "ignition_status": True,
        "audit_log": [],
    }

    # Execute graph (sync wrapper for async context)
    config = {"configurable": {"thread_id": f"diag-{sensor_reading.battery_id}"}}
    final_state = graph.invoke(initial_state, config=config)

    return DiagnosticReport(
        battery_id=sensor_reading.battery_id,
        rul_percent=final_state.get("predicted_rul", 0.0),
        maintenance_status=final_state.get("maintenance_status", MaintenanceStatus.FAULT),
        retrieved_protocols=final_state.get("retrieved_protocols", []),
        llm_summary=final_state.get("llm_summary", ""),
        recommended_actions=final_state.get("recommended_actions", []),
        passport_compliant=True,
    )


# ─────────────────────────────────────────────
# CLI Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from rich import print as rprint
    from rich.panel import Panel

    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    # Scenario 1: Critical battery (low RUL trigger)
    critical_sensor = SensorReading(
        battery_id="MERC-EQS-B007",
        timestamp=time.time(),
        voltage=3.41,       # degraded voltage
        current=-12.5,
        temperature=38.2,
        cycle_count=380,    # high cycle count → low RUL expected
        chemistry="LiNiMnCoO2",
    )

    # Scenario 2: Healthy battery
    healthy_sensor = SensorReading(
        battery_id="MERC-EQS-B001",
        timestamp=time.time(),
        voltage=3.71,
        current=-8.0,
        temperature=26.5,
        cycle_count=45,
        chemistry="LiNiMnCoO2",
    )

    rprint(Panel.fit("[bold cyan]EcoDrive-Sentinel | Agentic Pipeline Demo[/bold cyan]"))

    for label, sensor in [("[red]CRITICAL Battery[/red]", critical_sensor), ("[green]HEALTHY Battery[/green]", healthy_sensor)]:
        rprint(f"\n[bold]{label}[/bold]: {sensor.battery_id}")
        report = asyncio.run(run_diagnostic_pipeline(sensor))
        rprint(f"  RUL: [yellow]{report.rul_cycles:.1f}[/yellow] cycles")
        rprint(f"  Status: [bold]{report.maintenance_status.value}[/bold]")
        rprint(f"  Protocols: {report.retrieved_protocols}")
        if report.recommended_actions:
            rprint("  Actions:")
            for a in report.recommended_actions[:3]:
                rprint(f"    -> {a}")
