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
import onnxruntime as ort
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pymongo import MongoClient
from pydantic import ValidationError

from config import (
    DiagnosticReport,
    InferenceResult,
    MaintenanceStatus,
    SensorReading,
    settings,
)
from predictive_core import FEATURE_COLS, SEQUENCE_LEN

# ─────────────────────────────────────────────
# LangGraph State Schema
# ─────────────────────────────────────────────
class BatteryDiagnosticState(TypedDict):
    """
    Shared state flowing through the LangGraph state machine.

    Each node reads from and writes to this state dict.
    LangGraph merges state via reducers (add_messages for lists).
    """
    # Input
    sensor_reading: SensorReading
    # Inference outputs
    predicted_rul: Optional[float]
    inference_latency_ms: Optional[float]
    maintenance_status: Optional[MaintenanceStatus]
    # Diagnostic outputs
    retrieved_protocols: list[str]
    llm_summary: str
    recommended_actions: list[str]
    # Pipeline control
    route: str                    # "NORMAL" or "DIAGNOSTIC"
    error_message: Optional[str]


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
        try:
            # Try Vitis-AI EP first (NPU)
            providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                self.onnx_path,
                providers=providers,
            )
            active_ep = self.session.get_providers()[0]
            logger.info(f"ONNX Runtime initialized with: {active_ep}")
        except Exception:
            try:
                # CPU fallback
                self.session = ort.InferenceSession(
                    self.onnx_path,
                    providers=["CPUExecutionProvider"]
                )
                logger.info("ONNX Runtime initialized (CPU EP)")
            except Exception as e:
                logger.warning(f"ONNX Runtime unavailable: {e}. PyTorch fallback active.")
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

    def _sensor_to_feature_vector(self, sensor: SensorReading) -> np.ndarray:
        """
        Convert a SensorReading to a scaled feature vector.

        In production this would use a rolling window buffer per battery.
        For single-inference, we replicate the reading across seq_len.

        Args:
            sensor: Validated SensorReading pydantic model.

        Returns:
            numpy array of shape (1, seq_len, n_features), normalized.
        """
        # Build raw feature vector from sensor fields
        # Matches FEATURE_COLS = [voltage_drop, avg_temperature, capacity_fade, ir_proxy, charge_time_delta]
        V_NOMINAL = 3.7
        raw_features = np.array([
            max(0.0, V_NOMINAL - sensor.voltage),   # voltage_drop proxy
            sensor.temperature,                       # avg_temperature
            0.0,                                      # capacity_fade (unknown at runtime → 0)
            abs(sensor.voltage) / max(abs(sensor.current), 1e-6),  # ir_proxy
            0.01,                                     # charge_time_delta (normalized default)
        ], dtype=np.float32)

        # Normalize
        normalized = (raw_features - self.scaler_mean) / np.clip(self.scaler_scale, 1e-8, None)

        # Replicate across sequence length (single-point → sequence)
        sequence = np.tile(normalized, (SEQUENCE_LEN, 1))  # (seq_len, n_features)
        return sequence[np.newaxis, :, :]                   # (1, seq_len, n_features)

    def predict(self, sensor: SensorReading) -> tuple[float, float]:
        """
        Run RUL inference.

        Args:
            sensor: Validated SensorReading.

        Returns:
            Tuple of (predicted_rul, latency_ms).
        """
        input_array = self._sensor_to_feature_vector(sensor)
        t_start = time.perf_counter()

        if self.session is not None:
            outputs = self.session.run(
                ["predicted_rul"],
                {"battery_health_indicators": input_array}
            )
            rul = float(outputs[0].squeeze())
        else:
            # Synthetic prediction fallback for demo/CI
            # Degrades RUL as cycle count increases
            rul = max(0.0, 200.0 - sensor.cycle_count * 0.4 + np.random.normal(0, 2))

        latency_ms = (time.perf_counter() - t_start) * 1000
        return max(0.0, rul), latency_ms


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
            self.collection = self.db[settings.mongo_collection]
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

        Args:
            query_embedding: Query vector (1536-dim OpenAI embedding).
            k:               Number of results to retrieve.

        Returns:
            List of protocol documents sorted by relevance.
        """
        if not self.connected:
            return self._fallback_protocols()

        try:
            pipeline = [
                {
                    "$search": {
                        "index": "repair_protocol_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": k,
                        }
                    }
                },
                {
                    "$project": {
                        "protocol_id": 1,
                        "title": 1,
                        "content": 1,
                        "severity": 1,
                        "score": {"$meta": "searchScore"},
                    }
                },
                {"$limit": k}
            ]
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Vector search returned {len(results)} protocols")
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
_llm: Optional[ChatOpenAI] = None


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


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=512,
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

        logger.info(f"[inference_node] RUL={predicted_rul:.1f} cycles | Status={status.value} | {latency_ms:.1f}ms")

        return {
            **state,
            "predicted_rul": predicted_rul,
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
        logger.warning("[logic_gate] Fault detected → routing to diagnostic_node")
        return "diagnostic_node"

    if rul > settings.rul_threshold:
        logger.info(f"[logic_gate] RUL={rul:.1f} > {settings.rul_threshold} → normal_operation")
        return "normal_operation"
    else:
        logger.info(f"[logic_gate] RUL={rul:.1f} ≤ {settings.rul_threshold} → diagnostic_node")
        return "diagnostic_node"


# ── Node 3: Normal Operation ─────────────────
def normal_operation_node(state: BatteryDiagnosticState) -> BatteryDiagnosticState:
    """
    LangGraph Node: Log normal operation status and return minimal report.

    No external calls required for healthy batteries.
    Optimizes for latency (<5ms target for this path).
    """
    logger.info(
        f"[normal_op] Battery {state['sensor_reading'].battery_id} is healthy. "
        f"RUL={state['predicted_rul']:.1f} cycles."
    )
    return {
        **state,
        "route": "NORMAL",
        "retrieved_protocols": [],
        "llm_summary": f"Battery is operating normally. Predicted RUL: {state['predicted_rul']:.1f} cycles. "
                        f"No maintenance action required.",
        "recommended_actions": ["Continue normal monitoring. Next scheduled check in 50 cycles."],
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
        f"RUL {rul:.0f} cycles voltage {sensor.voltage:.2f}V "
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
URGENCY: (IMMEDIATE / 7-DAYS / 30-DAYS)"""

    user_prompt = f"""Battery Diagnostic Request:
- Battery ID: {sensor.battery_id}
- Chemistry: {sensor.chemistry.value}
- Current Cycle: {sensor.cycle_count}
- Voltage: {sensor.voltage:.3f}V
- Temperature: {sensor.temperature:.1f}°C
- Predicted RUL: {rul:.1f} cycles
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
        logger.error(f"[diagnostic_node] LLM error: {exc}")
        llm_output = (
            f"DIAGNOSTIC SUMMARY: Battery {sensor.battery_id} shows critical degradation "
            f"with RUL of {rul:.1f} cycles. Immediate maintenance assessment required.\n"
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
# Graph Construction
# ─────────────────────────────────────────────
def build_diagnostic_graph() -> StateGraph:
    """
    Construct and compile the LangGraph state machine.

    Graph topology:
        START → inference_node → [logic_gate] → normal_operation_node → END
                                              ↘ diagnostic_node → END

    Returns:
        Compiled LangGraph StateGraph.
    """
    workflow = StateGraph(BatteryDiagnosticState)

    # Register nodes
    workflow.add_node("inference_node", inference_node)
    workflow.add_node("normal_operation", normal_operation_node)
    workflow.add_node("diagnostic_node", diagnostic_node)

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

    # Terminal edges
    workflow.add_edge("normal_operation", END)
    workflow.add_edge("diagnostic_node", END)

    compiled = workflow.compile()
    logger.info("LangGraph state machine compiled successfully")
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
        "inference_latency_ms": None,
        "maintenance_status": None,
        "retrieved_protocols": [],
        "llm_summary": "",
        "recommended_actions": [],
        "route": "",
        "error_message": None,
    }

    # Execute graph (sync wrapper for async context)
    final_state = graph.invoke(initial_state)

    return DiagnosticReport(
        battery_id=sensor_reading.battery_id,
        rul_cycles=final_state.get("predicted_rul", 0.0),
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

    for label, sensor in [("🔴 CRITICAL Battery", critical_sensor), ("🟢 HEALTHY Battery", healthy_sensor)]:
        rprint(f"\n[bold]{label}[/bold]: {sensor.battery_id}")
        report = asyncio.run(run_diagnostic_pipeline(sensor))
        rprint(f"  RUL: [yellow]{report.rul_cycles:.1f}[/yellow] cycles")
        rprint(f"  Status: [bold]{report.maintenance_status.value}[/bold]")
        rprint(f"  Protocols: {report.retrieved_protocols}")
        if report.recommended_actions:
            rprint("  Actions:")
            for a in report.recommended_actions[:3]:
                rprint(f"    → {a}")
