from __future__ import annotations
# ──────────────────────────────────────────────────────────
# EcoDrive-Sentinel | Stateful LangGraph Orchestrator
# ──────────────────────────────────────────────────────────
# Refactored from custom Antigravity reactive streams to a production-grade
# LangGraph state machine with conditional routing and MemorySaver persistence.
#
# Hardware Allocation:
#     CPU  -> LangGraph state management, MongoDB I/O
#     NPU  -> CNN-LSTM RUL inference (VitisAI EP, Ryzen AI 8645HS)
#     GPU  -> Diagnostic reasoning (Ollama Llama 3, RTX 3050 6GB VRAM)
#
# State Machine Topology:
#     [START] -> [telemetry_node] -> [npu_inference_node] -> [route]
#                                                              |
#                     healthy -> [END]  <-----------------------+
#                     critical -> [diagnostic_reasoning_node] -> [END]
#
# Author: EcoDrive-Sentinel Team
# ──────────────────────────────────────────────────────────

import sys
sys.stdout.reconfigure(encoding='utf-8')


import os
import time
import asyncio
import yaml
import numpy as np
import httpx
from typing import Any, Optional, TypedDict
from loguru import logger
from pymongo import MongoClient

# Point directly to the AMD installation runtime firmware path
os.environ["XLNX_VART_FIRMWARE"] = r"C:\Program Files\AMD\RyzenAI\1x4.xclbin"

import onnxruntime as ort

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.core.config import SensorReading, settings, PROJECT_ROOT
from src.agents.agentic_layer import ONNXInferenceEngine, RepairProtocolVectorSearch

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MONGO_URI = settings.mongo_uri
DB_NAME = settings.mongo_db
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
RUL_THRESHOLD = 20
POLL_INTERVAL = 1.0
MODEL_PATH = settings.onnx_path
MAX_RUL_CYCLES = 200.0  # Assumed maximum RUL for percentage calculation

# ─────────────────────────────────────────────
# Structural Logging Sink → logs/pipeline.log
# ─────────────────────────────────────────────
_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "pipeline.log"

# Add file sink: captures ALL levels (DEBUG+) with structured format
logger.add(
    str(_LOG_FILE),
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} | {message}",
    enqueue=True,   # thread-safe async writes
    backtrace=True,
    diagnose=True,
)


# ─────────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────────
class SentinelState(TypedDict):
    """
    Stateful graph context flowing through every LangGraph node.

    Fields:
        telemetry_buffer  : Last N sensor readings for the active battery.
        battery_id        : Active battery identifier.
        cycle_count       : Current cycle number (monotonically increasing).
        npu_rul_prediction: Raw RUL output from the CNN-LSTM on the NPU.
        rul_percentage    : RUL as a percentage of max expected life.
        inference_latency : Last inference round-trip in milliseconds.
        is_critical       : True if RUL ≤ threshold (triggers diagnostic node).
        diagnostic_report : LLM-generated maintenance plan (empty if healthy).
        protocols_used    : Repair protocols retrieved from MongoDB.
        active_ep         : Active ONNX execution provider name.
        iteration         : Loop counter for the continuous monitoring cycle.
        error             : Last error message (empty if no error).
    """
    telemetry_buffer: list[dict[str, Any]]
    battery_id: str
    cycle_count: int
    npu_rul_prediction: float
    rul_percentage: float
    inference_latency: float
    is_critical: bool
    diagnostic_report: str
    protocols_used: list[str]
    active_ep: str
    iteration: int
    error: str


# ─────────────────────────────────────────────
# 2. HARDWARE SINGLETONS (initialized once)
# ─────────────────────────────────────────────
_npu_engine: Optional[ONNXInferenceEngine] = None
_vector_search: Optional[RepairProtocolVectorSearch] = None
_mongo_client: Optional[MongoClient] = None


def _get_npu_engine() -> ONNXInferenceEngine:
    """Lazy-init the NPU inference engine (VitisAI EP)."""
    global _npu_engine
    if _npu_engine is None:
        _npu_engine = ONNXInferenceEngine(MODEL_PATH)
        ep = _npu_engine.session.get_providers()[0] if _npu_engine.session else "None"
        logger.info(f"🧠 NPU Engine initialized | Active EP: {ep}")
    return _npu_engine


def _get_vector_search() -> RepairProtocolVectorSearch:
    """Lazy-init the MongoDB vector search client."""
    global _vector_search
    if _vector_search is None:
        _vector_search = RepairProtocolVectorSearch()
    return _vector_search


def _get_mongo_db():
    """Lazy-init the MongoDB client."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return _mongo_client[DB_NAME]


# ─────────────────────────────────────────────
# 3. NODE IMPLEMENTATIONS
# ─────────────────────────────────────────────

def telemetry_node(state: SentinelState) -> SentinelState:
    """
    [CPU] Telemetry Ingestion Node.

    Pulls the latest sensor reading from MongoDB or generates a synthetic
    OBD-II reading for demo purposes. Appends to the rolling telemetry buffer
    (max 30 readings to match the CNN-LSTM sequence length).
    """
    iteration = state.get("iteration", 0) + 1
    cycle = state.get("cycle_count", 99) + 1

    # ── Try MongoDB first, fall back to synthetic ──
    reading_dict = None
    try:
        db = _get_mongo_db()
        latest = db["battery_telemetry"].find_one(
            sort=[("timestamp", -1)]
        )
        if latest and "_id" in latest:
            latest.pop("_id")
            reading_dict = latest
    except Exception:
        pass

    # ── Synthetic OBD-II fallback ──
    if reading_dict is None:
        reading_dict = {
            "battery_id": "B0005",
            "voltage": 3.6 + np.random.normal(0, 0.05),
            "current": -2.0 + np.random.normal(0, 0.1),
            "temperature": 25.0 + np.random.normal(0, 0.5),
            "cycle_count": cycle,
            "timestamp": time.time(),
        }

    # ── Maintain rolling buffer (last 30) ──
    buffer = list(state.get("telemetry_buffer", []))
    buffer.append(reading_dict)
    if len(buffer) > 30:
        buffer = buffer[-30:]

    logger.debug(
        f"📡 Telemetry #{iteration} | Cycle {cycle} | "
        f"V={reading_dict.get('voltage', 0):.2f}V | "
        f"T={reading_dict.get('temperature', 0):.1f}°C"
    )

    return {
        **state,
        "telemetry_buffer": buffer,
        "battery_id": reading_dict.get("battery_id", "UNKNOWN"),
        "cycle_count": cycle,
        "iteration": iteration,
        "error": "",
    }


def npu_inference_node(state: SentinelState) -> SentinelState:
    """
    [NPU] CNN-LSTM Inference Node.

    Runs the quantized/FP32 CNN-LSTM model on the Ryzen AI NPU via the
    VitisAIExecutionProvider. Computes RUL and sets the is_critical flag
    based on the configured threshold (default: 20%).
    """
    engine = _get_npu_engine()
    active_ep = engine.session.get_providers()[0] if engine.session else "Synthetic"

    # Build SensorReading from latest buffer entry
    latest = state["telemetry_buffer"][-1]
    try:
        sensor = SensorReading(**latest)
        predicted_rul, latency_ms = engine.predict(sensor)
    except Exception as e:
        logger.error(f"NPU inference error: {e}")
        return {
            **state,
            "npu_rul_prediction": 0.0,
            "rul_percentage": 0.0,
            "inference_latency": 0.0,
            "is_critical": True,
            "active_ep": active_ep,
            "error": str(e),
        }

    rul_pct = min(predicted_rul, 100.0)
    is_critical = rul_pct <= RUL_THRESHOLD

    status_icon = "⚠️ CRITICAL" if is_critical else "✅ Healthy"
    logger.info(
        f"🔋 Cycle {state['cycle_count']} | "
        f"RUL: {predicted_rul:.1f} cycles ({rul_pct:.1f}%) | "
        f"{status_icon} | "
        f"Latency: {latency_ms:.1f}ms | EP: {active_ep}"
    )

    return {
        **state,
        "npu_rul_prediction": predicted_rul,
        "rul_percentage": rul_pct,
        "inference_latency": latency_ms,
        "is_critical": is_critical,
        "active_ep": active_ep,
        "error": "",
    }


def diagnostic_reasoning_node(state: SentinelState) -> SentinelState:
    """
    [GPU] Diagnostic Reasoning Node — only triggered when is_critical=True.

    1. Queries MongoDB maintenance_vectors via local cosine similarity search.
    2. Constructs a context-aware prompt from retrieved protocols.
    3. Sends the prompt to Ollama (Llama 3) running on the RTX 3050 GPU.
    4. Returns the generated repair plan as the diagnostic_report.
    """
    logger.warning(
        f"🚨 LOW RUL DETECTED: {state['rul_percentage']:.1f}% "
        f"(threshold: {RUL_THRESHOLD}%) — Triggering Diagnostic Reasoning"
    )

    # ── Step 1: Vector Search for repair protocols ──
    vs = _get_vector_search()
    dummy_query = [0.0] * 1536  # Placeholder embedding for air-gapped demo
    protocols = vs.search(dummy_query, k=3)
    protocol_titles = [p.get("title", "Unknown") for p in protocols]
    protocol_context = "\n".join([
        f"[{p.get('severity', 'INFO')}] {p.get('title', '')}: {p.get('content', '')}"
        for p in protocols
    ])

    logger.info(f"📚 Retrieved {len(protocols)} repair protocols from MongoDB")

    # ── Step 2: LLM Reasoning via Ollama (RTX 3050 GPU) ──
    prompt = f"""[ECODRIVE-SENTINEL MAINTENANCE ALERT]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Battery ID: {state['battery_id']}
Current Cycle: {state['cycle_count']}
Predicted RUL: {state['npu_rul_prediction']:.1f} cycles ({state['rul_percentage']:.1f}%)
Active Hardware: {state['active_ep']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RELEVANT REPAIR PROTOCOLS:
{protocol_context}

TASK: Based on the protocols above, provide a concise EU Battery Passport
compliant diagnostic report with:
1. Root cause analysis of the degradation
2. Immediate maintenance actions required
3. Estimated time-to-failure
4. Safety classification (NORMAL / WARNING / CRITICAL)

Keep the response under 250 words. Be specific and actionable."""

    diagnostic_report = ""
    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        if response.status_code == 200:
            diagnostic_report = response.json().get("response", "")
            logger.success("📋 Diagnostic Report generated via Ollama (RTX 3050 GPU)")
            print("\n" + "=" * 60)
            print("📋 MAINTENANCE DIAGNOSTIC REPORT")
            print("=" * 60)
            print(diagnostic_report)
            print("=" * 60 + "\n")
        else:
            diagnostic_report = f"Ollama error: HTTP {response.status_code}"
            logger.error(diagnostic_report)
    except Exception as e:
        diagnostic_report = f"Ollama unreachable: {e}"
        logger.error(diagnostic_report)

    return {
        **state,
        "diagnostic_report": diagnostic_report,
        "protocols_used": protocol_titles,
        "error": "",
    }


# ─────────────────────────────────────────────
# 4. CONDITIONAL EDGE LOGIC
# ─────────────────────────────────────────────

def route_after_inference(state: SentinelState) -> str:
    """
    Conditional router: determines the next node after NPU inference.

    Returns:
        "diagnostic_reasoning_node" — if RUL ≤ threshold (battery critical)
        END                        — if healthy (cycle complete, external loop continues)
    """
    if state.get("is_critical", False):
        return "diagnostic_reasoning_node"
    return END


# ─────────────────────────────────────────────
# 5. GRAPH CONSTRUCTION
# ─────────────────────────────────────────────

def build_sentinel_graph() -> StateGraph:
    """
    Construct the EcoDrive-Sentinel LangGraph state machine.

    Graph Topology:
        START → telemetry → npu_inference → [route]
                    ↑                          ↓
                    └──── healthy ←────────────┘
                                               ↓
                                          diagnostic → telemetry (loop)
    """
    graph = StateGraph(SentinelState)

    # ── Register Nodes ──
    graph.add_node("telemetry_node", telemetry_node)
    graph.add_node("npu_inference_node", npu_inference_node)
    graph.add_node("diagnostic_reasoning_node", diagnostic_reasoning_node)

    # ── Entry Point ──
    graph.add_edge(START, "telemetry_node")

    # ── Telemetry → NPU Inference (always) ──
    graph.add_edge("telemetry_node", "npu_inference_node")

    # ── Conditional Routing after Inference ──
    graph.add_conditional_edges(
        "npu_inference_node",
        route_after_inference,
        {
            "diagnostic_reasoning_node": "diagnostic_reasoning_node",
            END: END,
        },
    )

    # ── After Diagnostic → END (cycle complete) ──
    graph.add_edge("diagnostic_reasoning_node", END)

    return graph


# ─────────────────────────────────────────────
# 6. EXECUTION ENGINE
# ─────────────────────────────────────────────

def run_sentinel(max_iterations: int = 0, poll_interval: float = POLL_INTERVAL):
    """
    Boot and run the EcoDrive-Sentinel LangGraph state machine.

    Args:
        max_iterations: Maximum number of telemetry cycles (0 = infinite).
        poll_interval:  Seconds between telemetry polls (default: 0.5s).
    """
    # ── Configure console logger ──
    logger.info("Initializing Sentinel pipeline logging...")
    logger.info(f"Pipeline log sink active: {_LOG_FILE}")

    # ── Build graph with MemorySaver for fault-tolerance ──
    graph = build_sentinel_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # ── Initial State ──
    initial_state: SentinelState = {
        "telemetry_buffer": [],
        "battery_id": "B0005",
        "cycle_count": 99,
        "npu_rul_prediction": 0.0,
        "rul_percentage": 100.0,
        "inference_latency": 0.0,
        "is_critical": False,
        "diagnostic_report": "",
        "protocols_used": [],
        "active_ep": "Initializing...",
        "iteration": 0,
        "error": "",
    }

    # ── Thread config for MemorySaver persistence ──
    config = {"configurable": {"thread_id": "sentinel-main-loop"}}

    # ── Log system banner ──
    logger.info("=" * 60)
    logger.info("🛡️  ECODRIVE-SENTINEL | LangGraph State Machine")
    logger.info("=" * 60)
    logger.info(f"   Hardware:  CPU (LangGraph) + NPU (VitisAI) + GPU (Ollama)")
    logger.info(f"   Model:     {MODEL_PATH}")
    logger.info(f"   Threshold: RUL ≤ {RUL_THRESHOLD}% → Diagnostic Reasoning")
    logger.info(f"   Poll Rate: {poll_interval * 1000:.0f}ms")
    logger.info(f"   Persistence: MemorySaver (checkpointed)")
    limit_str = f"{max_iterations} cycles" if max_iterations > 0 else "∞ (Ctrl+C to stop)"
    logger.info(f"   Iterations: {limit_str}")
    logger.info("=" * 60)

    iteration = 0
    try:
        while True:
            iteration += 1
            if 0 < max_iterations < iteration:
                logger.info(f"🏁 Reached max iterations ({max_iterations}). Shutting down.")
                break

            # ── Invoke the full graph cycle ──
            result = app.invoke(initial_state, config=config)

            # ── Update state for next iteration ──
            initial_state = result

            # ── Respect the poll interval ──
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.warning("🛑 Sentinel shutdown requested (Ctrl+C)")

    # ── Log final state summary ──
    logger.info("=" * 60)
    logger.info("📊 FINAL STATE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   Battery:     {initial_state.get('battery_id', 'N/A')}")
    logger.info(f"   Last Cycle:  {initial_state.get('cycle_count', 'N/A')}")
    logger.info(f"   Last RUL:    {initial_state.get('npu_rul_prediction', 0):.1f} cycles "
          f"({initial_state.get('rul_percentage', 0):.1f}%)")
    logger.info(f"   Is Critical: {initial_state.get('is_critical', False)}")
    logger.info(f"   Active EP:   {initial_state.get('active_ep', 'N/A')}")
    logger.info(f"   Iterations:  {initial_state.get('iteration', 0)}")
    if initial_state.get("diagnostic_report"):
        logger.info(f"   Last Report: {initial_state['diagnostic_report'][:100]}...")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
# 7. ENTRYPOINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EcoDrive-Sentinel LangGraph Orchestrator")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Max iterations (0=infinite)")
    parser.add_argument("--poll-ms", type=int, default=int(POLL_INTERVAL * 1000),
                        help="Telemetry poll interval in ms")
    args = parser.parse_args()

    run_sentinel(
        max_iterations=args.cycles,
        poll_interval=args.poll_ms / 1000.0,
    )
