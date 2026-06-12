📋 Project Roadmap: EcoDrive-Sentinel
Goal: Build a predictive maintenance system for EV batteries with an Agentic diagnostic layer.

Status: ✅ ALL PHASES COMPLETE — System Operational


🟢 Phase 1: Data Engineering & Baseline ML ✅ COMPLETE
[x] Task 1.1: Data Exploration. Load metadata.csv and visualize Capacity Fade across B0005, B0006, B0007.
[x] Task 1.2: Feature Extraction. Calculate Mean_Voltage, Max_Temp, Discharge_Time per cycle.
[x] Task 1.3: Target Labeling. Calculate RUL = (Total_Cycles - Current_Cycle) for every row.
[x] Task 1.4: Random Forest Implementation. Train regressor and log MAE.
[x] Task 1.5: Feature Importance Analysis. Identify which sensor most impacts battery life.


🟢 Phase 2: Deep Learning & API ✅ COMPLETE
[x] Task 2.1: CNN-LSTM Implementation. Window data (last 30 cycles) and train PyTorch CNN-LSTM.
[x] Task 2.2: Model Comparison. Random Forest vs. CNN-LSTM performance table.
[x] Task 2.3: FastAPI Development. Built /predict endpoint with Pydantic validation.
[x] Task 2.4: ONNX Export. Export CNN-LSTM to ONNX format for edge deployment.


🟢 Phase 3: Agentic Orchestration ✅ COMPLETE
[x] Task 3.1: Vector Store Setup. Ingested 5 repair protocols into MongoDB maintenance_vectors.
[x] Task 3.2: Antigravity State Design. Replaced LangGraph with reactive Supervision Tree.
[x] Task 3.3: Conditional Routing. If RUL < 20% → Vector Search → Generate Report.
[x] Task 3.4: Local LLM Integration. Ollama (Llama 3) powers air-gapped diagnostic reasoning.
[x] Task 3.5: NPU Hardware Acceleration. VitisAIExecutionProvider active on Ryzen AI 8645HS.
[x] Task 3.6: INT8 Quantization. CNN-LSTM quantized for XDNA NPU compute tiles.


🟢 Phase 4: Validation & Documentation ✅ COMPLETE
[x] Task 4.1: RAGAS Evaluation. Faithfulness: 0.89 | Answer Relevancy: 0.84.
[x] Task 4.2: NPU Stress Test. 30s stress test passed — latency well under 500ms threshold.
[x] Task 4.3: Lifecycle Verification. Full air-gapped lifecycle (Ingest → NPU → Ollama) verified.
[x] Task 4.4: System Health Report. Generated System_Health_Report.json — Status: OPERATIONAL.
[x] Task 4.5: Final Documentation. Mermaid diagrams, walkthrough, and performance report complete.