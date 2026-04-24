📋 Project Roadmap: EcoDrive-SentinelGoal: Build a predictive maintenance system for EV batteries with an Agentic diagnostic layer.

Status: 🏗️ Phase 1 (Data & Baseline ML)


🟢 Phase 1: Data Engineering & Baseline ML (Week 1)
Focus: Mastering the NASA dataset and building the "Predictive Brain".[ ] 
Task 1.1: Data Exploration. Load metadata.csv and visualize Capacity Fade across Batteries B0005, B0006, and B0007.[ ] 
Task 1.2: Feature Extraction. Create a script to calculate Mean_Voltage, Max_Temp, and Discharge_Time for every discharge cycle.[ ] 
Task 1.3: Target Labeling. Calculate $RUL = (Total\_Cycles - Current\_Cycle)$ for every row in your training set.[ ] 
Task 1.4: Random Forest Implementation. Train a regressor to predict RUL and log the Mean Absolute Error (MAE).[ ] 
Task 1.5: Feature Importance Analysis. Identify which sensor (Temp vs. Voltage) most impacts battery life.


🟡 Phase 2: Deep Learning & API (Week 2)
Focus: Enhancing accuracy and preparing for deployment.[ ] 
Task 2.1: LSTM Implementation. Window the data (e.g., use the last 5 cycles to predict the next) and train a PyTorch LSTM.[ ] 
Task 2.2: Model Comparison. Create a table comparing Random Forest vs. LSTM performance.[ ] Task 2.3: FastAPI Development. Build an endpoint /predict that takes sensor values and returns the RUL.[ ] 
Task 2.4: Dockerization. Create a Dockerfile and run your ML API inside a container.


🔵 Phase 3: Agentic Orchestration (Week 3)
Focus: The "Secret Sauce" – turning the model into an Autonomous Agent.[ ] 
Task 3.1: Vector Store Setup. Ingest EV battery technical manuals into MongoDB Atlas Vector Search.[ ] 
Task 3.2: LangGraph State Design. Define the nodes: Predict_Health, Check_Threshold, and Retrieve_Manual.[ ] 
Task 3.3: Conditional Routing. Logic: If RUL < 20% -> Route to Vector_Search -> Generate Report.[ ] 
Task 3.4: Local LLM Integration. Use Ollama to power the "Reasoning" node of your graph for offline diagnostics.


🔴 Phase 4: Validation & Documentation (Week 4)
Focus: Professionalism and HOD Submission.[ ] 
Task 4.1: PyTest & RAGAS. Write tests for your API and evaluate the "Faithfulness" of the Agent's repair suggestions.[ ] 
Task 4.2: Mermaid Flowcharts. Finalize the system architecture diagrams for the report.[ ] Task 4.3: Final Project Report. Compile the abstract, methodology, results, and the "Future Scope" (German Automotive Standards).