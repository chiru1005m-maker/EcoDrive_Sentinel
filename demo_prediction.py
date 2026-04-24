
import pandas as pd
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path

def demo_nasa_prediction(battery_id="B0005", end_cycle=100):
    print(f"--- Demo Prediction for {battery_id} (Window ended at Cycle {end_cycle}) ---")
    
    # 1. Load the feature matrix
    matrix_path = Path("data/feature_matrix.parquet")
    if not matrix_path.exists():
        print("Error: feature_matrix.parquet not found.")
        return
    
    df = pd.read_parquet(matrix_path)
    
    # 2. Check available batteries
    if battery_id not in df["battery_id"].unique():
        battery_id = df["battery_id"].unique()[0]
        print(f"Battery B0005 not found. Using available battery: {battery_id}")

    battery_df = df[df["battery_id"] == battery_id].sort_values("cycle_number")
    
    # 3. Get sequence parameters from model checkpoint
    model_ckpt_path = Path("models/cnn_lstm.pt")
    if not model_ckpt_path.exists():
        print("Error: PyTorch model checkpoint not found.")
        return
        
    ckpt = torch.load(model_ckpt_path, map_location="cpu")
    feature_cols = ckpt.get("feature_cols", [])
    seq_len = ckpt.get("model_config", {}).get("seq_len", 30)
    scaler_mean = ckpt.get("scaler_mean")
    scaler_scale = ckpt.get("scaler_scale")

    if end_cycle < seq_len:
        end_cycle = seq_len
        print(f"Adjusting end cycle to minimum required window length: {end_cycle}")

    # Extract the sequence [end_cycle - seq_len, ..., end_cycle]
    # In the feature matrix, cycle_number is 0-indexed or 1-indexed?
    # Usually 0, 1, 2...
    sequence_data = battery_df[battery_df["cycle_number"] <= end_cycle].tail(seq_len)
    
    if len(sequence_data) < seq_len:
         print(f"Error: Not enough cycles for battery {battery_id}. Found {len(sequence_data)}, need {seq_len}.")
         return

    # 4. Prepare and Scale features
    X_raw = sequence_data[feature_cols].values.astype(np.float32)
    # Scale manual implementation (StandardScaler: (x - mean) / scale)
    X_scaled = (X_raw - scaler_mean) / scaler_scale
    
    actual_rul = sequence_data["rul"].iloc[-1]
    
    # Reshape for ONNX: (batch, seq_len, features)
    X_reshaped = X_scaled.reshape(1, seq_len, -1).astype(np.float32)
    
    # 5. Run Inference using ONNX
    onnx_path = Path("onnx/cnn_lstm.onnx")
    if not onnx_path.exists():
        print("Error: ONNX model not found.")
        return
        
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {session.get_inputs()[0].name: X_reshaped}
    prediction = session.run(None, inputs)[0]
    
    predicted_rul = float(prediction[0][0])
    
    # 6. Show Results
    print(f"\nBattery Status Details:")
    print(f"  - Source: NASA PCoE")
    print(f"  - Current Cycle Number: {end_cycle}")
    print(f"  - Input Sequence Range: Cycles {sequence_data['cycle_number'].iloc[0]} to {sequence_data['cycle_number'].iloc[-1]}")
    
    print(f"\nPrediction Outcome:")
    print(f"  - [ACTUAL] Remaining Useful Life: {actual_rul:.1f} cycles")
    print(f"  - [PREDICTED] Remaining Useful Life: {predicted_rul:.1f} cycles")
    
    error = abs(predicted_rul - actual_rul)
    print(f"  - Absolute Error: {error:.2f} cycles")
    
    if error < 10:
        print("\n✅ HIGH ACCURACY: The model is tracking the degradation curve perfectly.")
    elif error < 30:
        print("\n🟡 NOMINAL ACCURACY: The model is within the expected error margin for this chemistry.")
    else:
        print("\n🔴 DEVIATION: The battery is behaving slightly differently than the training set baseline.")

if __name__ == "__main__":
    demo_nasa_prediction()
