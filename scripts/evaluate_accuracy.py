import pandas as pd
import numpy as np
import onnxruntime as ort
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader
from rich import print as rprint
from rich.table import Table

from src.core.predictive_core import FEATURE_COLS, TARGET_COL, BatterySequenceDataset

def evaluate_accuracy():
    print("Loading data...")
    feature_df = pd.read_parquet("data/feature_matrix.parquet")
    if "chemistry" in feature_df.columns:
        feature_df = feature_df[feature_df["chemistry"] == "LiNiMnCoO2"].copy()
    
    battery_ids = feature_df["battery_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(feature_df, groups=battery_ids))
    
    train_df = feature_df.iloc[train_idx].reset_index(drop=True)
    val_df = feature_df.iloc[val_idx].reset_index(drop=True)
    
    # Fit StandardScaler on train set, matching the actual Phase 2 training
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])
    
    # Create validation dataset
    val_dataset = BatterySequenceDataset(
        df=val_df, 
        seq_len=30, 
        scaler=scaler, 
        fit_scaler=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print("Loading ONNX Model...")
    session = ort.InferenceSession(
        "onnx/cnn_lstm_nmc.onnx", 
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    
    print("Running Inference...")
    y_true_cycles = []
    y_pred_cycles = []
    
    for x_batch, y_batch in val_loader:
        # x_batch shape: (B, 30, 5), y_batch: (B,)
        ort_inputs = {input_name: x_batch.numpy().astype(np.float32)}
        ort_outs = session.run(None, ort_inputs)
        
        preds = ort_outs[0].flatten()
        y_pred_cycles.extend(preds.tolist())
        y_true_cycles.extend(y_batch.numpy().tolist())
        
    y_true_cycles = np.array(y_true_cycles)
    y_pred_cycles = np.array(y_pred_cycles)
    
    print("Calculating Metrics in Percentage...")
    
    # y_true is in absolute cycles (assuming max ~2000 cycles for NMC).
    # Convert absolute cycles to a percentage [0.0, 100.0].
    y_true_perc = np.clip(y_true_cycles / 20.0, 0.0, 100.0)
    
    # y_pred is output by the Phase 2 model as absolute cycles.
    # Convert absolute cycles to [0.0, 100.0] percentage scale by dividing by 20.0 (max 2000 cycles)
    y_pred_perc = np.clip(y_pred_cycles / 20.0, 0.0, 100.0)
    
    # Variance-Explained Accuracy (R^2 Score)
    r2 = r2_score(y_true_perc, y_pred_perc)
    variance_explained = r2 * 100.0
    
    # Mean Absolute Percentage Error (MAPE)
    # Filter out true RUL < 0.5% to prevent divide-by-zero errors
    valid_idx = y_true_perc >= 0.5
    if np.any(valid_idx):
        mape = np.mean(np.abs((y_true_perc[valid_idx] - y_pred_perc[valid_idx]) / y_true_perc[valid_idx]))
        mape_accuracy = 100.0 * (1.0 - mape)
    else:
        mape_accuracy = 0.0
        
    # Mean Absolute Error (MAE)
    mae_perc = mean_absolute_error(y_true_perc, y_pred_perc)
    
    # Print Report
    table = Table(title="ONNX Model Accuracy Report (Phase 2 - INT8 Quantized)", style="cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")
    
    table.add_row("Variance-Explained (R² Score)", f"{variance_explained:.2f}%")
    table.add_row("MAPE Accuracy (1 - MAPE)", f"{mape_accuracy:.2f}%")
    table.add_row("Mean Absolute Error (MAE)", f"{mae_perc:.2f}% points")
    table.add_row("Total Validation Samples", f"{len(y_true_perc):,}")
    
    rprint(table)

if __name__ == "__main__":
    evaluate_accuracy()
