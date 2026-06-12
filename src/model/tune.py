import sys
import copy
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.model_selection import ParameterGrid, GroupShuffleSplit
from torch.utils.data import DataLoader

from src.core.config import DATA_DIR, ONNX_DIR, settings
from src.core.predictive_core import (
    BatterySequenceDataset,
    CNN_LSTM_Regressor,
    SEQUENCE_LEN,
    export_to_onnx
)

logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

def tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Tuning device: {device}")

    # ── Load Data ──────────────────────────────
    feature_path = DATA_DIR / "feature_matrix.parquet"
    if not feature_path.exists():
        logger.error(f"Feature matrix not found at {feature_path}")
        return
    
    feature_df = pd.read_parquet(feature_path)
    
    if "chemistry" not in feature_df.columns:
        feature_df["chemistry"] = "NMC"
    feature_df = feature_df[feature_df["chemistry"] == "NMC"]

    # ── Data Split ──────────────────────────
    battery_ids = feature_df["battery_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(feature_df, groups=battery_ids))

    train_df = feature_df.iloc[train_idx].reset_index(drop=True)
    val_df = feature_df.iloc[val_idx].reset_index(drop=True)

    # ── Grid Search Space ────────────────────
    param_grid = {
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "lstm_hidden": [64, 128, 256],
        "dropout": [0.1, 0.2, 0.4],
        "batch_size": [32, 64]
    }
    grid = list(ParameterGrid(param_grid))
    logger.info(f"Starting Grid Search: {len(grid)} total combinations.")

    best_val_loss = float("inf")
    best_params = None
    best_model_state = None
    best_scaler = None
    best_train_batteries = train_df["battery_id"].nunique()
    best_val_batteries = val_df["battery_id"].nunique()

    # ── Execute Sweep ─────────────────────────
    for idx, params in enumerate(grid, 1):
        logger.info("-" * 60)
        logger.info(f"Run {idx}/{len(grid)} | Params: {params}")
        
        train_dataset = BatterySequenceDataset(train_df, seq_len=SEQUENCE_LEN, fit_scaler=True)
        val_dataset = BatterySequenceDataset(val_df, seq_len=SEQUENCE_LEN, scaler=train_dataset.scaler, fit_scaler=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=params["batch_size"],
            shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
        )
        val_loader = DataLoader(
            val_dataset, batch_size=params["batch_size"],
            shuffle=False, num_workers=0
        )

        model = CNN_LSTM_Regressor(
            n_features=5,
            seq_len=SEQUENCE_LEN,
            lstm_hidden=params["lstm_hidden"],
            dropout=params["dropout"]
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=1e-4)
        
        # Use ReduceLROnPlateau as requested
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss() # Watch Validation MSE Loss

        patience = 20
        patience_counter = 0
        min_val_loss_run = float("inf")

        for epoch in range(1, 301): # Upper limit to 300 epochs
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device).unsqueeze(1)
                    preds = model(x_batch)
                    val_loss += criterion(preds, y_batch).item()
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)

            if val_loss < min_val_loss_run:
                min_val_loss_run = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}. Best Val MSE: {min_val_loss_run:.4f}")
                break

        # Check against global best
        if min_val_loss_run < best_val_loss:
            logger.success(f"New Best Params Found! Val MSE: {min_val_loss_run:.4f}")
            best_val_loss = min_val_loss_run
            best_params = params
            best_model_state = copy.deepcopy({k: v.cpu().clone() for k, v in model.state_dict().items()})
            best_scaler = train_dataset.scaler

    logger.info("=" * 60)
    logger.success(f"Grid Search Complete! Best Val MSE: {best_val_loss:.4f}")
    logger.info(f"Best Hyperparameters: {best_params}")

    # ── Export Optimized Model ────────────────
    best_model = CNN_LSTM_Regressor(
        n_features=5,
        seq_len=SEQUENCE_LEN,
        lstm_hidden=best_params["lstm_hidden"],
        dropout=best_params["dropout"]
    ).to(device)
    
    best_model.load_state_dict(best_model_state)

    optimized_onnx_path = ONNX_DIR / "cnn_lstm_nmc_optimized.onnx"
    export_to_onnx(best_model, best_scaler, output_path=optimized_onnx_path, opset_version=14)
    logger.success(f"Optimized ONNX model successfully saved to: {optimized_onnx_path}")

if __name__ == "__main__":
    tune()
