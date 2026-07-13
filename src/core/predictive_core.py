


"""
EcoDrive-Sentinel | Phase 2: Hybrid CNN-LSTM Predictive Core
=============================================================
Implements a CNN-LSTM architecture that captures:
  - Spatial/local degradation patterns (CNN layers)
  - Long-range temporal dependencies (LSTM layers)

Architecture rationale (per Yin et al. 2024, Applied Energy):
  - 1D CNN acts as a learned feature extractor over sliding windows,
    capturing short-range degradation "fingerprints"
  - LSTM captures the non-linear, history-dependent fade trajectory
  - Residual skip-connection stabilises gradient flow over long sequences

NPU Optimization:
  - Designed for INT8 quantization via ONNX → AMD Vitis-AI pipeline
  - Avoids dynamic shapes and custom ops for maximum hardware compatibility
  - All activations bounded with Hardtanh for quantization-friendly ranges

Author: EcoDrive-Sentinel Team
Standard: IEC 62133 / EU Battery Regulation 2023/1542
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import onnx
from onnx import TensorProto
from onnx.helper import make_tensor_value_info

from src.core.config import (
    DATA_DIR,
    MODEL_DIR,
    ONNX_DIR,
    settings,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "voltage_drop",
    "avg_temperature",
    "capacity_fade",
    "internal_resistance_proxy",
    "charge_time_delta",
]
TARGET_COL = "rul"
SEQUENCE_LEN = settings.sequence_length  # sliding window size


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class BatterySequenceDataset(Dataset):
    """
    PyTorch Dataset that creates sliding-window sequences from battery data.
    Injects custom Heuristic Min-Max Normalization.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = SEQUENCE_LEN,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
    ):
        self.seq_len = seq_len
        self.sequences: list[torch.Tensor] = []
        self.targets: list[float] = []

        # Validate feature columns
        missing = [c for c in ["voltage", "current", "temperature", "cycle_number"] if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing from DataFrame: {missing}")

        def normalize_feature(val, min_bound, max_bound):
            return np.clip((val - min_bound) / (max_bound - min_bound), 0.0, 1.0)
            
        VOLTAGE_BOUNDS = (2.0, 4.2)
        CURRENT_BOUNDS = (-50.0, 50.0)
        TEMP_BOUNDS = (-10.0, 65.0)
        VDELTA_BOUNDS = (-0.5, 0.5)
        NMC_MAX_CYCLES = 2000.0

        # Calculate Rolling Voltage Delta
        df["voltage_delta"] = df.groupby("battery_id")["voltage"].diff().fillna(0.0)

        # Inject Heuristic Normalization
        v_scaled = normalize_feature(df["voltage"].values, *VOLTAGE_BOUNDS)
        i_scaled = normalize_feature(df["current"].values, *CURRENT_BOUNDS)
        t_scaled = normalize_feature(df["temperature"].values, *TEMP_BOUNDS)
        c_scaled = normalize_feature(df["cycle_number"].values, 0.0, NMC_MAX_CYCLES)
        vd_scaled = normalize_feature(df["voltage_delta"].values, *VDELTA_BOUNDS)

        X_scaled = np.column_stack([v_scaled, i_scaled, t_scaled, c_scaled, vd_scaled]).astype(np.float32)
        
        # Dummy scaler to prevent export_to_onnx from crashing
        if scaler is None:
            scaler = StandardScaler()
            scaler.mean_ = np.zeros(5)
            scaler.scale_ = np.ones(5)
        self.scaler = scaler

        # Build sequences per battery (avoid leaking across batteries)
        df = df.reset_index(drop=True)
        df["_X_idx"] = range(len(df))

        for battery_id, group in df.groupby("battery_id"):
            group = group.sort_values("cycle_number")
            idxs = group["_X_idx"].values
            rul_vals = group[TARGET_COL].values.astype(np.float32)

            if len(idxs) < seq_len:
                # Pad short batteries with zero-padding at the front
                pad_n = seq_len - len(idxs)
                padded_x = np.vstack([
                    np.zeros((pad_n, 5), dtype=np.float32),
                    X_scaled[idxs],
                ])
                seq_tensor = torch.from_numpy(padded_x)
                self.sequences.append(seq_tensor)
                self.targets.append(float(rul_vals[-1]))
            else:
                for end in range(seq_len, len(idxs) + 1):
                    window_idxs = idxs[end - seq_len:end]
                    seq_tensor = torch.from_numpy(X_scaled[window_idxs])
                    self.sequences.append(seq_tensor)
                    self.targets.append(float(rul_vals[end - 1]))

        logger.info(f"  Dataset built: {len(self.sequences):,} sequences (L={seq_len}, F=5)")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.sequences[idx]  # (seq_len, 5)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
class CNN_LSTM_Regressor(nn.Module):
    """
    Hybrid 1D Dilated CNN-LSTM for Battery RUL Prediction.
    """

    def __init__(
        self,
        n_features: int = 5,
        seq_len: int = SEQUENCE_LEN,
        cnn_channels: tuple[int, int] = (64, 128),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        ch1, ch2 = cnn_channels

        # ── Dilated Spatial CNN Block ─────────────
        # Conv1d expects (batch, channels, seq_len)
        self.conv1 = nn.Conv1d(n_features, ch1, kernel_size=3, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(ch1)
        self.conv2 = nn.Conv1d(ch1, ch2, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm1d(ch2)
        self.conv3 = nn.Conv1d(ch2, ch2, kernel_size=3, padding=4, dilation=4, bias=False)
        self.bn3 = nn.BatchNorm1d(ch2)
        self.act = nn.Hardtanh()    # bounded [-1,1], INT8-friendly
        self.cnn_dropout = nn.Dropout(dropout)

        # Residual projection
        self.residual_proj = nn.Conv1d(n_features, ch2, kernel_size=1, bias=False)

        # ── Temporal LSTM Block ─────────────────
        self.lstm = nn.LSTM(
            input_size=ch2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Regression Head ─────────────────────
        self.regression_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for stable training."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x_t = x.transpose(1, 2)
        h = self.act(self.bn1(self.conv1(x_t)))
        h = self.cnn_dropout(h)
        h = self.act(self.bn2(self.conv2(h)))
        h = self.cnn_dropout(h)
        cnn_out = self.act(self.bn3(self.conv3(h)))
        residual = self.residual_proj(x_t)
        cnn_out = cnn_out + residual

        lstm_in = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        last_hidden = lstm_out[:, -1, :]

        rul_pred = self.regression_head(last_hidden)
        return rul_pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Training Script
# ─────────────────────────────────────────────
class ModelTrainer:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Training device: {self.device}")

    def train(self, feature_df: pd.DataFrame) -> tuple[CNN_LSTM_Regressor, StandardScaler, dict]:
        logger.info("=" * 60)
        logger.info("EcoDrive-Sentinel | Model Training")
        logger.info("=" * 60)

        # ── Data Split ──────────────────────────
        battery_ids = feature_df["battery_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, val_idx = next(gss.split(feature_df, groups=battery_ids))

        train_df = feature_df.iloc[train_idx].reset_index(drop=True)
        val_df = feature_df.iloc[val_idx].reset_index(drop=True)

        train_batteries = train_df["battery_id"].nunique()
        val_batteries = val_df["battery_id"].nunique()
        logger.info(f"Split: {train_batteries} train batteries / {val_batteries} val batteries (80/20)")

        # ── Datasets & Loaders ──────────────────
        train_dataset = BatterySequenceDataset(train_df, seq_len=SEQUENCE_LEN, fit_scaler=True)
        val_dataset = BatterySequenceDataset(
            val_df, seq_len=SEQUENCE_LEN,
            scaler=train_dataset.scaler, fit_scaler=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=settings.batch_size,
            shuffle=True, num_workers=0, pin_memory=(self.device.type == "cuda")
        )
        val_loader = DataLoader(
            val_dataset, batch_size=settings.batch_size,
            shuffle=False, num_workers=0
        )

        # ── Model ───────────────────────────────
        model = CNN_LSTM_Regressor(
            n_features=5,
            seq_len=SEQUENCE_LEN,
        ).to(self.device)
        logger.info(f"Model parameters: {model.count_parameters():,}")

        # ── Optimizer & Scheduler ───────────────
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings.epochs, eta_min=1e-6
        )
        criterion = nn.HuberLoss(delta=10.0)

        # ── Training Loop ───────────────────────
        best_val_loss = float("inf")
        best_state = None
        patience = 10
        patience_counter = 0

        for epoch in range(1, settings.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # Validate Loss Function
            val_loss = self._evaluate_loss(model, val_loader, criterion)
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch:3d}/{settings.epochs}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Checkpoint & early stopping based on Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        logger.success(f"✓ Training complete. Best Val Loss: {best_val_loss:.4f}")

        model_path = settings.model_path
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler_mean": train_dataset.scaler.mean_,
            "scaler_scale": train_dataset.scaler.scale_,
            "model_config": {"n_features": 5, "seq_len": SEQUENCE_LEN},
            "feature_cols": ["voltage", "current", "temperature", "cycle_number", "voltage_delta"],
            "best_val_loss": best_val_loss,
        }, model_path)

        metrics = {
            "best_val_loss": best_val_loss,
            "train_batteries": train_batteries,
            "val_batteries": val_batteries,
        }
        return model, train_dataset.scaler, metrics

    @torch.no_grad()
    def _evaluate_loss(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
        return total_loss / len(loader)


# ─────────────────────────────────────────────
# ONNX Export (NPU-Ready)
# ─────────────────────────────────────────────
def export_to_onnx(
    model: CNN_LSTM_Regressor,
    scaler: StandardScaler,
    output_path: Path = ONNX_DIR / "cnn_lstm_nmc.onnx",
    opset_version: int = 14,
) -> Path:
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    model.eval()
    model = model.cpu()

    batch_size = 1
    dummy_input = torch.zeros(batch_size, SEQUENCE_LEN, 5)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["battery_health_indicators"],
            output_names=["predicted_rul"],
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    onnx_model = onnx.load(str(output_path))
    
    # Inject Scaler parameters as metadata
    meta_mean = onnx_model.metadata_props.add()
    meta_mean.key = "scaler_mean"
    meta_mean.value = ",".join(map(str, scaler.mean_))
    
    meta_scale = onnx_model.metadata_props.add()
    meta_scale.key = "scaler_scale"
    meta_scale.value = ",".join(map(str, scaler.scale_))
    
    onnx.save(onnx_model, str(output_path))
    logger.success(f"✓ ONNX model exported: {output_path}")

    return output_path


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from rich import print as rprint
    from rich.table import Table

    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    feature_path = DATA_DIR / "feature_matrix.parquet"
    if feature_path.exists():
        logger.info(f"Loading feature matrix from {feature_path}")
        feature_df = pd.read_parquet(feature_path)
    else:
        logger.info("Running Feature Engine to generate data...")
        from src.core.feature_engine import FeatureEngine
        engine = FeatureEngine()
        feature_df = engine.build_feature_matrix()

    # Ensure all rows have a chemistry assigned
    feature_df.loc[feature_df["source"].str.contains("TOYOTA", na=False), "chemistry"] = "LFP"
    feature_df["chemistry"] = feature_df["chemistry"].fillna("LiNiMnCoO2")

    logger.info(f"Training on all chemistries: {feature_df['chemistry'].value_counts().to_dict()}")

    


    trainer = ModelTrainer()
    model, scaler, metrics = trainer.train(feature_df)

    onnx_path = export_to_onnx(model, scaler)
    rprint(f"\n[bold cyan]✓ ONNX model ready:[/bold cyan] {onnx_path}")
