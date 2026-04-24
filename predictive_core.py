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

from config import (
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

    For each battery, cycles [t-L, t-L+1, ..., t] are packed into a
    sequence tensor of shape (L, n_features), and the target is RUL at
    cycle t.

    Args:
        df:          Feature DataFrame from FeatureEngine
        seq_len:     Sliding window length (default: settings.sequence_length)
        scaler:      Pre-fit StandardScaler (pass None to fit on this split)
        fit_scaler:  Whether to fit the scaler on this dataset
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
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing from DataFrame: {missing}")

        # Fit or reuse scaler
        X_raw = df[FEATURE_COLS].values.astype(np.float32)
        if scaler is None:
            scaler = StandardScaler()
        if fit_scaler:
            X_scaled = scaler.fit_transform(X_raw)
        else:
            X_scaled = scaler.transform(X_raw)
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
                    np.zeros((pad_n, len(FEATURE_COLS)), dtype=np.float32),
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

        logger.info(f"  Dataset built: {len(self.sequences):,} sequences (L={seq_len}, F={len(FEATURE_COLS)})")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.sequences[idx]  # (seq_len, n_features)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
class CNN_LSTM_Regressor(nn.Module):
    """
    Hybrid CNN-LSTM for Battery RUL Prediction.

    Architecture:
        Input: (batch, seq_len, n_features)
            ↓
        [Spatial CNN Block]
          Conv1d(in=n_features, out=64, kernel=3) → BN → Hardtanh → Dropout
          Conv1d(in=64, out=128, kernel=3) → BN → Hardtanh
          + Residual projection (skip connection)
            ↓
        [Temporal LSTM Block]
          LSTM(input=128, hidden=256, layers=2, dropout=0.2)
          → last hidden state h_T
            ↓
        [Regression Head]
          Linear(256 → 128) → ReLU → Dropout
          Linear(128 → 1)   → ReLU (RUL ≥ 0)

    Key design choices for NPU compatibility:
        - Conv1d (not Conv2d) maps naturally to NPU vector units
        - Hardtanh replaces ReLU in CNN for bounded activations (INT8-friendly)
        - No attention/softmax operators (poor INT8 fidelity on Hawk Point)
        - Fixed seq_len → static ONNX shape → no dynamic shape overhead

    Args:
        n_features:    Number of input Health Indicators (default: 5)
        seq_len:       Sequence window length
        cnn_channels:  (stage1_ch, stage2_ch) tuple
        lstm_hidden:   LSTM hidden dimension
        lstm_layers:   Number of stacked LSTM layers
        dropout:       Dropout probability
    """

    def __init__(
        self,
        n_features: int = len(FEATURE_COLS),
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

        # ── Spatial CNN Block ───────────────────
        # Conv1d expects (batch, channels, seq_len)
        # We treat each feature as a "channel"
        self.cnn_block = nn.Sequential(
            nn.Conv1d(n_features, ch1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch1),
            nn.Hardtanh(),          # bounded [-1,1], INT8-friendly
            nn.Dropout(dropout),
            nn.Conv1d(ch1, ch2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch2),
            nn.Hardtanh(),
        )

        # Residual projection (match channel dims for skip connection)
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
            nn.ReLU(),  # RUL is always non-negative
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
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)

        Returns:
            Predicted RUL tensor of shape (batch_size, 1)
        """
        # x: (B, T, F)
        # CNN expects (B, F, T) — transpose
        x_t = x.transpose(1, 2)                         # (B, F, T)

        # CNN feature extraction
        cnn_out = self.cnn_block(x_t)                   # (B, ch2, T)
        residual = self.residual_proj(x_t)               # (B, ch2, T)
        cnn_out = cnn_out + residual                     # residual connection

        # Back to (B, T, ch2) for LSTM
        lstm_in = cnn_out.transpose(1, 2)               # (B, T, ch2)

        # LSTM temporal modelling
        lstm_out, _ = self.lstm(lstm_in)                 # (B, T, hidden)
        last_hidden = lstm_out[:, -1, :]                 # (B, hidden) — last time step

        # Regression
        rul_pred = self.regression_head(last_hidden)     # (B, 1)
        return rul_pred

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Training Script
# ─────────────────────────────────────────────
class ModelTrainer:
    """
    Training orchestrator for CNN_LSTM_Regressor.

    Features:
        - Battery-aware GroupShuffleSplit (80/20) — prevents data leakage
        - Cosine annealing LR scheduler
        - Early stopping with patience
        - Checkpointing best model by validation MAE
        - Latency benchmarking for NPU compliance check
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Training device: {self.device}")

    def train(self, feature_df: pd.DataFrame) -> tuple[CNN_LSTM_Regressor, StandardScaler, dict]:
        """
        Full training pipeline.

        Args:
            feature_df: Feature matrix from FeatureEngine.build_feature_matrix()

        Returns:
            Tuple of (trained_model, scaler, metrics_dict)
        """
        logger.info("=" * 60)
        logger.info("EcoDrive-Sentinel | Model Training")
        logger.info("=" * 60)

        # ── Data Split ──────────────────────────
        # GroupShuffleSplit ensures no battery appears in both train and val
        battery_ids = feature_df["battery_id"].values
        unique_batteries = feature_df["battery_id"].unique()

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
            n_features=len(FEATURE_COLS),
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
        criterion = nn.HuberLoss(delta=10.0)  # robust to RUL outliers

        # ── Training Loop ───────────────────────
        best_val_mae = float("inf")
        best_state = None
        patience = 10
        patience_counter = 0
        history = {"train_loss": [], "val_mae": [], "lr": []}

        for epoch in range(1, settings.epochs + 1):
            # Train
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

            # Validate
            val_mae = self._evaluate_mae(model, val_loader)
            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            lr_now = scheduler.get_last_lr()[0]
            history["train_loss"].append(avg_loss)
            history["val_mae"].append(val_mae)
            history["lr"].append(lr_now)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch [{epoch:3d}/{settings.epochs}] "
                    f"Loss: {avg_loss:.4f} | Val MAE: {val_mae:.2f} cycles | LR: {lr_now:.2e}"
                )

            # Checkpoint & early stopping
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        # Restore best weights
        model.load_state_dict(best_state)
        logger.success(f"✓ Training complete. Best Val MAE: {best_val_mae:.2f} cycles")

        # ── Save Model ──────────────────────────
        model_path = settings.model_path
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler_mean": train_dataset.scaler.mean_,
            "scaler_scale": train_dataset.scaler.scale_,
            "model_config": {
                "n_features": len(FEATURE_COLS),
                "seq_len": SEQUENCE_LEN,
            },
            "feature_cols": FEATURE_COLS,
            "best_val_mae": best_val_mae,
        }, model_path)
        logger.success(f"✓ Model checkpoint saved: {model_path}")

        metrics = {
            "best_val_mae_cycles": best_val_mae,
            "train_batteries": train_batteries,
            "val_batteries": val_batteries,
            "total_epochs": epoch,
            "model_params": model.count_parameters(),
        }
        return model, train_dataset.scaler, metrics

    @torch.no_grad()
    def _evaluate_mae(self, model: nn.Module, loader: DataLoader) -> float:
        """Compute Mean Absolute Error on a DataLoader."""
        model.eval()
        all_preds, all_targets = [], []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            preds = model(x_batch).squeeze(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(y_batch.numpy().tolist())
        mae = float(np.mean(np.abs(np.array(all_preds) - np.array(all_targets))))
        return mae


# ─────────────────────────────────────────────
# ONNX Export (NPU-Ready)
# ─────────────────────────────────────────────
def export_to_onnx(
    model: CNN_LSTM_Regressor,
    scaler: StandardScaler,
    output_path: Path = ONNX_DIR / "cnn_lstm.onnx",
    opset_version: int = 17,
) -> Path:
    """
    Export CNN_LSTM_Regressor to ONNX format with NPU quantization metadata.

    Complies with:
        - AMD Vitis-AI 3.x INT8 quantization requirements
        - ONNX opset 17 for Hawk Point NPU support
        - EU Battery Passport traceability metadata

    INT8 Quantization Notes:
        - Static input shape (no dynamic axes) required for Vitis-AI EP
        - Calibration range metadata attached as model properties
        - Avoid gather/scatter ops — use fixed indexing patterns
        - LSTM is pre-traced; weights are folded into static graph

    Args:
        model:        Trained CNN_LSTM_Regressor (CPU weights)
        scaler:       Fitted StandardScaler for preprocessing metadata
        output_path:  ONNX output file path
        opset_version: ONNX opset (17 recommended for Hawk Point)

    Returns:
        Path to exported ONNX file.
    """
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    model.eval()
    model = model.cpu()

    # Static dummy input — required for Vitis-AI (no dynamic shapes)
    batch_size = 1  # edge deployment: single-sample inference
    dummy_input = torch.zeros(batch_size, SEQUENCE_LEN, len(FEATURE_COLS))

    # Benchmark latency before export
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            _ = model(dummy_input)
        latency_ms = (time.perf_counter() - start) / 100 * 1000

    logger.info(f"CPU inference latency (PyTorch): {latency_ms:.2f}ms")
    if latency_ms > settings.max_latency_ms:
        logger.warning(
            f"Latency {latency_ms:.1f}ms exceeds NPU target {settings.max_latency_ms}ms. "
            "Consider reducing model size or enabling INT8 quantization."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["battery_health_indicators"],
            output_names=["predicted_rul"],
            # Static shape — critical for NPU compilation
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    # ── Attach Metadata for Quantization Pipeline ──
    onnx_model = onnx.load(str(output_path))

    metadata = {
        # EU Battery Passport 2026
        "eu_battery_passport_version": "EU-BP-2026-v1",
        "regulation": "EU 2023/1542",
        "rul_unit": "cycles",
        "eol_threshold_capacity": "0.80",

        # NPU / Quantization hints
        "npu_target": settings.npu_target,
        "quantization_mode": "INT8_STATIC",
        "calibration_dataset": "NASA_PCoE_CALCE",
        "max_latency_ms": str(settings.max_latency_ms),
        "onnx_opset": str(opset_version),

        # Preprocessing metadata (for runtime denormalization)
        "scaler_mean": ",".join(f"{v:.6f}" for v in scaler.mean_),
        "scaler_scale": ",".join(f"{v:.6f}" for v in scaler.scale_),
        "feature_order": ",".join(FEATURE_COLS),
        "sequence_length": str(SEQUENCE_LEN),

        # Traceability
        "model_name": "EcoDrive-Sentinel-CNN-LSTM",
        "model_version": "1.0.0",
        "pytorch_latency_ms": f"{latency_ms:.3f}",
    }

    for key, value in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(onnx_model, str(output_path))

    # Validate exported model
    onnx.checker.check_model(onnx_model)
    logger.success(f"✓ ONNX model exported & validated: {output_path}")
    logger.info(f"  Model size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"  Metadata keys: {len(metadata)}")

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

    # Load features
    feature_path = DATA_DIR / "feature_matrix.parquet"
    if feature_path.exists():
        logger.info(f"Loading feature matrix from {feature_path}")
        feature_df = pd.read_parquet(feature_path)
    else:
        logger.info("Running Feature Engine to generate data...")
        from feature_engine import FeatureEngine
        engine = FeatureEngine()
        feature_df = engine.build_feature_matrix()

    # Train
    trainer = ModelTrainer()
    model, scaler, metrics = trainer.train(feature_df)

    # Display results
    table = Table(title="EcoDrive-Sentinel | Training Results", style="green")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="yellow")
    for k, v in metrics.items():
        table.add_row(k.replace("_", " ").title(), str(v) if not isinstance(v, float) else f"{v:.4f}")
    rprint(table)

    # Export ONNX
    onnx_path = export_to_onnx(model, scaler)
    rprint(f"\n[bold cyan]✓ ONNX model ready for Vitis-AI quantization:[/bold cyan] {onnx_path}")
    rprint("\n[bold yellow]Next step:[/bold yellow] Run AMD Vitis-AI quantizer:")
    rprint("  [dim]vai_q_onnx quantize_static --input_model onnx/cnn_lstm.onnx \\[/dim]")
    rprint("  [dim]    --calib_data_reader CalibrationDataReader \\[/dim]")
    rprint("  [dim]    --quant_format QDQ --activation_type QInt8[/dim]")
