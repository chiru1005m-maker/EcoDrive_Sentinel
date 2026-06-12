"""
scripts/train_universal.py
==========================
EcoDrive-Sentinel: Universal Battery Dataset Training Pipeline

Trains the CNN_LSTM_Regressor on data/processed/universal_battery_master.npy
— the merged Toyota + NASA PCoE master tensor (3,746,452 windows × 30 × 5).

RUL Label Derivation Strategy
------------------------------
The master tensor has no paired label file, so we derive a principled
Remaining Useful Life proxy directly from Channel 4 (Impedance / Capacity Fade):

    capacity_fade_score = mean(window[:, 4])      ← avg ch4 over 30 steps
    rul_label           = 1.0 - capacity_fade_score  ∈ [0.0, 1.0]

    • A window with ch4 ≈ 0.0  → nearly new cell  → RUL ≈ 1.0  (healthy)
    • A window with ch4 ≈ 1.0  → fully degraded   → RUL ≈ 0.0  (EOL)

This normalized RUL is a dimensionless health ratio. For cycle-count RUL,
multiply by max_cycles_per_dataset (e.g. 2000 cycles for NMC cells):
    rul_cycles = rul_label * 2000

Output
------
    models/cnn_lstm_universal.pt          — best-checkpoint weights
    onnx/cnn_lstm_universal.onnx          — ONNX FP32 graph
    onnx/cnn_lstm_universal_quantized.onnx — INT8 quantized (NPU-ready)

Usage
-----
    python scripts/train_universal.py

    Optional env overrides:
        EPOCHS=30 BATCH_SIZE=512 LEARNING_RATE=5e-4 python scripts/train_universal.py
"""

from __future__ import annotations

import os
import sys
import time
import math
import shutil

# ── UTF-8 console output on Windows ──────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8")

# ── Resolve project root so imports work regardless of CWD ───────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from loguru import logger
from src.core.predictive_core import CNN_LSTM_Regressor, FEATURE_COLS, SEQUENCE_LEN
from src.core.config import PROJECT_ROOT as CFG_ROOT, ONNX_DIR, MODEL_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
MASTER_NPY      = os.path.join(PROJECT_ROOT, "data", "processed", "universal_battery_master.npy")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_lstm_universal.pt")
ONNX_FP32_PATH  = os.path.join(PROJECT_ROOT, "onnx",   "cnn_lstm_universal.onnx")
ONNX_INT8_PATH  = os.path.join(PROJECT_ROOT, "onnx",   "cnn_lstm_universal_quantized.onnx")

# ── Hyper-parameters (env-overridable) ────────────────────────────────────────
EPOCHS        = int(os.getenv("EPOCHS",        "50"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE",    "256"))   # 256 better for LSTM convergence
LR            = float(os.getenv("LEARNING_RATE", "1e-3"))
VAL_RATIO     = float(os.getenv("VAL_RATIO",   "0.10"))  # 10 % val split
PATIENCE      = int(os.getenv("PATIENCE",      "8"))     # early-stop patience
N_WORKERS     = 0      # Windows: keep at 0 to avoid multiprocessing issues
PIN_MEMORY    = False  # no GPU

# MAX_SAMPLES: cap the number of windows used per run.
# CPU default 300 000 (~8% of full 3.7M) → ~60-90 s/epoch on a modern CPU.
# Set to 0 or -1 to use the full dataset (use on GPU only).
MAX_SAMPLES   = int(os.getenv("MAX_SAMPLES",   "300000"))

# ── Logger ────────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "train_universal.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger.remove()
logger.add(sys.stdout,  level="INFO",  colorize=False,
           format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add(LOG_PATH,    level="DEBUG", rotation="20 MB", retention="30 days",
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {module}:{line} | {message}")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — memory-mapped to avoid loading 2.25 GB into RAM at once
# ══════════════════════════════════════════════════════════════════════════════
class UniversalBatteryDataset(Dataset):
    """
    Memory-mapped Dataset over universal_battery_master.npy.

    Derives a normalized RUL label from channel 4 (Capacity Fade proxy):
        rul = 1.0 - mean(window[:, 4])

    This avoids loading the full 2.25 GB tensor into RAM — each worker
    reads only the rows it needs via numpy's mmap mechanism.
    """

    def __init__(self, master_path: str, indices: np.ndarray | None = None):
        """
        Args:
            master_path: Path to universal_battery_master.npy  (N, 30, 5)
            indices:     Explicit index subset. Supports MAX_SAMPLES cap and
                         train/val splits. If None, uses all N windows.
        """
        if not os.path.exists(master_path):
            raise FileNotFoundError(
                f"Master tensor not found: {master_path}\n"
                "Run:  python scripts/combine_all_datasets.py  first."
            )

        # Memory-map: file stays on disk, only accessed pages are loaded
        self._data = np.load(master_path, mmap_mode="r")   # (N, 30, 5)
        self._N    = self._data.shape[0]
        self._indices = indices if indices is not None else np.arange(self._N)

        logger.debug(
            f"UniversalBatteryDataset: {len(self._indices):,} samples "
            f"(mmap, total={self._N:,}, shape={self._data.shape})"
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self._indices[idx])

        # Copy from mmap to a writable numpy array (required by PyTorch)
        window = np.array(self._data[real_idx], dtype=np.float32)  # (30, 5)
        np.nan_to_num(window, copy=False, nan=0.0)

        # ── RUL label derivation ──────────────────────────────────────────
        # Channel 4 = normalized capacity / impedance fade proxy (0=fresh, 1=worn)
        # RUL ratio: 1.0 = healthy, 0.0 = end-of-life
        capacity_fade_score = float(window[:, 4].mean())
        rul = max(0.0, min(1.0, 1.0 - capacity_fade_score))

        x = torch.from_numpy(window)                         # (30, 5)
        y = torch.tensor([rul], dtype=torch.float32)         # (1,)
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
# Training Engine
# ══════════════════════════════════════════════════════════════════════════════
def run_training() -> None:
    print("=" * 68)
    print("🔋 ECODRIVE-SENTINEL: UNIVERSAL DATASET TRAINING PIPELINE")
    print("=" * 68)

    # ── Hardware binding ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hardware: {device.type.upper()}")
    logger.info(f"Master tensor: {MASTER_NPY}")

    # ── Dataset & split ───────────────────────────────────────────────────
    logger.info("Mapping master tensor (mmap)...")

    # Determine working index set (full or capped)
    probe = np.load(MASTER_NPY, mmap_mode="r")
    N_total = probe.shape[0]
    del probe

    use_cap = MAX_SAMPLES > 0 and MAX_SAMPLES < N_total
    if use_cap:
        rng = np.random.default_rng(42)
        working_indices = rng.choice(N_total, MAX_SAMPLES, replace=False)
        working_indices.sort()    # sorted → sequential disk access = faster mmap
        logger.info(
            f"MAX_SAMPLES cap active: using {MAX_SAMPLES:,} / {N_total:,} windows "
            f"({MAX_SAMPLES/N_total*100:.1f}% of full dataset)"
        )
    else:
        working_indices = np.arange(N_total)
        logger.info(f"Using full dataset: {N_total:,} windows")

    N = len(working_indices)
    val_size   = max(1, int(N * VAL_RATIO))
    train_size = N - val_size

    # Shuffle and split indices
    rng2 = np.random.default_rng(0)
    shuffled = rng2.permutation(N)
    train_idx = working_indices[shuffled[:train_size]]
    val_idx   = working_indices[shuffled[train_size:]]
    train_idx.sort()   # sorted for mmap locality
    val_idx.sort()

    train_ds = UniversalBatteryDataset(MASTER_NPY, indices=train_idx)
    val_ds   = UniversalBatteryDataset(MASTER_NPY, indices=val_idx)

    steps_per_epoch_est = math.ceil(train_size / BATCH_SIZE)
    # Rough timing estimate: ~3ms/step on modern CPU
    est_epoch_s = steps_per_epoch_est * 0.003 * BATCH_SIZE / 64
    logger.info(
        f"Split → train: {train_size:,}  val: {val_size:,}  "
        f"(batch={BATCH_SIZE}, epochs={EPOCHS})  "
        f"~{est_epoch_s:.0f}s/epoch estimated"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    steps_per_epoch = len(train_loader)
    logger.info(f"Steps per epoch: {steps_per_epoch:,}")

    if use_cap:
        logger.info(
            "  💡 Tip: set MAX_SAMPLES=0 to train on the full 3.7M-window dataset (GPU recommended)"
        )

    # ── Model ─────────────────────────────────────────────────────────────
    model = CNN_LSTM_Regressor(
        n_features=len(FEATURE_COLS),   # 5
        seq_len=SEQUENCE_LEN,            # 30
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"CNN_LSTM_Regressor: {n_params:,} trainable parameters")

    # ── Optimizer, Scheduler, Loss ────────────────────────────────────────
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Cosine anneal over full training run; warms from LR → 0 over T_max epochs
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    # HuberLoss: MSE for small errors, MAE for large errors → robust to RUL outliers
    criterion  = nn.HuberLoss(delta=0.1)  # delta=0.1 for [0,1]-range RUL

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_mae    = float("inf")
    best_state_dict = None
    patience_ctr    = 0
    history: dict[str, list] = {"train_loss": [], "val_mae": [], "lr": []}

    logger.info(f"\n{'─'*68}")
    logger.info(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val MAE':>10}  {'LR':>10}  {'Time':>8}")
    logger.info(f"{'─'*68}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()

        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)                                  # (B, 1)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

            # Progress pulse every ~25% of epoch
            if steps_per_epoch >= 4 and (step + 1) % max(1, steps_per_epoch // 4) == 0:
                pct = (step + 1) / steps_per_epoch * 100
                sys.stdout.write(f"\r  ▶  Epoch {epoch}/{EPOCHS} — {pct:5.1f}% ({step+1}/{steps_per_epoch} steps)  loss={epoch_loss/(step+1):.5f}")
                sys.stdout.flush()

        sys.stdout.write("\r" + " " * 80 + "\r")  # clear progress line

        avg_loss = epoch_loss / steps_per_epoch

        # ── Validate ──
        model.eval()
        val_mae = _evaluate_mae(model, val_loader, device)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(avg_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(lr_now)

        logger.info(
            f"{epoch:>6}  {avg_loss:>12.5f}  {val_mae:>10.5f}  {lr_now:>10.2e}  {elapsed:>6.1f}s"
        )

        # ── Checkpoint ──
        if val_mae < best_val_mae:
            best_val_mae    = val_mae
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr    = 0
            _save_checkpoint(model, epoch, best_val_mae, history)
            logger.info(f"  ✅ New best checkpoint  (Val MAE={best_val_mae:.5f})")
        else:
            patience_ctr += 1
            logger.debug(f"  No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                logger.info(f"  ⏹  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # ── Restore best weights ──────────────────────────────────────────────
    model.load_state_dict(best_state_dict)
    logger.info(f"\n{'='*68}")
    logger.info(f"✅ Training complete  |  Best Val MAE: {best_val_mae:.5f}")
    logger.info(f"   Checkpoint: {MODEL_SAVE_PATH}")
    logger.info(f"{'='*68}\n")

    # ── ONNX export ───────────────────────────────────────────────────────
    _export_onnx(model)

    # ── INT8 quantization ────────────────────────────────────────────────
    _quantize_int8()

    # ── Final summary ─────────────────────────────────────────────────────
    _print_summary(best_val_mae, history)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _evaluate_mae(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute Mean Absolute Error over a DataLoader."""
    model.eval()
    total_err, total_n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).squeeze(1).cpu()
        err  = (pred - yb.squeeze(1)).abs().sum().item()
        total_err += err
        total_n   += len(yb)
    return total_err / max(1, total_n)


def _save_checkpoint(model: nn.Module, epoch: int, val_mae: float, history: dict) -> None:
    """Save a full training checkpoint."""
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(
        {
            "epoch":            epoch,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "best_val_mae":     val_mae,
            "history":          history,
            "model_config": {
                "n_features": len(FEATURE_COLS),
                "seq_len":    SEQUENCE_LEN,
            },
            "feature_cols":     FEATURE_COLS,
            "rul_derivation":   "1.0 - mean(window[:, 4])  (normalized, [0,1])",
            "dataset":          "Toyota + NASA PCoE + CALCE Universal Dataset",
            "training_config": {
                "epochs":      EPOCHS,
                "batch_size":  BATCH_SIZE,
                "lr":          LR,
                "val_ratio":   VAL_RATIO,
                "optimizer":   "AdamW(weight_decay=1e-4)",
                "scheduler":   "CosineAnnealingLR",
                "loss":        "HuberLoss(delta=0.1)",
                "grad_clip":   1.0,
            },
        },
        MODEL_SAVE_PATH,
    )


def _export_onnx(model: nn.Module) -> None:
    """Export trained model to ONNX FP32 (static shape, opset 17)."""
    try:
        import onnx

        logger.info("Exporting to ONNX FP32 (opset 17)...")
        model.eval().cpu()
        dummy = torch.zeros(1, SEQUENCE_LEN, len(FEATURE_COLS))

        os.makedirs(os.path.dirname(ONNX_FP32_PATH), exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                ONNX_FP32_PATH,
                opset_version=17,
                input_names=["battery_health_indicators"],
                output_names=["predicted_rul"],
                dynamic_axes=None,          # static shape → NPU-compatible
                do_constant_folding=True,
                export_params=True,
                verbose=False,
            )

        # Attach EU Battery Passport + NPU metadata
        onnx_model = onnx.load(ONNX_FP32_PATH)
        metadata_kv = {
            "model_name":                 "EcoDrive-Sentinel-Universal-CNN-LSTM",
            "model_version":              "2.0.0",
            "dataset":                    "Toyota + NASA PCoE + CALCE Universal Dataset",
            "rul_unit":                   "normalized [0.0=EOL, 1.0=healthy]",
            "eu_battery_passport_version": "EU-BP-2026-v1",
            "regulation":                 "EU 2023/1542",
            "npu_target":                 "RYZEN_AI_HAWK_POINT",
            "quantization_mode":          "INT8_STATIC",
            "onnx_opset":                 "17",
            "sequence_length":            str(SEQUENCE_LEN),
            "n_features":                 str(len(FEATURE_COLS)),
            "feature_order":              ",".join(FEATURE_COLS),
        }
        for k, v in metadata_kv.items():
            prop = onnx_model.metadata_props.add()
            prop.key, prop.value = k, v

        onnx.save(onnx_model, ONNX_FP32_PATH)
        onnx.checker.check_model(onnx_model)

        size_kb = os.path.getsize(ONNX_FP32_PATH) / 1024
        logger.info(f"  ✅ ONNX FP32 exported & validated: {ONNX_FP32_PATH}  ({size_kb:.0f} KB)")

    except Exception as exc:
        logger.warning(f"  ⚠️  ONNX export failed: {exc}")


def _quantize_int8() -> None:
    """
    INT8 static quantization via onnxruntime.quantization (QDQ format).
    Uses 512 random calibration samples drawn from the master tensor.
    Falls back gracefully if onnxruntime-extensions are unavailable.
    """
    if not os.path.exists(ONNX_FP32_PATH):
        logger.warning("  ⚠️  ONNX FP32 not found — skipping INT8 quantization.")
        return

    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantFormat,
            QuantType,
        )

        logger.info("Quantizing to INT8 (QDQ, static calibration)...")

        # ── Calibration data reader ──
        class _CalibReader(CalibrationDataReader):
            def __init__(self, n_samples: int = 512):
                data = np.load(MASTER_NPY, mmap_mode="r")
                idxs = np.random.default_rng(0).choice(len(data), n_samples, replace=False)
                self._samples = [
                    np.array(data[int(i)], dtype=np.float32)[np.newaxis]  # (1,30,5)
                    for i in idxs
                ]
                self._pos = 0

            def get_next(self):
                if self._pos >= len(self._samples):
                    return None
                sample = {"battery_health_indicators": self._samples[self._pos]}
                self._pos += 1
                return sample

        quantize_static(
            model_input=ONNX_FP32_PATH,
            model_output=ONNX_INT8_PATH,
            calibration_data_reader=_CalibReader(n_samples=512),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )

        size_kb = os.path.getsize(ONNX_INT8_PATH) / 1024
        logger.info(f"  ✅ INT8 quantized model saved: {ONNX_INT8_PATH}  ({size_kb:.0f} KB)")

    except ImportError:
        logger.warning(
            "  ⚠️  onnxruntime.quantization not available. "
            "Install onnxruntime>=1.16 to enable INT8 quantization."
        )
    except Exception as exc:
        logger.warning(f"  ⚠️  INT8 quantization failed: {exc}")


def _print_summary(best_val_mae: float, history: dict) -> None:
    """Print a formatted training summary table."""
    losses = history["train_loss"]
    maes   = history["val_mae"]
    epochs_run = len(losses)

    print("\n" + "=" * 68)
    print("📊 TRAINING SUMMARY")
    print("=" * 68)
    rows = [
        ("Epochs run",           f"{epochs_run}"),
        ("Best Val MAE",         f"{best_val_mae:.5f}  (normalized RUL)"),
        ("Best Val MAE (cycles)",f"{best_val_mae * 2000:.1f}  (×2000 cycle scale)"),
        ("Final Train Loss",     f"{losses[-1]:.5f}"),
        ("Min Train Loss",       f"{min(losses):.5f}  (epoch {losses.index(min(losses))+1})"),
        ("Checkpoint",           MODEL_SAVE_PATH),
        ("ONNX FP32",            ONNX_FP32_PATH if os.path.exists(ONNX_FP32_PATH) else "N/A"),
        ("ONNX INT8 (NPU)",      ONNX_INT8_PATH if os.path.exists(ONNX_INT8_PATH) else "N/A"),
    ]
    for label, value in rows:
        print(f"  {label:<26} {value}")
    print("=" * 68)

    # Verify test pipeline can load the new model
    print("\n📌 Next steps:")
    print("  1. python tests/test_production_pipeline.py   ← re-validate with new model")
    print("  2. Update ONNX_PATH in configs/ to point to cnn_lstm_universal_quantized.onnx")
    print("  3. Run python scripts/eval_ragas.py           ← RAG evaluation on new weights")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Windows spawn safety
    run_training()
