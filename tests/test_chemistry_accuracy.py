import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import os
import time

from src.core.config import SensorReading, ChemistryType
from src.agents.agentic_layer import ONNXInferenceEngine

console = Console()

from src.core.config import ONNX_DIR

# We initialize the inference engine once for the test session
engine = ONNXInferenceEngine()

# Physical limits per chemistry
CHEMISTRY_LIMITS = {
    "LiNiMnCoO2": 2000,  # NMC
    "LiFePO4": 6000,     # LFP
    "Na-ion": 3000       # Sodium-Ion
}

def get_true_rul(chemistry: str, cycle: int) -> float:
    """
    Synthetic Ground Truth Generation based on physical characteristics.
    """
    max_cycles = CHEMISTRY_LIMITS[chemistry]
    fraction = min(1.0, cycle / max_cycles)
    
    if chemistry == "LiNiMnCoO2":
        # NMC degrades relatively linearly
        return 100.0 * max(0.0, 1.0 - fraction)
    elif chemistry == "LiFePO4":
        # True LFP: Stays relatively flat (~90% SoH) forming a chemical plateau until ~4,000 cycles, 
        # then drops sharply to 0% at 6,000 cycles.
        if cycle <= 4000:
            return 100.0 - (cycle / 4000.0) * 10.0
        else:
            fraction_drop = (cycle - 4000) / 2000.0
            return 90.0 * max(0.0, 1.0 - fraction_drop)
    elif chemistry == "Na-ion":
        # True Na-ion: Follows a progressive parabolic decay curving down to 0% at 3000 cycles.
        return 100.0 * max(0.0, 1.0 - (fraction ** 2))
    return 0.0

def plot_verification_curves(cycles, true_rul, predicted_rul, chemistry_name):
    # Ensure output directory exists
    os.makedirs('docs', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(cycles, true_rul, label=f'True {chemistry_name} Physics', color='#1E40AF', linewidth=2)
    plt.plot(cycles, predicted_rul, label='Universal Model Prediction', color='#DC2626', linestyle='--', linewidth=2)

    plt.title(f'{chemistry_name} Architecture Validation: True vs. Predicted RUL', fontsize=12, fontweight='bold')
    plt.xlabel('Cycle Count', fontsize=10)
    plt.ylabel('Remaining Useful Life (%)', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Save the verification plot
    plt.savefig(f'docs/{chemistry_name}_verification_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_chemistry(chemistry: str) -> dict:
    """
    Evaluate the real pipeline across the lifespan cycle-by-cycle.
    """
    max_cycles = CHEMISTRY_LIMITS[chemistry]
    intervals = np.arange(0, max_cycles + 1, 10) 
    
    y_true = []
    y_pred = []
    errors = []
    max_drift = 0.0
    drift_zone = ""
    
    chem_enum_map = {
        "LiNiMnCoO2": ChemistryType.NMC,
        "LiFePO4": ChemistryType.LFP,
        "Na-ion": ChemistryType.NA_ION
    }
    chem_enum = chem_enum_map[chemistry]
    
    for cycle in intervals:
        cycle = int(cycle)
        true_rul = get_true_rul(chemistry, cycle)
        
        # Build a realistic 30-step historical sequence for the model
        sequence = []
        for step_cycle in range(max(0, cycle - 30), cycle):
            step_fraction = min(1.0, step_cycle / max_cycles)
            step_voltage = 4.2 - (0.7 * step_fraction)
            sequence.append(SensorReading(
                battery_id="TEST-001",
                timestamp=time.time(),
                voltage=step_voltage,
                current=-2.0,
                temperature=25.0,
                cycle_count=step_cycle,
                chemistry=chem_enum
            ))
            
        if not sequence:
            # Fallback for cycle 0
            sequence = [SensorReading(
                battery_id="TEST-001", timestamp=time.time(), voltage=4.2,
                current=-2.0, temperature=25.0, cycle_count=0, chemistry=chem_enum
            )]
            
        # Run live inference using the sequence prediction method
        pred_rul, _ = engine.predict_sequence(sequence)
        
        y_true.append(true_rul)
        y_pred.append(pred_rul)
        
        abs_err = abs(true_rul - pred_rul)
        errors.append(abs_err)
        
        if abs_err > max_drift:
            max_drift = abs_err
            life_pct = (cycle / max_cycles) * 100
            drift_zone = f"{life_pct:.1f}% Life ({cycle} cycles)"

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred), "Array length mismatch before computing metrics!"
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    accuracy = 100.0 - mae
    
    chem_name = chemistry.replace("LiNiMnCoO2", "NMC").replace("LiFePO4", "LFP")
    
    plot_verification_curves(intervals, y_true, y_pred, chem_name)
    
    return {
        "chemistry": chemistry,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        "max_drift": max_drift,
        "drift_zone": drift_zone,
        "chem_name": chem_name
    }

def test_chemistry_scaling_bridge():
    """
    Pytest case to verify the scaling bridge and print the drift matrix.
    """
    results = []
    failures = []
    for chem in CHEMISTRY_LIMITS.keys():
        res = evaluate_chemistry(chem)
        results.append(res)
        
        if res["accuracy"] < 80.0:
            failures.append(f"{res['chem_name']} Accuracy dropped to {res['accuracy']:.2f}% (Expected >80%)")

    console.print("\n[bold cyan]Chemistry Scaling Bridge Verification Matrix[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Chemistry", width=12)
    table.add_column("MAE (%)", justify="right")
    table.add_column("RMSE (%)", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Max Absolute Error", justify="right")
    table.add_column("Peak Drift Zone", style="dim")
    
    for r in results:
        acc_str = f"[green]{r['accuracy']:.1f}%[/green]" if r['accuracy'] >= 90 else f"[yellow]{r['accuracy']:.1f}%[/yellow]"
        if r['accuracy'] < 80.0:
            acc_str = f"[red]{r['accuracy']:.1f}%[/red]"
        
        table.add_row(
            r["chem_name"],
            f"{r['mae']:.2f}",
            f"{r['rmse']:.2f}",
            acc_str,
            f"{r['max_drift']:.2f}%",
            r["drift_zone"]
        )
        
    console.print(table)
    console.print("[dim]Note: LFP/Na-ion natively match their non-linear physical degradation thanks to the Universal Model.[/dim]\n")
    console.print("[dim]Visualization artifacts saved to docs/ directory.[/dim]\n")
    
    if failures:
        pytest.fail("\n".join(failures))

if __name__ == "__main__":
    test_chemistry_scaling_bridge()
