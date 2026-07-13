import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from rich import print as rprint

DATA_DIR = Path("data")
MAT_FILE = DATA_DIR / "raw" / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
PARQUET_FILE = DATA_DIR / "feature_matrix.parquet"

def parse_toyota():
    if not MAT_FILE.exists():
        rprint(f"[red]Error: {MAT_FILE} not found.[/red]")
        return
        
    rprint(f"[cyan]Parsing {MAT_FILE.name}...[/cyan]")
    
    rows = []
    with h5py.File(MAT_FILE, 'r') as f:
        batch = f["batch"]
        summary = batch["summary"]
        cycles_ds = batch["cycles"]
        num_cells = summary.shape[0]
        
        rprint(f"[cyan]Found {num_cells} cells. Extracting features...[/cyan]")
        
        for i in range(num_cells):
            # Extract summary data for cell i
            cell_summary_ref = summary[i, 0]
            cell_summary = f[cell_summary_ref]
            
            # Arrays are typically (1, N)
            cycles = np.array(cell_summary["cycle"]).flatten()
            tavg = np.array(cell_summary["Tavg"]).flatten()
            ir = np.array(cell_summary["IR"]).flatten()
            q_discharge = np.array(cell_summary["QDischarge"]).flatten()
            charge_time = np.array(cell_summary["chargetime"]).flatten()
            
            n_cycles = len(cycles)
            max_cycle = cycles.max() if n_cycles > 0 else 1
            
            for j in range(n_cycles):
                # Calculate RUL
                rul = max(0, max_cycle - cycles[j])
                
                rows.append({
                    "battery_id": f"TOYOTA_B1_C{i}",
                    "source": "Toyota_Severson",
                    "cycle_number": float(cycles[j]),
                    "voltage_drop": 0.0, # Not in summary
                    "avg_temperature": float(tavg[j]),
                    "capacity_fade": float(q_discharge[j]),
                    "internal_resistance_proxy": float(ir[j]),
                    "charge_time_delta": float(charge_time[j]),
                    "voltage": 3.3, # Nominal LFP voltage
                    "current": -4.0, # Fast discharge C-rate approx
                    "temperature": float(tavg[j]),
                    "rul": float(rul),
                    "chemistry": "LiFePO4"
                })

    df_toyota = pd.DataFrame(rows)
    rprint(f"[green]Extracted {len(df_toyota)} cycle records.[/green]")
    
    if PARQUET_FILE.exists():
        df_master = pd.read_parquet(PARQUET_FILE)
        
        # Ensure chemistry column exists in master
        if "chemistry" not in df_master.columns:
            df_master["chemistry"] = "LiNiMnCoO2"
            
        rprint(f"[cyan]Appending to existing feature matrix ({len(df_master)} rows)...[/cyan]")
        df_combined = pd.concat([df_master, df_toyota], ignore_index=True)
    else:
        df_combined = df_toyota
        
    df_combined.to_parquet(PARQUET_FILE, index=False)
    rprint(f"[bold green]Successfully saved {len(df_combined)} rows to {PARQUET_FILE.name}![/bold green]")

if __name__ == "__main__":
    parse_toyota()
