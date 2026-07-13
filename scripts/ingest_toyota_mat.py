import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def ingest_toyota():
    mat_path = Path("data/raw/2018-04-12_batchdata_updated_struct_errorcorrect.mat")
    if not mat_path.exists():
        print(f"File not found: {mat_path}")
        return

    print(f"Loading {mat_path}...")
    
    rows = []
    with h5py.File(mat_path, 'r') as f:
        batch = f['batch']
        num_bat = batch['barcode'].shape[0]
        print(f"Found {num_bat} batteries in batch.")
        
        for b_idx in range(num_bat):
            print(f"Processing battery {b_idx+1}/{num_bat}...")
            # Extract summary data
            summary = f[batch['summary'][b_idx, 0]]
            tavg = np.array(summary['Tavg']).flatten()
            qd = np.array(summary['QDischarge']).flatten()
            
            # Extract cycles data
            cycles = f[batch['cycles'][b_idx, 0]]
            num_cycles = min(len(tavg), cycles['V'].shape[0])
            
            for c_idx in range(num_cycles):
                try:
                    # Get arrays for the cycle
                    v_array = f[cycles['V'][c_idx, 0]][:]
                    i_array = f[cycles['I'][c_idx, 0]][:]
                    
                    v_mean = np.mean(v_array) if len(v_array) > 0 else 3.5
                    i_mean = np.mean(i_array) if len(i_array) > 0 else 1.0
                    
                    row = {
                        "battery_id": f"TOYOTA_2018_04_12_{b_idx:03d}",
                        "cycle_number": c_idx + 1,
                        "voltage": v_mean,
                        "current": i_mean,
                        "temperature": tavg[c_idx],
                        "capacity": qd[c_idx],
                        "source": "TOYOTA_FASTCHARGE"
                    }
                    rows.append(row)
                except Exception as e:
                    # Some cycles might be corrupted
                    pass

    if not rows:
        print("No data extracted.")
        return

    df_toyota = pd.DataFrame(rows)
    print(f"Extracted {len(df_toyota)} cycle rows from Toyota dataset.")
    
    # Append to existing feature matrix
    fm_path = Path("data/feature_matrix.parquet")
    if fm_path.exists():
        df_existing = pd.read_parquet(fm_path)
        print(f"Loaded existing feature matrix with {len(df_existing)} rows.")
        # Ensure we don't duplicate
        df_existing = df_existing[~df_existing['battery_id'].str.startswith('TOYOTA_2018_04_12_')]
        df_combined = pd.concat([df_existing, df_toyota], ignore_index=True)
    else:
        df_combined = df_toyota

    # Calculate RUL for Toyota data (since it's raw, we need to add RUL)
    # RUL is EOL - cycle_number
    # We will do this per battery
    print("Computing RUL labels...")
    rul_frames = []
    for bid, group in df_combined.groupby("battery_id"):
        if "rul" not in group.columns or group["rul"].isna().all():
            # For toyota, we just use max cycle in the group as EOL since they were run to failure
            eol_cycle = group["cycle_number"].max()
            group["rul"] = (eol_cycle - group["cycle_number"]).clip(lower=0).astype(float)
        rul_frames.append(group)
    
    df_combined = pd.concat(rul_frames, ignore_index=True)

    print(f"Saving combined feature matrix with {len(df_combined)} rows...")
    df_combined.to_parquet(fm_path, index=False)
    print("Done!")

if __name__ == "__main__":
    ingest_toyota()
