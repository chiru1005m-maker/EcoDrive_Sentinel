import os
import sys
import glob
import numpy as np
import scipy.io
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
TOYOTA_DIR   = os.path.join(PROC_DIR, "toyota_batches")
OUTPUT_PATH  = os.path.join(PROC_DIR, "universal_battery_master.npy")

SEQ_LEN  = 30
N_CH     = 5

print("=" * 65)
print("🔋 ECODRIVE-SENTINEL: UNIVERSAL DATASET ALIGNMENT PIPELINE")
print("=" * 65)

# ── Helpers ───────────────────────────────────────────────────────────────────

def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)

def slice_into_windows(signal_2d: np.ndarray, seq_len: int = SEQ_LEN, stride: int = None) -> np.ndarray:
    if stride is None:
        stride = seq_len
    T = signal_2d.shape[0]
    if T < seq_len:
        return np.empty((0, seq_len, N_CH), dtype=np.float32)
    n_windows = (T - seq_len) // stride + 1
    
    windows = []
    for i in range(n_windows):
        windows.append(signal_2d[i*stride : i*stride + seq_len])
        
    return np.array(windows, dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Toyota pre-processed numpy batches
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Loading Toyota pre-processed batch tensors...")

toyota_arrays = []
batch_files = sorted(glob.glob(os.path.join(TOYOTA_DIR, "batch_*.npy")))

if not batch_files:
    print("  ⚠️  No Toyota batch files found — skipping.")
else:
    for bf in batch_files:
        arr = np.load(bf)
        if arr.ndim == 3 and arr.shape[1] == SEQ_LEN and arr.shape[2] == N_CH:
            toyota_arrays.append(arr)
        else:
            print(f"  ⚠️  Unexpected shape {arr.shape} in {os.path.basename(bf)} — skipping.")

    toyota_tensor = np.concatenate(toyota_arrays, axis=0) if toyota_arrays else np.empty((0, SEQ_LEN, N_CH))
    # Normalize
    for ch in range(N_CH):
        ch_data = toyota_tensor[:, :, ch].flatten()
        lo, hi = ch_data.min(), ch_data.max()
        if hi - lo > 1e-9:
            toyota_tensor[:, :, ch] = ((toyota_tensor[:, :, ch] - lo) / (hi - lo)).astype(np.float32)

    print(f"  ✅ Toyota: {len(batch_files)} batch files -> {toyota_tensor.shape[0]:,} windows")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Parse NASA PCoE (.mat) using scipy.io.loadmat
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Parsing NASA PCoE batch .mat files...")

nasa_windows = []
mat_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.mat")))

for mat_path in mat_files:
    fname = os.path.basename(mat_path)
    # Skip known Toyota files if possible, but let's try to load anyway
    try:
        data = scipy.io.loadmat(mat_path)
        print(f"  Processing: {fname}")
        
        # Look for common key names in NASA .mat files
        v_key = next((k for k in data.keys() if "volt" in k.lower() or k == 'V'), None)
        i_key = next((k for k in data.keys() if "curr" in k.lower() or k == 'I'), None)
        t_key = next((k for k in data.keys() if "temp" in k.lower() or k == 'T'), None)
        
        if v_key and i_key and t_key:
            V = data[v_key].flatten().astype(np.float64)
            I = np.abs(data[i_key].flatten().astype(np.float64))
            T = data[t_key].flatten().astype(np.float64)
            
            # Align lengths
            min_len = min(len(V), len(I), len(T))
            V, I, T = V[:min_len], I[:min_len], T[:min_len]
            
            # SOC Proxy
            soc_key = next((k for k in data.keys() if "soc" in k.lower()), None)
            if soc_key:
                SOC = data[soc_key].flatten().astype(np.float64)[:min_len]
                SOC = np.clip(SOC, 0.0, 1.0)
            else:
                SOC = np.clip((V - 2.7) / (4.2 - 2.7), 0.0, 1.0)
                
            # Health Index Proxy (Capacity or Internal Resistance)
            hi_key = next((k for k in data.keys() if "cap" in k.lower() or "re" in k.lower()), None)
            if hi_key:
                HI = data[hi_key].flatten().astype(np.float64)[:min_len]
                # If capacity, normalize to max. If resistance, just minmax normalize
                if "cap" in hi_key.lower():
                    max_cap = HI.max() if HI.max() > 1e-9 else 1.0
                    HI = np.clip(HI / max_cap, 0.0, 1.0)
                else:
                    HI = minmax_normalize(HI)
            else:
                HI = np.ones_like(V)
                
            series = np.stack([
                minmax_normalize(V),
                minmax_normalize(I),
                minmax_normalize(T),
                SOC.astype(np.float32),
                HI.astype(np.float32)
            ], axis=-1)
            
            windows = slice_into_windows(series)
            if windows.shape[0] > 0:
                nasa_windows.append(windows)
                print(f"  ✅ NASA {fname}: {windows.shape[0]} windows")
    except NotImplementedError:
        print(f"  ⚠️  Falling back to h5py for {fname}...")
        try:
            import h5py
            with h5py.File(mat_path, 'r') as f:
                batch = f["batch"]
                cycles_ds = batch["cycles"]
                n_cells = min(cycles_ds.shape[0], 2) # Sample max 2 cells
                cell_windows_count = 0
                for cell_idx in range(n_cells):
                    cell_ref = cycles_ds[cell_idx, 0]
                    cell_grp = f[cell_ref]
                    n_cycles = min(cell_grp["V"].shape[0], 25) # Sample max 25 cycles
                    all_V, all_I, all_T, all_Qd = [], [], [], []
                    
                    for cyc_idx in range(n_cycles):
                        try:
                            V_ref  = cell_grp["V"][cyc_idx, 0]
                            I_ref  = cell_grp["I"][cyc_idx, 0]
                            T_ref  = cell_grp["T"][cyc_idx, 0]
                            Qd_ref = cell_grp["Qd"][cyc_idx, 0]

                            V  = np.array(f[V_ref]).flatten()
                            I  = np.array(f[I_ref]).flatten()
                            T  = np.array(f[T_ref]).flatten()
                            Qd = np.array(f[Qd_ref]).flatten()

                            min_len = min(len(V), len(I), len(T), len(Qd))
                            if min_len < SEQ_LEN:
                                continue
                            all_V.append(V[:min_len])
                            all_I.append(I[:min_len])
                            all_T.append(T[:min_len])
                            all_Qd.append(Qd[:min_len])
                        except Exception:
                            continue
                            
                    if not all_V:
                        continue
                        
                    V_full  = np.concatenate(all_V)
                    I_full  = np.concatenate(all_I)
                    T_full  = np.concatenate(all_T)
                    Qd_full = np.concatenate(all_Qd)
                    
                    max_cap = Qd_full.max() if Qd_full.max() > 1e-9 else 1.0
                    SOC_full = np.clip(Qd_full / max_cap, 0.0, 1.0)
                    
                    series = np.stack([
                        minmax_normalize(V_full),
                        minmax_normalize(np.abs(I_full)),
                        minmax_normalize(T_full),
                        SOC_full.astype(np.float32),
                        minmax_normalize(Qd_full),
                    ], axis=-1)
                    
                    windows = slice_into_windows(series, stride=2)
                    if windows.shape[0] > 0:
                        nasa_windows.append(windows)
                        cell_windows_count += windows.shape[0]
                print(f"  ✅ NASA {fname} (h5py): {cell_windows_count} windows")
        except Exception as e:
            print(f"  ⚠️  Failed h5py parse for {fname}: {e}")
    except Exception as e:
        print(f"  ⚠️  Failed to parse {fname}: {e}")

nasa_tensor = np.concatenate(nasa_windows, axis=0) if nasa_windows else np.empty((0, SEQ_LEN, N_CH), dtype=np.float32)
print(f"  ✅ NASA PCoE total: {nasa_tensor.shape[0]:,} windows")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Parse CALCE (.mat / .csv)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Scanning for CALCE dataset files...")

calce_windows = []
calce_csv_files = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)

for csv_path in calce_csv_files:
    try:
        df = pd.read_csv(csv_path)
        
        # Clean '[]' and convert everything to numeric
        df = df.replace('[]', np.nan)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='ignore')
                
        v_col  = next((c for c in df.columns if "volt" in c.lower()), None)
        i_col  = next((c for c in df.columns if "curr" in c.lower()), None)
        t_col  = next((c for c in df.columns if "temp" in c.lower()), None)
        
        if not (v_col and t_col):
            continue

        df = df.dropna(subset=[v_col, t_col])

        V  = df[v_col].values.astype(np.float64)
        I  = np.abs(df[i_col].values.astype(np.float64)) if i_col else np.zeros_like(V)
        T  = df[t_col].values.astype(np.float64)
        
        # SOC proxy
        soc_col = next((c for c in df.columns if "soc" in c.lower()), None)
        if soc_col:
            SOC = df[soc_col].values.astype(np.float64)
            SOC = np.clip(SOC, 0.0, 1.0)
        else:
            SOC = np.clip((V - 2.7) / (4.2 - 2.7), 0.0, 1.0)

        # Health Index proxy
        q_col  = next((c for c in df.columns if "capa" in c.lower() or "res" in c.lower()), None)
        if q_col:
            Qd = df[q_col].values.astype(np.float64)
            if "capa" in q_col.lower():
                max_cap = Qd.max() if Qd.max() > 1e-9 else 1.0
                HI = np.clip(Qd / max_cap, 0.0, 1.0)
            else:
                HI = minmax_normalize(Qd)
        else:
            HI = np.ones_like(V)

        series = np.stack([
            minmax_normalize(V),
            minmax_normalize(I),
            minmax_normalize(T),
            SOC.astype(np.float32),
            HI.astype(np.float32),
        ], axis=-1)

        windows = slice_into_windows(series, stride=2)
        if windows.shape[0] > 0:
            calce_windows.append(windows)
            print(f"  ✅ CALCE {os.path.basename(csv_path)}: {windows.shape[0]} windows")
    except Exception as e:
        print(f"  ⚠️  CALCE CSV {os.path.basename(csv_path)} error: {e}")

calce_tensor = np.concatenate(calce_windows, axis=0) if calce_windows else np.empty((0, SEQ_LEN, N_CH), dtype=np.float32)
print(f"  ✅ CALCE total: {calce_tensor.shape[0]:,} windows")

# ══════════════════════════════════════════════════════════════════════════════
# CONCATENATE & SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("📦 Concatenating all sources...")

parts = []
sources = {"Toyota": toyota_tensor, "NASA PCoE": nasa_tensor, "CALCE": calce_tensor}

for name, tensor in sources.items():
    if tensor.shape[0] > 0:
        parts.append(tensor.astype(np.float32))
        print(f"  + {name:12s}: {tensor.shape[0]:>8,} windows")

if not parts:
    print("\n❌ No data assembled — nothing to save.")
    sys.exit(1)

master = np.concatenate(parts, axis=0)
master = np.clip(master, 0.0, 1.0)

os.makedirs(PROC_DIR, exist_ok=True)
np.save(OUTPUT_PATH, master)

size_mb = os.path.getsize(OUTPUT_PATH) / 1e6

print(f"\n{'=' * 65}")
print(f"✅ MASTER TENSOR SAVED")
print(f"   Path  : data/processed/universal_battery_master.npy")
print(f"   Shape : {master.shape}")
print(f"   Size  : {size_mb:.1f} MB")
print(f"{'=' * 65}\n")
