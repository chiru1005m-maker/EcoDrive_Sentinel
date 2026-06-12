import pandas as pd
import os
from tqdm import tqdm

# 1. Setup Paths
data_folder = 'NASA_PCoE_dataset/data/'
output_file = 'final_training_data.csv'

# 2. Load and Prepare Metadata
metadata = pd.read_csv('NASA_PCoE_dataset/metadata.csv')
metadata['cycle_number'] = metadata.groupby(['battery_id', 'type']).cumcount() + 1

# Filter for discharge cycles only
discharge_list = metadata[metadata['type'] == 'discharge'].copy()
features = []

print(f"Starting Feature Extraction for {len(discharge_list)} cycles...")

# 3. The Feature Engine
for index, row in tqdm(discharge_list.iterrows(), total=discharge_list.shape[0]):
    # The astro__pat dataset has filenames in the 'filename' column
    file_path = os.path.join(data_folder, row['filename'])
    
    if os.path.exists(file_path):
        try:
            # Load the specific sensor log for this cycle
            cycle_df = pd.read_csv(file_path)
            
            # Calculate input features (X)
            avg_v = cycle_df['Voltage_measured'].mean()
            avg_t = cycle_df['Temperature_measured'].mean()
            max_t = cycle_df['Temperature_measured'].max()
            
            # Calculate the Target (y): RUL
            # We find the max cycle for this specific battery ID
            max_c = discharge_list[discharge_list['battery_id'] == row['battery_id']]['cycle_number'].max()
            rul = max_c - row['cycle_number']
            
            features.append({
                'battery_id': row['battery_id'],
                'cycle': row['cycle_number'],
                'avg_voltage': avg_v,
                'avg_temp': avg_t,
                'max_temp': max_t,
                'capacity': row['Capacity'],
                'RUL': rul
            })
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

# 4. Save the processed dataset
if features:
    final_df = pd.DataFrame(features)
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Success! Final training data saved to: {output_file}")
    print(final_df.head())
else:
    print("❌ No features were extracted. Check your 'cleaned_dataset/data/' folder path.")