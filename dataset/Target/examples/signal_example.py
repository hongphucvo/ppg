import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_reader import UnifiedDataReader

def process_dalia_signals():
    # Initialize the reader
    reader = UnifiedDataReader(base_path="dataset")
    
    # 1. List available subjects
    print("1. Finding available subjects...")
    dalia_subjects = reader.list_available_subjects("dalia")
    print(f"Found {len(dalia_subjects)} subjects: {dalia_subjects[:5]}...")
    
    # 2. Process a single subject
    print("\n2. Processing single subject data...")
    subject_id = dalia_subjects[0]
    
    try:
        # Read and process the signal data
        signal_data = reader.read_signal("dalia", subject_id)
        
        # Show basic information
        print("\nData shape:", signal_data.shape)
        print("\nAvailable features:", signal_data.columns.tolist())
        print("\nFirst few rows:")
        print(signal_data.head())
        
        # Show basic statistics
        print("\nBasic statistics:")
        print(signal_data.describe())
        
        # 3. Visualize the signals
        print("\n3. Creating visualizations...")
        
        # Create output directory for plots
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot a segment of PPG and acceleration data
        plt.figure(figsize=(15, 10))
        
        # Plot PPG signal
        plt.subplot(2, 1, 1)
        window = slice(0, 1000)  # Plot first 1000 samples
        plt.plot(signal_data['timestamp'].iloc[window], 
                signal_data['ppg_signal'].iloc[window])
        plt.title(f'PPG Signal - Subject {subject_id}')
        plt.xlabel('Time')
        plt.ylabel('PPG')
        
        # Plot acceleration
        plt.subplot(2, 1, 2)
        for axis in ['acceleration_x', 'acceleration_y', 'acceleration_z']:
            plt.plot(signal_data['timestamp'].iloc[window], 
                    signal_data[axis].iloc[window], 
                    label=axis)
        plt.title('Acceleration')
        plt.xlabel('Time')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'dalia_signals_{subject_id}.png'))
        plt.close()
        
        # 4. Process multiple subjects
        print("\n4. Processing multiple subjects...")
        
        # Initialize containers for statistics
        ppg_stats = []
        
        # Process first 3 subjects
        for subject_id in dalia_subjects[:3]:
            try:
                data = reader.read_signal("dalia", subject_id)
                
                # Calculate statistics
                stats = {
                    'subject_id': subject_id,
                    'ppg_mean': data['ppg_signal'].mean(),
                    'ppg_std': data['ppg_signal'].std(),
                    'hr_mean': data['heart_rate'].mean(),
                    'hr_std': data['heart_rate'].std(),
                    'duration_minutes': (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 60
                }
                
                ppg_stats.append(stats)
                
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(ppg_stats)
        print("\nStatistics across subjects:")
        print(stats_df)
        
        # 5. Save processed data
        print("\n5. Saving processed data...")
        output_dir = "processed_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed signal data
        signal_data.to_csv(os.path.join(output_dir, f"dalia_signals_{subject_id}_processed.csv"), 
                          index=False)
        
        # Save the statistics
        stats_df.to_csv(os.path.join(output_dir, "dalia_signals_statistics.csv"), 
                       index=False)
        
        print(f"Processed data saved to {output_dir}")
        print(f"Plots saved to {plot_dir}")
        
    except Exception as e:
        print(f"Error in processing: {e}")

if __name__ == "__main__":
    process_dalia_signals()
