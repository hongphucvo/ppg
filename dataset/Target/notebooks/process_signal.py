import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from src.data_reader import UnifiedDataReader
import pandas as pd

def process_signal_data():
    # Initialize the data readers
    reader = UnifiedDataReader()
    
    # Define output directories
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "processed_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # # Process Dalia data
    # print("\nProcessing Dalia data...")
    # dalia_output_dir = os.path.join(output_dir, "dalia")
    # os.makedirs(dalia_output_dir, exist_ok=True)
    
    # # Get list of Dalia subjects
    # dalia_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Dalia", "data", "PPG_FieldStudy")
    # print(f"Looking for Dalia data in: {dalia_dir}")
    
    # if os.path.exists(dalia_dir):
    #     subjects = [d for d in os.listdir(dalia_dir) if os.path.isdir(os.path.join(dalia_dir, d))]
        
    #     for subject in subjects:
    #         try:
    #             print(f"\nProcessing Dalia subject: {subject}")
    #             data = reader.read_signal("dalia", subject)
                
    #             # Save processed data
    #             output_path = os.path.join(dalia_output_dir, f"{subject}_processed.parquet")
    #             data.to_parquet(output_path)
    #             print(f"Saved processed data to: {output_path}")
                
    #         except Exception as e:
    #             print(f"Error processing Dalia subject {subject}: {str(e)}")
    # else:
    #     print(f"Dalia directory not found: {dalia_dir}")
    
    # Process PTT data
    # print("\nProcessing PTT data...")
    ptt_output_dir = os.path.join(output_dir, "ptt")
    # os.makedirs(ptt_output_dir, exist_ok=True)
    
    # Get demographics dataframe
    # demographics_df = reader.read_demographics("ptt")
    # print(f"Found {len(demographics_df)} subjects in demographics")
    
    # Get unique subject IDs from demographics
    subjects = [f's{i}' for i in range(23)]
    
    for subject in subjects:
        try:
            print(f"\nProcessing PTT subject: {subject}")
            data = reader.read_signal("ptt", subject)
            
            # Save processed data
            output_path = os.path.join(ptt_output_dir, f"{subject}_processed.parquet")
            data.to_parquet(output_path)
            print(f"Saved processed data to: {output_path}")
            
        except Exception as e:
            print(f"Error processing PTT subject {subject}: {str(e)}")

if __name__ == "__main__":
    process_signal_data()
