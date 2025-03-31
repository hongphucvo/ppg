import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_reader import UnifiedDataReader

def process_demographics():
    # Initialize the reader
    reader = UnifiedDataReader(base_path="dataset")
    
    # 1. Extract demographics from different sources
    print("1. Extracting demographics...")
    
    # Get all demographics at once
    try:
        dalia_demo = reader.read_demographics("dalia")
        print("\nDalia Demographics (all subjects):")
        print(dalia_demo.head())
        print(f"\nTotal subjects in Dalia: {len(dalia_demo)}")
        
        ptt_demo = reader.read_demographics("ptt")
        print("\nPTT Demographics (all subjects):")
        print(ptt_demo.head())
        print(f"\nTotal subjects in PTT: {len(ptt_demo)}")
        
        # 2. Show basic statistics
        print("\n2. Basic statistics for Dalia demographics:")
        print("\nNumerical features:")
        print(dalia_demo.describe())
        
        print("\nGender distribution:")
        print(dalia_demo['gender'].value_counts())
        
        print("\nMissing values:")
        print(dalia_demo.isnull().sum())
        
        # 3. Save processed demographics
        print("\n3. Saving processed demographics...")
        output_dir = "processed_data"
        os.makedirs(output_dir, exist_ok=True)
        
        dalia_demo.to_csv(os.path.join(output_dir, "dalia_demographics_processed.csv"), index=False)
        ptt_demo.to_csv(os.path.join(output_dir, "ptt_demographics_processed.csv"), index=False)
        
        print(f"\nProcessed demographics saved to {output_dir}")
        
    except Exception as e:
        print(f"Error processing demographics: {e}")

if __name__ == "__main__":
    process_demographics()
