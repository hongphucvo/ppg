import os
import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime


def bmi(height: float, weight: float) -> float:
    """Calculate BMI (Body Mass Index).
    
    Args:
        height (float): Height in centimeters
        weight (float): Weight in kilograms
        
    Returns:
        float: BMI value
    """
    # Convert height from cm to meters
    height_m = height / 100
    return weight / (height_m ** 2)


def save_dataset(data: pd.DataFrame, filename: str, output_dir: str):
    """Save dataset to specified format and location."""
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, filename)
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.csv':
        data.to_csv(file_path, index=False)
    elif ext == '.parquet':
        data.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)
        
def create_metadata(data: pd.DataFrame, source: str) -> Dict[str, Any]:
    """Create metadata for a dataset."""
    return {
        'source': source,
        'n_rows': len(data),
        'n_columns': len(data.columns),
        'columns': list(data.columns),
        'dtypes': data.dtypes.astype(str).to_dict(),
        'created_at': datetime.now().isoformat(),
        'memory_usage': data.memory_usage(deep=True).sum(),
    }
    
def validate_data(data: pd.DataFrame, schema: Dict) -> bool:
    """Validate dataset against a schema."""
    for column, requirements in schema.items():
        # Check if required column exists
        if requirements.get('required', False) and column not in data.columns:
            return False
            
        if column in data.columns:
            # Check data type
            if 'dtype' in requirements:
                if str(data[column].dtype) != requirements['dtype']:
                    return False
                    
            # Check value range
            if 'range' in requirements:
                min_val, max_val = requirements['range']
                if data[column].min() < min_val or data[column].max() > max_val:
                    return False
                    
    return True
