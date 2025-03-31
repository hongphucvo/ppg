import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy import signal

class DataStandardizer:
    def __init__(self):
        """Initialize DataStandardizer with unit conversion mappings."""
        self.unit_conversions = {
            'height': {
                'cm': 1.0,
                'm': 100.0,
                'in': 2.54,
                'ft': 30.48
            },
            'weight': {
                'kg': 1.0,
                'g': 0.001,
                'lbs': 0.453592
            },
            'pressure': {
                'mmHg': 1.0,
                'kPa': 7.50062,
                'bar': 750.062
            }
        }
        
    def standardize_units(self, data: pd.DataFrame, column: str, 
                         from_unit: str, to_unit: str) -> pd.Series:
        """Convert values from one unit to another."""
        if column not in self.unit_conversions:
            raise ValueError(f"No conversion defined for {column}")
            
        conversions = self.unit_conversions[column]
        if from_unit not in conversions or to_unit not in conversions:
            raise ValueError(f"Unknown unit conversion: {from_unit} to {to_unit}")
            
        factor = conversions[from_unit] / conversions[to_unit]
        return data[column] * factor
        
    def resample_signal(self, signal_data: np.ndarray, 
                       original_rate: float, target_rate: float) -> np.ndarray:
        """Resample time series data to a target sampling rate."""
        # Calculate number of samples for target rate
        n_samples = int(len(signal_data) * target_rate / original_rate)
        return signal.resample(signal_data, n_samples)
        
    def normalize_signal(self, signal_data: np.ndarray, 
                        method: str = 'minmax') -> np.ndarray:
        """Normalize signal data using specified method."""
        if method == 'minmax':
            return (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data))
        elif method == 'zscore':
            return (signal_data - np.mean(signal_data)) / np.std(signal_data)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
    def standardize_dataset(self, data: pd.DataFrame, 
                          unit_mappings: Dict[str, Dict[str, str]],
                          sampling_rates: Dict[str, float],
                          target_rate: Optional[float] = None) -> pd.DataFrame:
        """Standardize an entire dataset including units and sampling rates."""
        standardized = data.copy()
        
        # Standardize units
        for column, units in unit_mappings.items():
            if column in standardized:
                standardized[column] = self.standardize_units(
                    standardized, column, units['from'], units['to']
                )
                
        # Resample time series data if needed
        if target_rate:
            for column, rate in sampling_rates.items():
                if column in standardized and rate != target_rate:
                    standardized[column] = self.resample_signal(
                        standardized[column].values, rate, target_rate
                    )
                    
        return standardized
