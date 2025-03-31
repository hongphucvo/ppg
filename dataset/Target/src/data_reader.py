import pandas as pd
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Literal
import numpy as np
from scipy import signal
from .utils import bmi

DataType = Literal['signal', 'demographic']
Demographic = ['subject', 'gender', 'age', 'height', 'weight', 'bmi']
Signal = ['time', 'pleth']

class DataSourceStrategy(ABC):
    """Abstract base class for data source specific strategies."""
    
    def __init__(self, data_type: DataType = 'signal'):
        """Initialize strategy with data type.
        
        Args:
            data_type (DataType): Type of data to process ('signal' or 'demographic')
        """
        self.data_type = data_type
    
    @abstractmethod
    def read_raw_data(self, file_path: str) -> pd.DataFrame:
        """Read raw data from the source."""
        pass
        
    @abstractmethod
    def get_feature_mapping(self) -> Dict[str, str]:
        """Get mapping of standard feature names to source-specific names."""
        pass

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply source-specific preprocessing including feature mapping, unit conversion, and downsampling."""
        # 1. Apply feature mapping
        feature_mapping = self.get_feature_mapping()
        data = data.rename(columns={v: k for k, v in feature_mapping.items()})
        
        # 2. Apply unit conversions
        data = self._convert_units(data)
        
        # 3. Apply downsampling
        data = self._downsample_data(data, target_freq=100)
        
        # 4. Apply any additional source-specific preprocessing
        data = self._additional_preprocessing(data)
        
        return data
    
    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert units to standardized format. Override in subclasses."""
        return data
        
    def _downsample_data(self, data: pd.DataFrame, target_freq: Optional[float] = None) -> pd.DataFrame:
        """Downsample time series data to target frequency.
        
        Args:
            data (pd.DataFrame): Input data with timestamp and PPG columns
            target_freq (float): Target frequency in Hz (e.g., 100 for 100Hz)
            
        Returns:
            pd.DataFrame: Downsampled data
        """
        if 'timestamp' not in data.columns or not target_freq:
            return data
            
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Set timestamp as index
        data = data.set_index('timestamp')
        
        # For PPG column which contains lists, we need special handling
        if 'PPG' in data.columns:
            # Iterate through each row
            for timestamp, row in data.iterrows():
                ppg_values = row['PPG']
                num_samples = len(ppg_values)

                ppg_sampled = []
                if num_samples < target_freq:
                    ppg_sampled = ppg_values
                else:
                    ppg_sampled = np.interp(np.linspace(0, num_samples-1, target_freq), np.arange(num_samples), ppg_values)
                    
                data.loc[timestamp, 'PPG'] = ppg_sampled.tolist()            
        else:
            # For regular numeric columns, use standard resampling
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            resampled = data[numeric_cols].resample(f'{int(1000/target_freq)}ms').mean()
            data = resampled.reset_index()
            
        return data
    
    def _additional_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Additional source-specific preprocessing. Override in subclasses."""
        return data

class DaliaSignalStrategy(DataSourceStrategy):
    """Strategy for processing Dalia signal data (PPG, acceleration, etc.)."""
    
    def __init__(self):
        super().__init__(data_type='signal')
    
    def read_raw_data(self, file_path: str) -> pd.DataFrame:
        """Read Dalia-specific data format from PPG_FieldStudy directory.
        
        Args:
            file_path (str): Base path to the subject directory
                           e.g., 'dataset/Dalia/data/PPG_FieldStudy/subject_01'
        """
        # Extract subject ID from the path
        subject_id = os.path.basename(file_path)
        
        # Construct paths for different signal files
        bvp_path = os.path.join(file_path, 'BVP.csv')
        # ppg_path = os.path.join(file_path, 'PPG.csv')
        # hr_path = os.path.join(file_path, 'HR.csv')
        
        # Read the files
        data_frames = []
        
        if os.path.exists(bvp_path):
            # Read the first two lines to get timestamp and sampling rate
            with open(bvp_path, 'r') as f:
                timestamp = float(f.readline().strip())
                sampling_rate = float(f.readline().strip())
            
            # Read the actual BVP data (skipping the first two lines)
            bvp_data = pd.read_csv(bvp_path, names=['BVP'], skiprows=2, header=None)
            
            # Create timestamp column using timedelta
            timestamps = pd.to_datetime(timestamp, unit='s')
            time_deltas = pd.to_timedelta(np.arange(len(bvp_data)) / sampling_rate, unit='s')
            bvp_data['timestamp'] = timestamps + time_deltas
            
            # Add second and label columns
            bvp_data['second'] = bvp_data['timestamp'].dt.floor('s')
            bvp_data['label'] = 'Not yet implement'
            
            # Group by second to create lists of PPG values
            compressed_data = bvp_data.groupby(['second', 'label']).agg({'BVP': lambda x: list(x)}).reset_index()
            
            # Print information about the data
            print(f"Loaded BVP data with {len(bvp_data)} samples at {sampling_rate}Hz")
            print(f"Compressed into {len(compressed_data)} seconds of data")
            
            data_frames.append(compressed_data)
            
        # if os.path.exists(ppg_path):
        #     ppg_data = pd.read_csv(ppg_path)
        #     ppg_data = ppg_data.rename(columns=self.get_feature_mapping())
        #     data_frames.append(ppg_data)
            
        # if os.path.exists(hr_path):
        #     hr_data = pd.read_csv(hr_path)
        #     hr_data = hr_data.rename(columns=self.get_feature_mapping())
        #     data_frames.append(hr_data)
        
        if not data_frames:
            raise ValueError(f"No data files found for subject {subject_id}")
        
        # Merge all dataframes on second and label
        merged_data = data_frames[0]
        for df in data_frames[1:]:
            merged_data = pd.merge(merged_data, df, on=['second', 'label'], how='outer')
        
        # Sort by second
        merged_data = merged_data.sort_values('second')
        
        # Rename columns according to feature mapping
        merged_data = merged_data.rename(columns=self.get_feature_mapping())
        
        return merged_data

    def get_feature_mapping(self) -> Dict[str, str]:
        return {
            'second': 'timestamp',
            'BVP':'PPG'
            # 'ppg_signal': 'ppg',
            # 'heart_rate': 'hr',
            # 'acceleration_x': 'acc_x',
            # 'acceleration_y': 'acc_y',
            # 'acceleration_z': 'acc_z'
        }
    
    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert Dalia-specific units to standard units."""
        # if 'ppg_signal' in data.columns:
        #     # Convert PPG signal to standard units (if needed)
        #     pass
            
        # if 'heart_rate' in data.columns:
        #     # Ensure heart rate is in BPM
        #     pass
            
        # # Convert acceleration to m/s^2 if needed
        # acc_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z']
        # for col in acc_cols:
        #     if col in data.columns:
        #         # Example: data[col] = data[col] * 9.81  # if units are in g
        #         pass
                
        return data
    
    def _additional_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dalia-specific signal preprocessing steps."""
        # if 'ppg_signal' in data.columns:
        #     # Apply bandpass filter to PPG signal
        #     nyquist = 0.5 * 100  # Assuming 100Hz sampling rate
        #     low = 0.5 / nyquist
        #     high = 8.0 / nyquist
        #     b, a = signal.butter(3, [low, high], btype='band')
        #     data['ppg_signal'] = signal.filtfilt(b, a, data['ppg_signal'])
        
        return data

class DaliaDemographicStrategy(DataSourceStrategy):
    """Strategy for processing Dalia demographic data."""
    
    def __init__(self):
        super().__init__(data_type='demographic')
    
    def read_raw_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
    
    def get_feature_mapping(self) -> Dict[str, str]:
        return {
            'Subject ID': 'subject',
            'Gender': 'gender',
            'Age [years]': 'age',
            'Height [cm]': 'height',
            'Weight [kg]': 'weight'
        }
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Dalia demographic data.
        
        Args:
            data (pd.DataFrame): Raw demographic data
            
        Returns:
            pd.DataFrame: Processed demographic data with standardized features
        """
        # Rename columns based on feature mapping
        mapping = self.get_feature_mapping()
        data = data.rename(columns=mapping)
        
        # Select only the mapped columns
        data = data[list(mapping.values())]
        
        # Convert gender values: 'm' to 'male' and 'f' to 'female'
        data['gender'] = data['gender'].str.lower().map({'m': 'male', 'f': 'female'})
        data['bmi'] = data.apply(lambda row: bmi(row['height'], row['weight']), axis=1)
        return data

    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert Dalia-specific demographic units to standard units."""
        if 'height' in data.columns:
            # Convert height to cm if needed
            pass
            
        if 'weight' in data.columns:
            # Convert weight to kg if needed
            pass
            
        if 'bmi' in data.columns:
            # Recalculate BMI if height and weight are present
            if 'height' in data.columns and 'weight' in data.columns:
                height_m = data['height'] / 100  # convert cm to m
                data['bmi'] = data['weight'] / (height_m ** 2)
                
        return data
    
    def _additional_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dalia-specific demographic preprocessing steps."""
        # Handle missing values
        numeric_cols = ['age', 'height', 'weight', 'bmi']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
        # Standardize gender values
        if 'gender' in data.columns:
            data['gender'] = data['gender'].str.lower()
            data['gender'] = data['gender'].map({'m': 'male', 'f': 'female'})
            
        return data

class PTTDemographicStrategy(DataSourceStrategy):
    """Strategy for processing PTT demographic data."""
    
    def __init__(self):
        super().__init__(data_type='demographic')
    
    def read_raw_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
        
    def get_feature_mapping(self) -> Dict[str, str]:
        # No feature mapping needed, columns are already correctly named
        return {}
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess PTT demographic data.
        
        Args:
            data (pd.DataFrame): Raw demographic data
            
        Returns:
            pd.DataFrame: Processed demographic data with BMI calculated
        """
        # Calculate BMI
        data['bmi'] = data.apply(lambda row: bmi(row['height'], row['weight']), axis=1)
        
        return data
    
    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        # No unit conversion needed for PTT demographics
        return data
    
    def _additional_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """PTT-specific demographic preprocessing steps."""
        # Handle missing values
        if 'age' in data.columns:
            data['age'] = pd.to_numeric(data['age'], errors='coerce')
            
        # Standardize gender values
        if 'gender' in data.columns:
            data['gender'] = data['gender'].str.lower()
            data['gender'] = data['gender'].map({'male': 'm', 'female': 'f'})
            
        return data

class PTTSignalStrategy(DataSourceStrategy):
    """Strategy for processing PTT signal data."""
    
    def __init__(self):
        super().__init__(data_type='signal')
    
    def read_raw_data(self, subject_id: str) -> pd.DataFrame:
        """Read PTT data from csv files.
        
        Args:
            subject_id (str): Subject ID (e.g., 's1', 's2', etc.)
        """
        # Get the base directory using the module's location
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        base_dir = os.path.join(base_dir, 'dataset', 'PTT-PPG', 'data', 'csv')
        
        # List all CSV files for this subject
        activities = ['run', 'sit', 'walk']
        data_frames = []
        
        # Extract the subject number and activity from the subject_id
        # subject_num, activity = subject_id.split('_')
        
        for activity in activities:     
            # Look for the file with the correct format
            file_name = f"{subject_id}_{activity}.csv"
            activity_path = os.path.join(base_dir, file_name)
            
            if os.path.exists(activity_path):
                print(f"Reading file: {activity_path}")
                df = pd.read_csv(activity_path)
                # Convert time column to datetime if it's not already
                if isinstance(df['time'][0], str):
                    df['timestamp'] = pd.to_datetime(df['time'])
                else:
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df['second'] = df['timestamp'].dt.floor('s')
                df['activity'] = activity
                compressed_data = df.groupby(['second', 'activity']).agg({'pleth_1': lambda x: list(x)}).reset_index()
                data_frames.append(compressed_data)
            else:
                print(f"File not found: {activity_path}")
        
        # if not data_frames:
        #     raise ValueError(f"No data files found for subject {subject_id}")
        
        # Concatenate all activities
        if not data_frames:
            raise ValueError(f"No data files found for subject {subject_id}")
            
        combined_data = pd.concat(data_frames, ignore_index=True)
        
        # Apply feature mapping
        combined_data = combined_data.rename(columns=self.get_feature_mapping())
        
        return combined_data
        
    def get_feature_mapping(self) -> Dict[str, str]:
        return {
            'second': 'timestamp',
            'pleth_1': 'PPG',
            # 'ecg_signal': 'ECG',
            # 'ptt': 'PTT',
            'activity': 'label'  
        }
    
    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert PTT-specific units to standard units."""
        # if 'ptt' in data.columns:
        #     # Convert PTT to milliseconds if needed
        #     pass
            
        # if 'ppg_signal' in data.columns:
        #     # Normalize PPG signal if needed
        #     pass
            
        # if 'ecg_signal' in data.columns:
        #     # Normalize ECG signal if needed
        #     pass
            
        return data
    
    def _additional_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """PTT-specific preprocessing steps."""
        # Apply any PTT-specific preprocessing
        # if 'ppg_signal' in data.columns:
        #     # Apply bandpass filter to PPG signal
        #     nyquist = 0.5 * 100  # Assuming 100Hz sampling rate
        #     low = 0.5 / nyquist
        #     high = 8.0 / nyquist
        #     b, a = signal.butter(3, [low, high], btype='band')
        #     data['ppg_signal'] = signal.filtfilt(b, a, data['ppg_signal'])
        
        # if 'ecg_signal' in data.columns:
        #     # Apply bandpass filter to ECG signal
        #     nyquist = 0.5 * 100
        #     low = 0.5 / nyquist
        #     high = 40.0 / nyquist
        #     b, a = signal.butter(3, [low, high], btype='band')
        #     data['ecg_signal'] = signal.filtfilt(b, a, data['ecg_signal'])
        
        return data

class DataSourceFactory:
    """Factory for creating data source strategies."""
    
    _strategies = {
        ('dalia', 'signal'): DaliaSignalStrategy,
        ('dalia', 'demographic'): DaliaDemographicStrategy,
        ('ptt', 'signal'): PTTSignalStrategy,
        ('ptt', 'demographic'): PTTDemographicStrategy,
        # ('ptt', 'signal'): PTTSignalStrategy
    }
    
    @classmethod
    def get_strategy(cls, source_type: str, data_type: DataType = 'signal') -> DataSourceStrategy:
        """Get the appropriate strategy for a data source and type.
        
        Args:
            source_type (str): Type of data source (e.g., 'dalia', 'ptt')
            data_type (DataType): Type of data ('signal' or 'demographic')
            
        Returns:
            DataSourceStrategy: Appropriate strategy for the source and data type
        """
        strategy_class = cls._strategies.get((source_type.lower(), data_type))
        if not strategy_class:
            raise ValueError(f"Unsupported combination of source '{source_type}' and type '{data_type}'")
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, source_type: str, data_type: DataType, strategy_class: type):
        """Register a new data source strategy."""
        cls._strategies[(source_type.lower(), data_type)] = strategy_class

class UnifiedDataReader:
    """Main class for reading and processing data from any source."""
    
    def __init__(self, base_path: str = "dataset"):
        self.base_path = base_path
    
    def read_demographics(self, source_type: str) -> pd.DataFrame:
        """Read and process demographic data for all subjects from a source.
        
        Args:
            source_type (str): Type of data source (e.g., 'dalia', 'ptt-ppg', 'ptt')
            
        Returns:
            pd.DataFrame: Combined demographic data for all subjects
        """
        strategy = DataSourceFactory.get_strategy(source_type, data_type='demographic')
        
        # Get the demographics file path based on source type
        if source_type.lower() == 'dalia':
            demo_file = os.path.join(self.base_path, "Dalia", "data", "demographic.csv")
        elif source_type.lower() == 'ptt':
            demo_file = os.path.join(self.base_path, "PTT-PPG", "data", "demographic.csv")
        else:
            demo_file = os.path.join(self.base_path, source_type.lower(), "data", "demographic.csv")
            
        if not os.path.exists(demo_file):
            raise ValueError(f"Demographics file not found at {demo_file}")
            
        # Read and process demographics
        raw_data = strategy.read_raw_data(demo_file)
        processed_data = strategy.preprocess_data(raw_data)
        
        return processed_data
    
    def read_signal(self, source_type: str, subject_id: str) -> pd.DataFrame:
        """Read and process signal data from the specified source.
        
        Args:
            source_type (str): Type of data source ('dalia' or 'ptt')
            subject_id (str): ID of the subject to process
            
        Returns:
            pd.DataFrame: Processed signal data
            
        Raises:
            ValueError: If source_type is not recognized
        """
        if source_type == 'dalia':
            source_dir = os.path.join(self.base_path, 'Dalia', 'data', 'PPG_FieldStudy', subject_id)
            strategy = DaliaSignalStrategy()
            
            if not os.path.exists(source_dir):
                raise ValueError(f"Signal data directory not found for subject '{subject_id}' at {source_dir}")
                
        elif source_type == 'ptt':
            strategy = PTTSignalStrategy()
            
        else:
            raise ValueError(f"Unknown source type: {source_type}")
            
        # Read and process the signal data
        raw_data = strategy.read_raw_data(subject_id)
        processed_data = strategy.preprocess_data(raw_data)
        
        return processed_data
    
    def list_available_subjects(self, source_type: str) -> List[str]:
        """List available subjects for a given source."""
        if source_type.lower() == 'dalia':
            source_dir = os.path.join(self.base_path, "Dalia", "data", "PPG_FieldStudy")
        elif source_type.lower() == 'ptt':
            source_dir = os.path.join(self.base_path, "PTT-PPG", "data", "csv")
        else:
            source_dir = os.path.join(self.base_path, source_type.lower(), "data")
            
        if not os.path.exists(source_dir):
            return []
            
        # List all directories (subjects) in the source directory
        return [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

class DataDictionary:
    def __init__(self, data_path: str):
        """Initialize DataDictionary with path to data directory."""
        self.data_path = data_path
        self.data_dict = {}
        
    def read_data_dictionary(self, source: str) -> Dict:
        """Read and parse data dictionary for a specific data source."""
        # Implement specific data dictionary reading logic
        # This will be customized based on the format of your data dictionaries
        pass
    
    def get_feature_metadata(self, source: str, feature: str) -> Dict:
        """Get metadata for a specific feature from a data source."""
        return self.data_dict.get(source, {}).get(feature, {})
    
    def list_common_features(self, sources: List[str]) -> List[str]:
        """Find common features across multiple data sources."""
        feature_sets = [set(self.data_dict.get(source, {}).keys()) for source in sources]
        return list(set.intersection(*feature_sets))

class DataLoader:
    def __init__(self, data_path: str):
        """Initialize DataLoader with path to data directory."""
        self.data_path = data_path
        
    def load_dataset(self, source: str, subset: Optional[str] = None) -> pd.DataFrame:
        """Load dataset from a specific source and optional subset."""
        file_path = os.path.join(self.data_path, source)
        if subset:
            file_path = os.path.join(file_path, subset)
            
        # Add specific loading logic based on file format
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format for {file_path}")

class DemographicLoader:
    def __init__(self, data_path: str):
        """Initialize DemographicLoader with path to demographic data directory."""
        self.data_path = data_path
        
    def load_demographics(self, source: str) -> pd.DataFrame:
        """Load demographic data from a specific source."""
        file_path = os.path.join(self.data_path, source)
        
        # Add specific loading logic based on file format
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format for {file_path}")
