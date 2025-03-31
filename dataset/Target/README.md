# PPG Dataset Harmonization Project

This project provides tools for harmonizing and analyzing PPG datasets from multiple sources.

## Features
- Data dictionary reading and management
- Cross-database feature mapping
- Demographics EDA and visualization
- Tree graph visualization for data relationships
- Data standardization (units, sampling rates)
- Dataset export functionality

## Project Structure
```
Target/
├── data/               # Data storage
├── src/               # Source code
│   ├── data_reader.py        # Data dictionary and reading utilities
│   ├── feature_mapper.py     # Cross-database feature mapping
│   ├── eda.py               # EDA and visualization tools
│   ├── standardizer.py      # Data standardization utilities
│   └── utils.py            # Helper functions
└── notebooks/         # Jupyter notebooks for analysis
    └── eda.ipynb     # EDA examples and documentation

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your raw datasets in the `data` directory
2. Use the provided modules in `src/` for data processing
3. Check `notebooks/` for usage examples and documentation
