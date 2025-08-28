# Running Tests for Heart Attack Prediction Project

This folder contains unit tests for the Heart Attack Prediction project. The tests cover:

- Data loading and preprocessing (`data_utils.py`)
- Feature engineering (`feature_engineering.py`)
- Model training and evaluation (`model.py`)
- Visualization functions (`visualize.py`)

## Prerequisites

Make sure the required Python packages are installed:

```bash
pip install -r requirements.txt

## Windows PowerShell  
$env:PYTHONPATH = "D:\Github_work\Heart-Attact-Prediction"
pytest -v