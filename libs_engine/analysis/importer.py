# libs_engine/analysis/importer.py
# (Use the robust code provided in the previous answer)
# ... includes load_calibration_data, load_ml_model_scaler, load_analysis_config ...
# (Paste the full code here)
import logging; log = logging.getLogger(__name__) # Add logger
import json; import os; from typing import Optional, Dict, Any, Union
import pandas as pd
try: import yaml
except ImportError: yaml = None
try: import joblib
except ImportError: joblib = None
from ..core._exceptions import DataLoadingError, ConfigurationError

def load_calibration_data(filepath: str, file_format: Optional[str] = None) -> Dict[str, Any]:
    # ... (full function code) ...
    log.info(f"Loading calibration data: {filepath}")
    if not os.path.exists(filepath): raise FileNotFoundError(f"Calib file not found: {filepath}")
    if file_format is None: _, ext = os.path.splitext(filepath); ext = ext.lower(); file_format = 'json' if ext == '.json' else 'csv' if ext == '.csv' else None
    if file_format is None: raise ConfigurationError(f"Cannot guess calib format for '{filepath}'. Specify format.")
    file_format = file_format.lower(); calibration_data: Dict[str, Any] = {}
    try:
        if file_format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f: calibration_data = json.load(f)
            if not isinstance(calibration_data, dict): raise DataLoadingError(f"Invalid calib JSON: root not dict '{filepath}'.")
            log.debug("Loaded JSON calibration data.")
        elif file_format == 'csv':
            df = pd.read_csv(filepath, comment='#'); required = ['element', 'wavelength', 'concentration', 'intensity']
            if not all(c in df.columns for c in required): raise DataLoadingError(f"CSV calib missing cols {required} in '{filepath}'.")
            calibration_data = {}
            for (element, wavelength), group in df.groupby(['element', 'wavelength']):
                el_str=str(element); wl_str=f"{wavelength:.2f}"; curve_id=f"{el_str}_{wl_str}"; group = group.sort_values('concentration')
                calibration_data[curve_id] = {'element': el_str, 'wavelength': float(wavelength), 'concentration': group['concentration'].tolist(), 'intensity': group['intensity'].tolist(), 'source_file': filepath, 'source_format': 'csv'}
            log.debug(f"Loaded CSV calibration data ({len(calibration_data)} curves).")
        else: raise ConfigurationError(f"Unsupported calib format: '{file_format}'")
        # Add validation if needed: _validate_calibration_structure(calibration_data)
        return calibration_data
    except json.JSONDecodeError as e: raise DataLoadingError(f"JSON decode error '{filepath}': {e}") from e
    except pd.errors.ParserError as e: raise DataLoadingError(f"CSV parse error '{filepath}': {e}") from e
    except Exception as e: raise DataLoadingError(f"Load/process calib error '{filepath}': {e}") from e

def load_ml_model_scaler(filepath: str) -> Any:
    # ... (full function code) ...
    log.info(f"Loading ML model/scaler: {filepath}")
    if joblib is None: raise ConfigurationError("'joblib' not installed.")
    if not os.path.exists(filepath): raise FileNotFoundError(f"Model/scaler file not found: {filepath}")
    try: loaded_object = joblib.load(filepath); log.debug(f"Loaded object type {type(loaded_object).__name__}"); return loaded_object
    except Exception as e: raise DataLoadingError(f"Failed loading '{filepath}': {e}") from e

def load_analysis_config(filepath: str) -> Dict[str, Any]:
    # ... (full function code) ...
    log.info(f"Loading analysis config: {filepath}")
    if yaml is None: raise ConfigurationError("'PyYAML' not installed.")
    if not os.path.exists(filepath): raise FileNotFoundError(f"Config file not found: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f: config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict): raise DataLoadingError(f"Invalid config: root not dict '{filepath}'.")
        log.debug(f"Loaded YAML config: {filepath}"); return config_data
    except yaml.YAMLError as e: raise DataLoadingError(f"YAML parse error '{filepath}': {e}") from e
    except Exception as e: raise DataLoadingError(f"Load/process config error '{filepath}': {e}") from e