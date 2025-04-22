# libs_engine/core/data_manager.py
import logging; log = logging.getLogger(__name__)
from typing import Optional, Dict, Any, List, Union
import pandas as pd; import numpy as np

class DataManager:
    def __init__(self):
        log.debug("Initializing DataManager.")
        self._raw_data: Optional[pd.DataFrame] = None; self._processed_data: Optional[pd.DataFrame] = None
        self._peaks: Optional[pd.DataFrame] = None; self._peak_fit_results: Optional[Dict[int, Dict]] = None # Key by int index
        self._quant_results: Optional[Dict[str, float]] = None; self._cf_libs_results: Optional[Dict[str, Any]] = None
        self._ml_results: Optional[Any] = None; self._nist_lines: Optional[Any] = None
        self._source_filepath: Optional[str] = None; self._source_format: Optional[str] = None

    @property
    def raw_data(self) -> Optional[pd.DataFrame]: return self._raw_data
    @property
    def processed_data(self) -> Optional[pd.DataFrame]: return self._processed_data
    @property
    def peaks(self) -> Optional[pd.DataFrame]: return self._peaks
    @property
    def peak_fit_results(self) -> Optional[Dict[int, Dict]]: return self._peak_fit_results
    @property
    def quant_results(self) -> Optional[Dict[str, float]]: return self._quant_results
    @property
    def cf_libs_results(self) -> Optional[Dict[str, Any]]: return self._cf_libs_results
    @property
    def ml_results(self) -> Optional[Any]: return self._ml_results
    @property
    def nist_lines(self) -> Optional[Any]: return self._nist_lines
    @property
    def source_filepath(self) -> Optional[str]: return self._source_filepath
    @property
    def source_format(self) -> Optional[str]: return self._source_format

    def set_raw_data(self, data: Optional[pd.DataFrame], filepath: Optional[str] = None, fileformat: Optional[str] = None):
        if data is not None and not isinstance(data, pd.DataFrame): raise TypeError("raw_data must be DataFrame or None.")
        self._raw_data = data.copy() if data is not None else None; self._source_filepath = filepath; self._source_format = fileformat
        log.debug(f"Raw data set. Source: {filepath or 'N/A'}")
    def set_processed_data(self, data: Optional[pd.DataFrame]):
        if data is not None and not isinstance(data, pd.DataFrame): raise TypeError("processed_data must be DataFrame or None.")
        self._processed_data = data.copy() if data is not None else None; log.debug("Processed data set.")
    def set_peaks(self, data: Optional[pd.DataFrame]):
        if data is not None and not isinstance(data, pd.DataFrame): raise TypeError("peaks must be DataFrame or None.")
        self._peaks = data.copy() if data is not None else None; log.debug(f"Peaks set ({len(data) if data is not None else 0} peaks).")
    def set_peak_fit_results(self, data: Optional[Dict[int, Dict]]):
        if data is not None and not isinstance(data, dict): raise TypeError("peak_fit_results must be dict or None.")
        # Ensure keys are integers
        self._peak_fit_results = {int(k): v for k,v in data.items()} if data is not None else None
        log.debug(f"Fit results set ({len(data) if data is not None else 0} fits).")
    def set_quant_results(self, data: Optional[Dict[str, float]]):
        if data is not None and not isinstance(data, dict): raise TypeError("quant_results must be dict or None.")
        self._quant_results = data.copy() if data is not None else None; log.debug("Quant results set.")
    def set_cf_libs_results(self, data: Optional[Dict[str, Any]]):
        if data is not None and not isinstance(data, dict): raise TypeError("cf_libs_results must be dict or None.")
        self._cf_libs_results = data.copy() if data is not None else None; log.debug("CF-LIBS results set.")
    def set_ml_results(self, data: Optional[Any]): self._ml_results = data; log.debug("ML results set.")
    def set_nist_lines(self, data: Optional[Any]): self._nist_lines = data; log.debug(f"NIST lines set ({len(data) if data is not None else 0} lines).")

    def clear_all(self):
        log.warning("Clearing all data in DataManager."); self._raw_data = None; self._processed_data = None; self._peaks = None
        self._peak_fit_results = None; self._quant_results = None; self._cf_libs_results = None; self._ml_results = None
        self._nist_lines = None; self._source_filepath = None; self._source_format = None
    def clear_derived_data(self, clear_processed: bool = True):
        log.info("Clearing derived analysis data."); self._processed_data = None if clear_processed else self._processed_data; self._peaks = None; self._peak_fit_results = None
        self._quant_results = None; self._cf_libs_results = None; self._ml_results = None
    def clear_analysis_results(self): log.info("Clearing final analysis results."); self._quant_results = None; self._cf_libs_results = None; self._ml_results = None
    def clear_fitting_and_downstream(self): log.info("Clearing fitting & downstream."); self._peak_fit_results = None; self.clear_analysis_results()
    def clear_peaks_and_downstream(self): log.info("Clearing peaks & downstream."); self._peaks = None; self.clear_fitting_and_downstream()

    def has_data(self, data_type: str) -> bool:
        attr_map = {"raw": self._raw_data, "processed": self._processed_data, "peaks": self._peaks, "fit_results": self._peak_fit_results,
                    "quant": self._quant_results, "cf_libs": self._cf_libs_results, "ml": self._ml_results, "nist": self._nist_lines}
        data = attr_map.get(data_type.lower())
        if isinstance(data, pd.DataFrame): return data is not None and not data.empty
        if isinstance(data, dict): return data is not None and bool(data) # Check if dict not empty
        return data is not None