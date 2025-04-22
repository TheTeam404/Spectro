# libs_engine/analysis/quantification.py
import logging; log = logging.getLogger(__name__)
from typing import Optional, Dict, Any, Union
import pandas as pd; import numpy as np
from ..core._exceptions import AnalysisError, DataNotFoundError, ConfigurationError

def calculate_composition(peak_fit_results: Dict[Any, Dict], nist_lines: Optional[Any]=None, method: str='nist_linear', params: Optional[Dict[str, Any]]=None) -> Dict[str, float]:
    if not peak_fit_results: raise DataNotFoundError("Peak fitting results required.")
    params = params or {}; log.info(f"Quantifying using method '{method}'.")
    if method == 'nist_linear':
        log.warning("Using placeholder 'nist_linear' quantification. Results likely inaccurate.")
        if nist_lines is None: raise DataNotFoundError("NIST data required for 'nist_linear'.")
        elements_of_interest = params.get('elements_of_interest')
        if not elements_of_interest:
             elements_in_peaks = set(p.get('element') for p in peak_fit_results.values() if p.get('element'))
             if not elements_in_peaks: raise ConfigurationError("Need 'elements_of_interest' param or identified elements in fits.")
             elements_of_interest = list(elements_in_peaks); log.info(f"Using inferred elements: {elements_of_interest}")
        element_signal: Dict[str, float] = {el: 0.0 for el in elements_of_interest}; peaks_used_count = 0
        for peak_id, fit_info in peak_fit_results.items():
            element = fit_info.get('element'); amplitude = fit_info.get('amplitude') # Or intensity_area
            if element and element in elements_of_interest and amplitude is not None:
                if fit_info.get('r_squared', 0) > params.get('min_fit_quality', 0.9): element_signal[element] += amplitude; peaks_used_count += 1
                else: log.debug(f"Skipping peak {peak_id} ({element}) due to low fit quality.")
        if peaks_used_count == 0: log.warning("No valid peaks found for quantification."); return {el: 0.0 for el in elements_of_interest}
        total_signal = sum(element_signal.values());
        if total_signal <= 0: log.warning("Total signal <= 0."); return {el: 0.0 for el in elements_of_interest}
        concentrations = {el: (signal / total_signal) * 100.0 for el, signal in element_signal.items()}
        log.info(f"'nist_linear' complete. Used {peaks_used_count} peaks.")
        return concentrations
    elif method == 'calibration_curve': raise NotImplementedError("'calibration_curve' not implemented.")
    else: raise ConfigurationError(f"Unknown quantification method: '{method}'")
    return {} # Should not be reached