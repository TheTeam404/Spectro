# libs_engine/core/engine.py
import logging; log = logging.getLogger(__name__)
import os; import time; from enum import Enum, auto
from typing import Optional, Dict, Any, List, Union, Callable
import numpy as np; import pandas as pd

from ..data_import import loader
from ..processing import peak_detection, peak_fitting, smoothing
from ..analysis import quantification, cf_libs, ml_methods
from ..external import nist_fetcher # Use correct import path

from .data_manager import DataManager
from ._exceptions import ( LIBSEngineError, ConfigurationError, DataLoadingError, DataNotFoundError, ProcessingError, PeakFindingError, PeakFittingError, AnalysisError, DatabaseError )

class EngineState(Enum):
    INITIALIZED = auto(); DATA_LOADED = auto(); SMOOTHED = auto(); PEAKS_FOUND = auto()
    PEAKS_FITTED = auto(); QUANTIFIED = auto(); CF_LIBS_RUN = auto(); ML_APPLIED = auto(); ERROR = auto()

class LIBSAnalysisEngine:
    def __init__(self):
        log.info("Initializing LIBSAnalysisEngine.")
        self._state: EngineState = EngineState.INITIALIZED
        self.data = DataManager()
        self.last_error: Optional[str] = None

    @property
    def state(self) -> EngineState: return self._state

    def _set_state(self, new_state: EngineState):
        if self._state != new_state:
            log.info(f"Engine state: {self._state.name} -> {new_state.name}")
            self._state = new_state
            if new_state == EngineState.ERROR: log.error(f"Engine entered ERROR state. Last error: {self.last_error}")

    def _check_state(self, required: Union[EngineState, List[EngineState]]):
        req_list = required if isinstance(required, list) else [required]
        if self._state not in req_list:
            msg = f"Invalid state: {self._state.name}, requires one of {[s.name for s in req_list]}"
            log.error(msg); self.last_error = msg; self._set_state(EngineState.ERROR); raise LIBSEngineError(msg)

    def _validate_df(self, df: Optional[pd.DataFrame], cols: List[str], name: str) -> bool:
        if df is None: self.last_error=f"{name} is None."; log.error(self.last_error); return False
        if not isinstance(df, pd.DataFrame): self.last_error=f"{name} not DataFrame."; log.error(self.last_error); self._set_state(EngineState.ERROR); raise TypeError(self.last_error)
        missing = [c for c in cols if c not in df.columns];
        if missing: self.last_error=f"{name} missing cols: {missing}. Has: {list(df.columns)}"; log.error(self.last_error); self._set_state(EngineState.ERROR); raise ValueError(self.last_error)
        if df.empty: log.warning(f"{name} is empty.")
        return True

    def _reset_state(self, to_state: EngineState = EngineState.INITIALIZED):
        log.warning(f"Resetting engine state to {to_state.name} and clearing data.")
        self.data.clear_all(); self.last_error = None; self._set_state(to_state)

    def load_data(self, filepath: str, **kwargs: Any) -> bool:
        log.info(f"Loading data from '{filepath}'")
        self._reset_state(EngineState.INITIALIZED)
        try:
            loaded_data = loader.load_spectrum(filepath, **kwargs)
            if loaded_data is None: raise DataLoadingError(f"Loader returned None for '{filepath}'.")
            if not self._validate_df(loaded_data, ['wavelength', 'intensity'], "Loaded data"): raise DataLoadingError("Loaded data failed validation.")
            self.data.set_raw_data(loaded_data, filepath=filepath, fileformat=kwargs.get('file_format', 'unknown'))
            self.data.set_processed_data(self.data.raw_data)
            self.last_error = None; self._set_state(EngineState.DATA_LOADED)
            log.info(f"Loaded {len(self.data.raw_data)} points.")
            return True
        except (FileNotFoundError, DataLoadingError, ValueError, TypeError) as e:
            self.last_error=f"Load failed: {e}"; log.error(self.last_error); self._set_state(EngineState.ERROR)
            raise DataLoadingError(self.last_error) from e if isinstance(e, DataLoadingError) else e
        except Exception as e: self.last_error=f"Load error: {e}"; log.exception(self.last_error); self._set_state(EngineState.ERROR); raise DataLoadingError(self.last_error) from e

    def find_peaks(self, method: str = 'simple', params: Optional[Dict[str, Any]] = None, nist_data: Optional[pd.DataFrame] = None, progress_callback=None) -> bool:
        self._check_state([EngineState.DATA_LOADED, EngineState.SMOOTHED])
        proc_data = self.data.processed_data;
        if not self._validate_df(proc_data, ['wavelength', 'intensity'], "Processed data"): raise DataNotFoundError("Invalid processed data for peak finding.")
        log.info(f"Finding peaks: method='{method}', params={params}")
        self.data.clear_peaks_and_downstream()
        try:
            if progress_callback: progress_callback(0.1)
            found_peaks = peak_detection.find_peaks(proc_data, method, params, nist_data=nist_data) # Pass nist_data here
            if progress_callback: progress_callback(0.9)
            if found_peaks is None: raise PeakFindingError("Peak detector returned None.")
            # Validation happens within find_peaks now, just store
            self.data.set_peaks(found_peaks)
            self.last_error = None; self._set_state(EngineState.PEAKS_FOUND)
            log.info(f"Found {len(self.data.peaks)} peaks.")
            if progress_callback: progress_callback(1.0)
            return True
        except (PeakFindingError, ValueError, TypeError, ConfigurationError, DataNotFoundError) as e:
            self.last_error = f"Peak find failed: {e}"; log.error(self.last_error); self._set_state(EngineState.ERROR); raise PeakFindingError(self.last_error) from e
        except Exception as e: self.last_error=f"Peak find error: {e}"; log.exception(self.last_error); self._set_state(EngineState.ERROR); raise PeakFindingError(self.last_error) from e

    def smooth_data(self, method: str = 'savitzky_golay', params: Optional[Dict[str, Any]] = None) -> bool:
        self._check_state([EngineState.DATA_LOADED, EngineState.SMOOTHED])
        current_proc = self.data.processed_data;
        if not self._validate_df(current_proc, ['wavelength', 'intensity'], "Processed data"): raise DataNotFoundError("Invalid data for smoothing.")
        log.info(f"Smoothing: method='{method}', params={params}")
        self.data.clear_peaks_and_downstream() # Smoothing invalidates peaks
        try:
            smoothed_df = smoothing.smooth_spectrum(current_proc, method, params)
            if smoothed_df is None: raise ProcessingError(f"Smoothing returned None.")
            # Validate result structure
            if not self._validate_df(smoothed_df, ['wavelength', 'intensity'], "Smoothed data"): raise ProcessingError("Smoothed data validation failed.")
            self.data.set_processed_data(smoothed_df)
            self.last_error = None; self._set_state(EngineState.SMOOTHED)
            log.info("Smoothing applied.")
            return True
        except (ProcessingError, ValueError, TypeError, ConfigurationError, DataNotFoundError) as e:
            self.last_error = f"Smooth failed: {e}"; log.error(self.last_error); self._set_state(EngineState.ERROR); raise ProcessingError(self.last_error) from e
        except Exception as e: self.last_error=f"Smooth error: {e}"; log.exception(self.last_error); self._set_state(EngineState.ERROR); raise ProcessingError(self.last_error) from e

    def fit_peaks(self, params: Optional[Dict[str, Any]] = None, progress_callback=None) -> bool:
        self._check_state(EngineState.PEAKS_FOUND)
        peaks_df = self.data.peaks; proc_df = self.data.processed_data
        if not self._validate_df(peaks_df, ['index', 'wavelength'], "Peaks data"): raise DataNotFoundError("Invalid peaks data for fitting.")
        if not self._validate_df(proc_df, ['wavelength', 'intensity'], "Processed data"): raise DataNotFoundError("Invalid processed data for fitting.")
        log.info(f"Fitting peaks: params={params}")
        self.data.clear_fitting_and_downstream()
        try:
             # Progress reporting needs to be integrated inside fit_peaks_detailed potentially
             if progress_callback: progress_callback(0.05)
             fit_results = peak_fitting.fit_peaks_detailed(proc_df, peaks_df, params)
             if progress_callback: progress_callback(0.95)
             if not isinstance(fit_results, dict): raise PeakFittingError(f"Fit module returned non-dict: {type(fit_results).__name__}.")
             self.data.set_peak_fit_results(fit_results)
             self.last_error = None; self._set_state(EngineState.PEAKS_FITTED)
             log.info(f"Fitting complete. Results stored for {len(self.data.peak_fit_results)} peaks.")
             if progress_callback: progress_callback(1.0)
             return True
        except (PeakFittingError, ValueError, TypeError, ConfigurationError, DataNotFoundError) as e:
            self.last_error = f"Fit failed: {e}"; log.error(self.last_error); self._set_state(EngineState.ERROR); raise PeakFittingError(self.last_error) from e
        except Exception as e: self.last_error=f"Fit error: {e}"; log.exception(self.last_error); self._set_state(EngineState.ERROR); raise PeakFittingError(self.last_error) from e

    def fetch_nist_data(self, elements: List[str], **kwargs) -> bool:
        if not elements: log.warning("NIST fetch called with empty element list."); self.data.set_nist_lines(None); return True
        log.info(f"Fetching NIST data for {elements}")
        try:
            fetched_data = nist_fetcher.get_nist_data(elements, **kwargs)
            self.data.set_nist_lines(fetched_data) # Store Astropy Table or None
            if fetched_data is None or len(fetched_data) == 0: log.warning(f"NIST query returned no results for {elements}.")
            else: log.info(f"Fetched {len(self.data.nist_lines)} lines from NIST.")
            self.last_error = None; return True
        except ImportError as e: log.error(f"NIST fetch failed: {e}"); self.last_error=str(e); raise # Re-raise import error
        except DatabaseError as e: self.last_error = f"NIST fetch failed: {e}"; log.error(self.last_error); self.data.set_nist_lines(None); raise # Re-raise db error
        except Exception as e: self.last_error=f"NIST fetch error: {e}"; log.exception(self.last_error); self.data.set_nist_lines(None); raise DatabaseError(self.last_error) from e

    def perform_quantification(self, method: str = 'nist_linear', params: Optional[Dict[str, Any]] = None) -> bool:
        self._check_state(EngineState.PEAKS_FITTED)
        fit_res = self.data.peak_fit_results; nist_ln = self.data.nist_lines
        if fit_res is None: raise DataNotFoundError("Fit results missing for quantification.")
        if 'nist' in method.lower() and nist_ln is None: raise DataNotFoundError(f"NIST data missing for method '{method}'.")
        log.info(f"Quantifying: method='{method}', params={params}")
        self.data.set_quant_results(None)
        try:
             quant_res = quantification.calculate_composition(fit_res, nist_ln, method, params)
             if not isinstance(quant_res, dict): raise AnalysisError(f"Quantification returned non-dict: {type(quant_res).__name__}.")
             self.data.set_quant_results(quant_res)
             self.last_error = None; self._set_state(EngineState.QUANTIFIED)
             log.info(f"Quantification complete. Results: {self.data.quant_results}")
             return True
        except (AnalysisError, DataNotFoundError, ConfigurationError) as e:
             self.last_error=f"Quant failed: {e}"; log.error(self.last_error); self._set_state(EngineState.ERROR); raise AnalysisError(self.last_error) from e
        except Exception as e: self.last_error=f"Quant error: {e}"; log.exception(self.last_error); self._set_state(EngineState.ERROR); raise AnalysisError(self.last_error) from e

    def perform_cf_libs(self, params: Optional[Dict[str, Any]] = None, progress_callback=None) -> bool:
        self._check_state(EngineState.PEAKS_FITTED)
        fit_res = self.data.peak_fit_results; nist_ln = self.data.nist_lines
        if fit_res is None: raise DataNotFoundError("Fit results missing for CF-LIBS.")
        if nist_ln is None: raise DataNotFoundError("NIST data missing for CF-LIBS.")
        log.info(f"Performing CF-LIBS: params={params}")
        self.data.set_cf_libs_results(None)
        try:
             if progress_callback: progress_callback(0.1)
             cf_res = cf_libs.run_cf_libs(fit_res, nist_ln, params) # Progress inside?
             if progress_callback: progress_callback(0.9)
             if not isinstance(cf_res, dict): raise AnalysisError(f"CF-LIBS returned non-dict: {type(cf_res).__name__}.")
             self.data.set_cf_libs_results(cf_res)
             self.last_error = None; self._set_state(EngineState.CF_LIBS_RUN)
             log.info(f"CF-LIBS complete. Results: {self.data.cf_libs_results.get('status')}")
             if progress_callback: progress_callback(1.0)
             return cf_res.get('status') == 'Success' # Return success based on internal status
        except (AnalysisError, DataNotFoundError, ConfigurationError) as e:
             self.last_error=f"CF-LIBS failed: {e}"; log.error(self.last_error); self._set_state(EngineState.ERROR); raise AnalysisError(self.last_error) from e
        except Exception as e: self.last_error=f"CF-LIBS error: {e}"; log.exception(self.last_error); self._set_state(EngineState.ERROR); raise AnalysisError(self.last_error) from e

    def apply_ml(self, model_type: str = 'pca', params: Optional[Dict[str, Any]] = None, data_source: str = 'processed') -> bool:
        req_states = [EngineState.DATA_LOADED, EngineState.SMOOTHED, EngineState.PEAKS_FOUND, EngineState.PEAKS_FITTED, EngineState.QUANTIFIED, EngineState.CF_LIBS_RUN, EngineState.ML_APPLIED]
        if data_source=='peaks': req_states = [EngineState.PEAKS_FOUND, EngineState.PEAKS_FITTED, EngineState.QUANTIFIED, EngineState.CF_LIBS_RUN, EngineState.ML_APPLIED]
        self._check_state(req_states)
        data_map = {'raw': self.data.raw_data, 'processed': self.data.processed_data, 'peaks': self.data.peaks}
        data_to_use = data_map.get(data_source)
        if data_to_use is None: raise DataNotFoundError(f"ML source '{data_source}' not available.")
        log.info(f"Applying ML: model='{model_type}', source='{data_source}', params={params}")
        self.data.set_ml_results(None)
        try:
            ml_res = ml_methods.apply_model(data_to_use, model_type, params)
            # Assuming apply_model returns dict with 'status', 'message', 'output' etc.
            if not isinstance(ml_res, dict) or ml_res.get('status') != 'Success':
                raise AnalysisError(f"ML application '{model_type}' failed: {ml_res.get('message', 'Unknown reason')}")
            self.data.set_ml_results(ml_res) # Store full result dict
            self.last_error = None; self._set_state(EngineState.ML_APPLIED)
            log.info(f"ML model '{model_type}' applied.")
            return True
        except (AnalysisError, DataNotFoundError, ConfigurationError) as e:
             self.last_error=f"ML failed: {e}"; log.error(self.last_error); self._set_state(EngineState.ERROR); raise AnalysisError(self.last_error) from e
        except Exception as e: self.last_error=f"ML error: {e}"; log.exception(self.last_error); self._set_state(EngineState.ERROR); raise AnalysisError(self.last_error) from e

    def run_auto_analysis(self, elements: List[str] = ['Fe', 'Si', 'Al'], progress_callback=None) -> Dict[str, Any]:
        log.info("--- Starting Auto Analysis Sequence ---")
        auto_results = {'status': 'Failed', 'message': '', 'analysis_type': None, 'output': None, 'error_origin': None}
        def _report(step, prog): log.debug(f"Auto Progress: {step} ({prog*100:.0f}%)"); progress_callback(step, prog) if progress_callback else None
        current_step = "Init"
        try:
            self._check_state([EngineState.DATA_LOADED, EngineState.SMOOTHED])
            if self.data.processed_data is None: raise DataNotFoundError("Processed data needed.")
            _report(current_step, 0.0)
            current_step = "Fetch NIST"; _report(current_step, 0.05)
            try: self.fetch_nist_data(elements)
            except (DatabaseError, ImportError) as db_err: log.warning(f"Auto: NIST fetch failed ({db_err}), CF-LIBS may fail.")
            _report(current_step, 0.20)
            current_step = "Detect Peaks"; _report(current_step, 0.25)
            peak_params = None # Simple auto params
            if self.data.processed_data is not None:
                 intensity_std = self.data.processed_data['intensity'].std(); intensity_median = self.data.processed_data['intensity'].median()
                 peak_params = {'prominence': intensity_std * 1.5, 'height': intensity_median + intensity_std}
            self.find_peaks(method='simple', params=peak_params, nist_data=self.data.nist_lines) # Pass NIST data even for simple
            if self.state == EngineState.ERROR: raise LIBSEngineError(self.last_error or "Peak detection failed")
            if not self.data.has_data('peaks'): raise LIBSEngineError("Auto: No peaks detected.")
            _report(current_step, 0.50)
            current_step = "Fit Peaks"; _report(current_step, 0.55)
            fit_params = {'auto_roi': True, 'baseline_method': 'snip', 'selection_criterion': 'aic'}
            self.fit_peaks(params=fit_params)
            if self.state == EngineState.ERROR: raise LIBSEngineError(self.last_error or "Peak fitting failed")
            _report(current_step, 0.85)
            current_step = "CF-LIBS"; _report(current_step, 0.90)
            if self.data.nist_lines is None: raise AnalysisError("Auto: Cannot perform CF-LIBS, NIST data missing.")
            self.perform_cf_libs() # Uses results stored in data manager
            if self.state == EngineState.ERROR: raise LIBSEngineError(self.last_error or "CF-LIBS failed")
            auto_results['status'] = 'Success'; auto_results['message'] = 'Auto analysis (CF-LIBS) complete.';
            auto_results['analysis_type'] = 'CF-LIBS'; auto_results['output'] = self.data.cf_libs_results
            log.info("--- Auto Analysis Finished Successfully ---"); _report("Complete", 1.0)
        except (LIBSEngineError, DataNotFoundError, PeakFindingError, PeakFittingError, AnalysisError, DatabaseError, ValueError, TypeError) as e:
            auto_results['message'] = f"Auto Failed at '{current_step}': {e}"; auto_results['error_origin'] = type(e); log.error(auto_results['message'])
            if self.state != EngineState.ERROR: self._set_state(EngineState.ERROR); _report(current_step + " Failed", 1.0)
        except Exception as e:
            self.last_error = f"Unexpected Auto error at '{current_step}': {e}"; auto_results['message']=self.last_error; auto_results['error_origin'] = type(e)
            log.exception(self.last_error); self._set_state(EngineState.ERROR); _report(current_step + " Failed", 1.0)
        return auto_results