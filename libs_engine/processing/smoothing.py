# libs_engine/processing/smoothing.py
# (Use the robust code provided in the previous answer)
# ... includes _smooth_savitzky_golay, _smooth_moving_average, _smooth_gaussian_filter, SMOOTHING_FUNCTIONS, smooth_spectrum ...
# (Paste the full code here)
import logging; log = logging.getLogger(__name__) # Add logger
from typing import Optional, Dict, Any, Callable, Union
import numpy as np; import pandas as pd
try: from scipy.signal import savgol_filter
except ImportError: savgol_filter = None
try: from scipy.ndimage import gaussian_filter1d
except ImportError: gaussian_filter1d = None
from ..core._exceptions import ProcessingError, ConfigurationError, DataNotFoundError

def _smooth_savitzky_golay(y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    if savgol_filter is None: raise ConfigurationError("SciPy needed for Savitzky-Golay.")
    window_length = params.get('window_length'); polyorder = params.get('polyorder')
    if window_length is None or polyorder is None: raise ConfigurationError("SavGol requires 'window_length' and 'polyorder'.")
    if not isinstance(window_length, int) or not isinstance(polyorder, int): raise ConfigurationError("SavGol params must be integers.")
    if window_length <= 0 or polyorder < 0: raise ConfigurationError("Invalid SavGol window/order values.")
    if window_length % 2 == 0: raise ConfigurationError(f"SavGol 'window_length' ({window_length}) must be odd.")
    if polyorder >= window_length: raise ConfigurationError(f"SavGol 'polyorder' ({polyorder}) must be < 'window_length' ({window_length}).")
    n_points = len(y);
    if window_length > n_points: raise ConfigurationError(f"SavGol 'window_length' ({window_length}) > data points ({n_points}).")
    deriv = params.get('deriv', 0); delta = params.get('delta', 1.0); mode = params.get('mode', 'interp'); cval = params.get('cval', 0.0)
    log.debug(f"Applying SavGol: window={window_length}, order={polyorder}, deriv={deriv}, mode='{mode}'")
    try: return savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode, cval=cval)
    except Exception as e: raise ProcessingError(f"SavGol filter error: {e}") from e

def _smooth_moving_average(y: Union[np.ndarray, pd.Series], params: Dict[str, Any]) -> np.ndarray:
    window_size = params.get('window_size'); min_periods = params.get('min_periods', 1); center = params.get('center', True)
    if window_size is None: raise ConfigurationError("Moving average requires 'window_size'.")
    if not isinstance(window_size, int) or window_size <= 0: raise ConfigurationError("'window_size' must be positive integer.")
    n_points = len(y);
    if window_size > n_points: log.warning(f"Moving average 'window_size' ({window_size}) > data points ({n_points})."); window_size = min(window_size, n_points)
    y_series = pd.Series(y) if isinstance(y, np.ndarray) else y
    log.debug(f"Applying Moving Average: window={window_size}, center={center}, min_periods={min_periods}")
    try:
        smoothed_series = y_series.rolling(window=window_size, min_periods=min_periods, center=center).mean()
        smoothed_series = smoothed_series.fillna(method='bfill').fillna(method='ffill') # Handle edge NaNs
        return smoothed_series.values
    except Exception as e: raise ProcessingError(f"Moving average error: {e}") from e

def _smooth_gaussian_filter(y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    if gaussian_filter1d is None: raise ConfigurationError("SciPy needed for Gaussian filter.")
    sigma = params.get('sigma'); order = params.get('order', 0); mode = params.get('mode', 'reflect'); truncate = params.get('truncate', 4.0)
    if sigma is None: raise ConfigurationError("Gaussian filter requires 'sigma'.")
    if not isinstance(sigma, (int, float)) or sigma <= 0: raise ConfigurationError("'sigma' must be positive.")
    log.debug(f"Applying Gaussian Filter: sigma={sigma}, order={order}, mode='{mode}', truncate={truncate}")
    try: return gaussian_filter1d(y, sigma=sigma, order=order, mode=mode, truncate=truncate)
    except Exception as e: raise ProcessingError(f"Gaussian filter error: {e}") from e

SMOOTHING_FUNCTIONS: Dict[str, Callable] = {"savitzky_golay": _smooth_savitzky_golay, "savgol": _smooth_savitzky_golay, "moving_average": _smooth_moving_average, "boxcar": _smooth_moving_average, "gaussian": _smooth_gaussian_filter}

def smooth_spectrum(data: pd.DataFrame, method: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if data is None or not all(col in data.columns for col in ['wavelength', 'intensity']): raise DataNotFoundError("Invalid data for smoothing.")
    if data.empty: log.warning("Input data empty."); return pd.DataFrame(columns=['wavelength', 'intensity'])
    method_lower = method.lower(); params = params or {}
    if method_lower not in SMOOTHING_FUNCTIONS: raise ConfigurationError(f"Unknown smoothing method: '{method}'. Available: {list(SMOOTHING_FUNCTIONS.keys())}")
    smoothing_func = SMOOTHING_FUNCTIONS[method_lower]; intensity_data = data['intensity']
    log.info(f"Applying smoothing: '{method}'...")
    try:
        smoothed_intensity = smoothing_func(intensity_data, params)
        if smoothed_intensity is None or len(smoothed_intensity) != len(data): raise ProcessingError(f"Smoothing '{method}' returned invalid output.")
        if not np.all(np.isfinite(smoothed_intensity)): nan_count = np.sum(~np.isfinite(smoothed_intensity)); log.warning(f"Smoothed intensity has {nan_count} non-finite values.")
        smoothed_data_df = pd.DataFrame({'wavelength': data['wavelength'].values, 'intensity': smoothed_intensity}, index=data.index)
        log.info(f"Smoothing '{method}' successful.")
        return smoothed_data_df
    except (ConfigurationError, DataNotFoundError, ProcessingError) as e: log.error(f"Smoothing failed: {e}"); raise
    except Exception as e: log.exception(f"Unexpected smoothing error ('{method}'): {e}"); raise ProcessingError(f"Unexpected smoothing error: {e}") from e