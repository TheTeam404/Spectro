# libs_engine/processing/peak_detection.py
# (Use the robust code provided in the previous answer)
# ... includes find_peaks_simple, find_peaks_advanced_nist, find_peaks ...
# (Paste the full code here)
import logging; log = logging.getLogger(__name__) # Add logger
from typing import Optional, Dict, Any, List, Union
import numpy as np; import pandas as pd
from scipy.signal import find_peaks as scipy_find_peaks
from ..core._exceptions import PeakFindingError, ConfigurationError, DataNotFoundError

def find_peaks_simple(data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    # ... (full function code) ...
    if data is None or not all(col in data.columns for col in ['wavelength', 'intensity']): raise DataNotFoundError("Invalid input data for peak finding.")
    if data.empty: log.warning("Input data empty."); return pd.DataFrame(columns=['index', 'wavelength', 'intensity', 'prominence', 'width_heights', 'width_fwhm', 'height', 'threshold', 'width_fwhm_wl'])
    intensities = data['intensity'].values; wavelengths = data['wavelength'].values
    default_params = {'height': None, 'threshold': None, 'distance': None, 'prominence': np.std(intensities) * 0.5 if len(intensities)>1 else None, 'width': None, 'rel_height': 0.5}
    if params: find_peaks_args = {k: params.get(k, default_params.get(k)) for k in default_params if k in params or default_params.get(k) is not None}
    else: find_peaks_args = {'prominence': default_params['prominence'], 'rel_height': 0.5};
    if find_peaks_args.get('prominence') is None: find_peaks_args = {'rel_height': 0.5}
    if find_peaks_args.get('width') is not None or find_peaks_args.get('prominence') is not None: find_peaks_args['rel_height'] = params.get('rel_height', 0.5) if params else 0.5
    log.debug(f"find_peaks_simple params: {find_peaks_args}")
    log.debug(f"Using find_peaks parameters: {find_peaks_args}")
    log.debug(f"Intensity array length: {len(intensities)}")
    if len(intensities) > 0:
        log.debug(f"Intensity stats: min={np.min(intensities):.2f}, max={np.max(intensities):.2f}, mean={np.mean(intensities):.2f}, std={np.std(intensities):.2f}")
    try: 
        peak_indices, properties = scipy_find_peaks(intensities, **find_peaks_args)
        log.info(f"SCIPY Found {len(peak_indices)} indices: {peak_indices}")
        log.debug(f"SCIPY properties: {properties}")
    except Exception as e: raise PeakFindingError(f"scipy.signal.find_peaks failed: {e}") from e
    if len(peak_indices) == 0: log.info("No peaks found."); return pd.DataFrame(columns=['index', 'wavelength', 'intensity', 'prominence', 'width_heights', 'width_fwhm', 'height', 'threshold', 'width_fwhm_wl'])
    fwhm, width_heights = properties.get("widths"), properties.get("width_heights")
    peak_info = pd.DataFrame({'index': peak_indices, 'wavelength': wavelengths[peak_indices], 'intensity': intensities[peak_indices],
                              'prominence': properties.get('prominences', np.nan), 'width_heights': width_heights if width_heights is not None else np.nan,
                              'width_fwhm': fwhm if fwhm is not None else np.nan, 'height': properties.get('peak_heights', intensities[peak_indices]),
                              'threshold': properties.get('thresholds', np.nan)})
    if 'width_fwhm' in peak_info.columns and not peak_info['width_fwhm'].isna().all():
        if len(wavelengths) > 1: peak_info['width_fwhm_wl'] = peak_info['width_fwhm'] * np.median(np.diff(wavelengths))
        else: peak_info['width_fwhm_wl'] = np.nan
    else: peak_info['width_fwhm_wl'] = np.nan
    log.info(f"Simple detection found {len(peak_info)} peaks.")
    return peak_info

def find_peaks_advanced_nist(data: pd.DataFrame, nist_data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    # ... (full function code) ...
    if data is None or not all(col in data.columns for col in ['wavelength', 'intensity']): raise DataNotFoundError("Invalid spectral data for NIST peak finding.")
    if nist_data is None or nist_data.empty or 'wavelength_nm' not in nist_data.columns: raise DataNotFoundError("Valid NIST data DataFrame required.")
    elements = params.get('elements'); window_nm = params.get('search_window_nm'); min_nist_intensity = params.get('min_nist_intensity')
    local_find_peaks_params = {k: v for k, v in params.items() if k not in ['elements', 'search_window_nm', 'min_nist_intensity']}
    if not elements: raise ConfigurationError("Parameter 'elements' required.")
    if window_nm is None or window_nm <= 0: raise ConfigurationError("Parameter 'search_window_nm' required.")
    log.info(f"Advanced NIST finding: elements={elements}, window={window_nm} nm.")
    nist_subset = nist_data[nist_data['element'].isin(elements)].copy()
    if min_nist_intensity is not None and 'rel_intensity' in nist_subset.columns:
        nist_subset['rel_intensity_num'] = pd.to_numeric(nist_subset['rel_intensity'], errors='coerce').fillna(0)
        nist_subset = nist_subset[nist_subset['rel_intensity_num'] >= min_nist_intensity]
    if nist_subset.empty: log.warning(f"No NIST lines found for {elements}."); return pd.DataFrame(columns=['index', 'wavelength', 'intensity', 'prominence', 'width_heights', 'width_fwhm', 'height', 'threshold', 'width_fwhm_wl'])
    all_found_peaks = []; wavelengths = data['wavelength'].values; intensities = data['intensity'].values
    for _, nist_line in nist_subset.iterrows():
        nist_wl = nist_line['wavelength_nm']; wl_min, wl_max = nist_wl - window_nm, nist_wl + window_nm; roi_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        if not np.any(roi_mask): continue
        roi_indices = np.where(roi_mask)[0]; roi_data_df = pd.DataFrame({'wavelength': wavelengths[roi_indices], 'intensity': intensities[roi_indices]})
        try:
             local_peaks_df = find_peaks_simple(roi_data_df, local_find_peaks_params)
             if not local_peaks_df.empty:
                 local_peaks_df['original_index'] = roi_indices[local_peaks_df['index']]; local_peaks_df['nist_match_wl'] = nist_wl; local_peaks_df['nist_element'] = nist_line['element']
                 all_found_peaks.append(local_peaks_df)
        except (DataNotFoundError, PeakFindingError) as e: log.warning(f"Local peak finding failed for NIST line {nist_wl:.2f}: {e}")
        except Exception as e: log.error(f"Error processing ROI for NIST line {nist_wl:.2f}: {e}", exc_info=True)
    if not all_found_peaks: log.info("No peaks found in NIST regions."); return pd.DataFrame(columns=['index', 'wavelength', 'intensity', 'prominence', 'width_heights', 'width_fwhm', 'height', 'threshold', 'width_fwhm_wl'])
    combined_peaks = pd.concat(all_found_peaks, ignore_index=True); combined_peaks.rename(columns={'original_index': 'index'}, inplace=True)
    combined_peaks['nist_dist'] = abs(combined_peaks['wavelength'] - combined_peaks['nist_match_wl'])
    final_peaks = combined_peaks.sort_values('nist_dist').drop_duplicates(subset=['index'], keep='first')
    final_peaks = final_peaks.drop(columns=['nist_match_wl', 'nist_element', 'nist_dist'], errors='ignore')
    log.info(f"Advanced NIST found {len(final_peaks)} unique peaks.")
    standard_cols = ['index', 'wavelength', 'intensity', 'prominence', 'width_heights', 'width_fwhm', 'height', 'threshold', 'width_fwhm_wl']
    for col in standard_cols:
        if col not in final_peaks.columns: final_peaks[col] = np.nan
    return final_peaks[standard_cols].sort_values('index').reset_index(drop=True)

def find_peaks(data: pd.DataFrame, method: str = 'simple', params: Optional[Dict[str, Any]] = None, nist_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    log.info(f"Dispatching peak finding: '{method}'")
    params = params or {}
    if method == 'simple': return find_peaks_simple(data, params)
    elif method == 'advanced_nist':
        if nist_data is None: raise DataNotFoundError("NIST data required for 'advanced_nist'.")
        if 'elements' not in params or 'search_window_nm' not in params: raise ConfigurationError("'elements' and 'search_window_nm' required for 'advanced_nist'.")
        return find_peaks_advanced_nist(data, nist_data, params)
    else: raise ConfigurationError(f"Unknown peak finding method: '{method}'")