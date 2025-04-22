# libs_engine/processing/peak_fitting.py
# (Code from previous response up to log.debug inside fit_single_peak)
# ... (Profile functions, Baseline functions, ROI, Initial Guess, GoF) ...

import logging; log = logging.getLogger(__name__) # Add logger
import warnings; from typing import Optional, Dict, Any, List, Union, Callable, Tuple
import numpy as np; import pandas as pd
from scipy.optimize import curve_fit; from scipy.special import voigt_profile; from scipy.stats import skew
from ..core._exceptions import PeakFittingError, DataNotFoundError, ConfigurationError

# --- Profile Functions ---
def gaussian(x: np.ndarray, amp: float, cen: float, sigma: float) -> np.ndarray:
    if sigma <= 1e-9: return np.inf; return amp * np.exp(-((x - cen)**2) / (2 * sigma**2))
def lorentzian(x: np.ndarray, amp: float, cen: float, gamma: float) -> np.ndarray:
    if gamma <= 1e-9: return np.inf; return amp / (1 + ((x - cen) / gamma)**2)
def pseudo_voigt(x: np.ndarray, amp: float, cen: float, fwhm: float, eta: float) -> np.ndarray:
    if fwhm <= 1e-9: return np.inf; sigma_pv = fwhm/(2*np.sqrt(2*np.log(2))); gamma_pv = fwhm/2; eta_clipped = np.clip(eta,0,1)
    g = np.exp(-((x - cen)**2) / (2 * sigma_pv**2)); l = 1.0 / (1 + ((x - cen) / gamma_pv)**2)
    return amp * (eta_clipped * l + (1 - eta_clipped) * g)
def asymmetric_gaussian(x: np.ndarray, amp: float, cen: float, sigma_l: float, sigma_r: float) -> np.ndarray:
    if sigma_l <= 1e-9 or sigma_r <= 1e-9: return np.inf; sigma = np.where(x <= cen, sigma_l, sigma_r)
    return amp * np.exp(-((x - cen)**2) / (2 * sigma**2))
FIT_FUNCTIONS: Dict[str, Callable] = {"gaussian": gaussian, "lorentzian": lorentzian, "pseudo_voigt": pseudo_voigt, "asymmetric_gaussian": asymmetric_gaussian}
FIT_PARAM_SHAPE_COUNT: Dict[str, int] = {"gaussian": 1, "lorentzian": 1, "pseudo_voigt": 2, "asymmetric_gaussian": 2}

# --- Baseline Functions ---
def subtract_linear_baseline(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 2: return y, np.zeros_like(y); dx = x[-1] - x[0];
    if abs(dx) < 1e-9: return y, np.full_like(y, np.mean(y))
    slope = (y[-1] - y[0]) / dx; baseline = y[0] + slope * (x - x[0]); return y - baseline, baseline
def subtract_polynomial_baseline(x: np.ndarray, y: np.ndarray, order: int = 1, edge_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    n_points = len(x);
    if n_points < order + 1 or n_points < 4: return y, np.zeros_like(y)
    num_edge_points = max(2, int(n_points * edge_fraction)); num_edge_points = min(num_edge_points, n_points // 2);
    if num_edge_points < 1: num_edge_points = 1
    edge_indices = np.unique(np.concatenate((np.arange(num_edge_points), np.arange(n_points - num_edge_points, n_points))))
    if len(edge_indices) < order + 1: log.warning(f"Not enough edge points ({len(edge_indices)}) for poly baseline order {order}. Falling back."); return subtract_linear_baseline(x, y)
    try: coeffs = np.polyfit(x[edge_indices], y[edge_indices], deg=order); baseline = np.polyval(coeffs, x); return y - baseline, baseline
    except (np.linalg.LinAlgError, ValueError) as e: log.warning(f"Poly baseline fit (order {order}) failed: {e}. Falling back."); return subtract_linear_baseline(x, y)
def _snip_iteration(y: np.ndarray, width: int) -> np.ndarray:
    y_padded = np.pad(y, pad_width=width, mode='reflect'); y_snip = y.copy(); indices = np.arange(len(y))
    window_starts = indices; window_ends = indices + 2 * width
    avg_endpoints = (y_padded[window_starts] + y_padded[window_ends]) / 2.0
    mask = y_snip > avg_endpoints; y_snip[mask] = avg_endpoints[mask]; return y_snip
def subtract_snip_baseline(y: np.ndarray, max_iterations: int = 20, window_scale_factor: float = 1.0, convergence_threshold: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    n_points = len(y);
    if n_points < 5: return y, np.zeros_like(y)
    baseline = y.astype(float).copy()
    for i in range(max_iterations):
        width = int(np.clip((i + 1) * window_scale_factor, 1, n_points // 3));
        if width == 0: width = 1
        baseline_new = _snip_iteration(baseline, width); diff = np.abs(baseline - baseline_new)
        relative_diff = diff / (np.abs(baseline) + 1e-9)
        if np.max(relative_diff) < convergence_threshold: baseline = baseline_new; break
        baseline = baseline_new
    else: log.debug(f"SNIP max iterations ({max_iterations}) reached.")
    return y - baseline, baseline
BASELINE_FUNCTIONS: Dict[str, Callable] = {"linear": subtract_linear_baseline, "polynomial": subtract_polynomial_baseline, "snip": subtract_snip_baseline, "none": lambda x, y: (y, np.zeros_like(y))}

# --- ROI Definition ---
def define_roi(wavelengths: np.ndarray, peak_index: int, all_peak_indices: List[int], fwhm_guess: Optional[float], params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    roi_width_factor = params.get('roi_width_factor', 7.0); min_roi_points = params.get('min_roi_points', 7); n_points = len(wavelengths)
    if fwhm_guess is not None and fwhm_guess > 1e-9: width_wl = fwhm_guess * roi_width_factor / 2.0
    else:
        if n_points > 1: median_diff = np.median(np.diff(wavelengths)); width_wl = median_diff * (min_roi_points // 2) if min_roi_points > 2 else median_diff * 2
        else: width_wl = 0.1
        if width_wl <= 0: width_wl = 0.1; log.warning(f"Using fallback ROI width (~{width_wl*2:.2f} nm) for peak index {peak_index}.")
    peak_wl = wavelengths[peak_index]; initial_roi_min_wl = peak_wl - width_wl; initial_roi_max_wl = peak_wl + width_wl
    roi_min_wl = initial_roi_min_wl; roi_max_wl = initial_roi_max_wl
    try:
        sorted_indices = sorted(all_peak_indices); current_pos = sorted_indices.index(peak_index)
        if current_pos > 0: left_neighbour_idx = sorted_indices[current_pos - 1]; midpoint_wl = (peak_wl + wavelengths[left_neighbour_idx]) / 2.0; roi_min_wl = max(roi_min_wl, midpoint_wl)
        if current_pos < len(sorted_indices) - 1: right_neighbour_idx = sorted_indices[current_pos + 1]; midpoint_wl = (peak_wl + wavelengths[right_neighbour_idx]) / 2.0; roi_max_wl = min(roi_max_wl, midpoint_wl)
    except (ValueError, IndexError) as e: log.error(f"ROI neighbor check error for peak {peak_index}: {e}"); roi_min_wl = initial_roi_min_wl; roi_max_wl = initial_roi_max_wl
    roi_indices_mask = (wavelengths >= roi_min_wl) & (wavelengths <= roi_max_wl); roi_point_count = np.sum(roi_indices_mask)
    if roi_point_count < min_roi_points: log.warning(f"ROI for peak {peak_index} has {roi_point_count} points (min {min_roi_points}, bounds [{roi_min_wl:.2f}, {roi_max_wl:.2f}]). Fit might be unstable.")
    if roi_point_count < 3: log.error(f"ROI for peak {peak_index} has < 3 points. Cannot fit."); return None, None
    return roi_indices_mask, wavelengths[roi_indices_mask]

# --- Initial Parameter Estimation ---
def estimate_initial_params(x_roi: np.ndarray, y_roi_corrected: np.ndarray) -> Dict[str, float]:
    n_roi = len(x_roi);
    if n_roi < 2: return {}
    max_index = np.argmax(y_roi_corrected); amplitude_guess = y_roi_corrected[max_index]; center_guess = x_roi[max_index]
    if amplitude_guess <= 0: log.warning(f"ROI max intensity <= 0 ({amplitude_guess:.2e})."); amplitude_guess = max(np.mean(np.abs(y_roi_corrected)) + 1e-6, 1e-6)
    fwhm_guess = 0.0
    try:
        half_max = amplitude_guess / 2.0; above_half_max = y_roi_corrected >= half_max
        if np.any(above_half_max):
             start_idx = np.where(above_half_max)[0][0]; end_idx = np.where(above_half_max)[0][-1]
             x_left = x_roi[start_idx];
             if start_idx > 0: y1, y2 = y_roi_corrected[start_idx-1:start_idx+1]; x1, x2 = x_roi[start_idx-1:start_idx+1]; x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1) if abs(y2 - y1) > 1e-9 else x1
             x_right = x_roi[end_idx];
             if end_idx < n_roi - 1: y1, y2 = y_roi_corrected[end_idx:end_idx+2]; x1, x2 = x_roi[end_idx:end_idx+2]; x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1) if abs(y2 - y1) > 1e-9 else x1
             fwhm_guess = abs(x_right - x_left)
        if fwhm_guess < 1e-9: median_diff = np.median(np.diff(x_roi)); fwhm_guess = median_diff * 2 if n_roi > 1 else 0.1
    except Exception as e: log.warning(f"FWHM estimation failed: {e}. Using fallback."); median_diff = np.median(np.diff(x_roi)) if n_roi > 1 else 0.1 ; fwhm_guess = median_diff * 2
    fwhm_guess = max(fwhm_guess, 1e-9); sigma_guess = fwhm_guess / (2.0*np.sqrt(2.0*np.log(2.0))); gamma_guess = fwhm_guess / 2.0
    return {"amp": amplitude_guess, "cen": center_guess, "fwhm": fwhm_guess, "sigma": sigma_guess, "gamma": gamma_guess, "eta": 0.5, "sigma_l": sigma_guess, "sigma_r": sigma_guess}

# --- Goodness-of-Fit Calculation ---
def calculate_goodness_of_fit(y_true: np.ndarray, y_pred: np.ndarray, n_params: int, n_points: int) -> Dict[str, float]:
    if n_points <= n_params or n_points < 1: log.warning(f"Cannot calc metrics: n_points ({n_points}) <= n_params ({n_params})."); return {'rss': np.nan, 'r_squared': np.nan, 'aic': np.inf, 'bic': np.inf}
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()): log.warning("Cannot calc metrics: Non-finite values."); return {'rss': np.nan, 'r_squared': np.nan, 'aic': np.inf, 'bic': np.inf}
    residuals = y_true - y_pred; rss = max(np.sum(residuals**2), 1e-20); tss = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1.0 - (rss / tss) if tss > 1e-15 else 1.0; log_likelihood_term = n_points * np.log(rss / n_points)
    aic = log_likelihood_term + 2 * n_params; bic = log_likelihood_term + n_params * np.log(n_points)
    return {'rss': rss, 'r_squared': r_squared, 'aic': aic, 'bic': bic}

# --- Single Peak Fitting Orchestrator ---
def fit_single_peak(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    peak_info: pd.Series,
    all_peak_indices: List[int],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Performs the full fitting process for a single detected peak."""
    peak_index = int(peak_info['index'])
    peak_wl = peak_info['wavelength']
    fwhm_guess_wl = peak_info.get('width_fwhm_wl')
    result: Dict[str, Any] = {
        'status': 'Failed', 'message': 'Fit not started.', 'best_fit_type': None,
        'best_params': None, 'best_metrics': None, 'quality_flags': [],
        'roi_mask': None, 'roi_slice': None, 'baseline_corrected_roi_y': None,
        'baseline_roi_y': None, 'all_fit_results': {}, 'initial_guesses': None,
        'final_bounds': None
    }
    log.info(f"--- Fitting peak index {peak_index} ({peak_wl:.3f} nm) ---")

    try:
        # --- 1. Define ROI ---
        roi_params = {k: params.get(k) for k in ['roi_width_factor', 'min_roi_points'] if k in params}
        roi_mask, x_roi = define_roi(wavelengths, peak_index, all_peak_indices, fwhm_guess_wl, roi_params)
        if roi_mask is None:
            raise PeakFittingError("Failed ROI definition.")
        result['roi_mask'] = roi_mask
        roi_indices = np.where(roi_mask)[0]
        # Store slice info as dict for JSON compatibility
        result['roi_slice'] = {'start': int(roi_indices[0]), 'stop': int(roi_indices[-1]) + 1}
        y_roi = intensities[roi_mask]
        log.debug(f"Defined ROI: {x_roi[0]:.3f} - {x_roi[-1]:.3f} nm ({len(x_roi)} points).")

        # --- 2. Local Baseline Correction ---
        baseline_method = params.get('baseline_method', 'linear')
        baseline_func = BASELINE_FUNCTIONS.get(baseline_method)
        if baseline_func is None:
            raise ConfigurationError(f"Unknown baseline method: '{baseline_method}'")
        baseline_params = params.get('baseline_params', {})
        try:
            if baseline_method in ["polynomial", "linear", "none"]:
                 y_roi_corrected, baseline_roi_y = baseline_func(x_roi, y_roi, **baseline_params)
            elif baseline_method == "snip":
                 y_roi_corrected, baseline_roi_y = baseline_func(y_roi, **baseline_params)
            else: # Should not happen
                 y_roi_corrected, baseline_roi_y = baseline_func(x_roi, y_roi)
            result['baseline_corrected_roi_y'] = y_roi_corrected.tolist() # Store as list
            result['baseline_roi_y'] = baseline_roi_y.tolist() # Store as list
            log.debug(f"Applied baseline correction: {baseline_method}")
        except Exception as base_err:
             raise PeakFittingError(f"Baseline correction ('{baseline_method}') failed: {base_err}") from base_err

        # --- 3. Estimate Initial Parameters ---
        initial_guesses = estimate_initial_params(x_roi, y_roi_corrected)
        if not initial_guesses:
             raise PeakFittingError("Failed initial parameter estimation.")
        # Refine center guess based on max of corrected ROI
        initial_guesses['cen'] = x_roi[np.argmax(y_roi_corrected)]
        result['initial_guesses'] = initial_guesses # Store guesses
        log.debug(f"Initial guesses: { {k: f'{v:.3f}' if isinstance(v, float) else v for k, v in initial_guesses.items()} }")

        # --- 4. Perform Fits for Requested Profiles ---
        profile_types_to_fit = params.get('profile_types', list(FIT_FUNCTIONS.keys()))
        fitting_bounds_config = params.get('fitting_bounds', {})
        max_fit_iterations = params.get('max_fit_iterations', 5000)
        fit_results = {}

        fwhm_roi_guess = initial_guesses.get('fwhm', 0.1) # Use estimate for bounds

        for profile_type in profile_types_to_fit:
            if profile_type not in FIT_FUNCTIONS:
                log.warning(f"Skipping unknown profile type: '{profile_type}'")
                continue
            fit_func = FIT_FUNCTIONS[profile_type]

            p0, bounds = None, (-np.inf, np.inf)
            try:
                # --- Define p0 and default bounds ---
                # Use reasonable limits based on ROI and initial guesses
                min_wl_roi, max_wl_roi = x_roi[0], x_roi[-1]
                max_amp_roi = max(1e-9, initial_guesses['amp'] * 5) # Allow larger amplitude than initial guess
                max_width_roi = max(1e-6, (max_wl_roi - min_wl_roi) * 2) # Max width can be up to 2x ROI width

                if profile_type == 'gaussian':
                    p0 = [initial_guesses['amp'], initial_guesses['cen'], initial_guesses['sigma']]
                    bounds = ([0, min_wl_roi, 1e-6], [max_amp_roi, max_wl_roi, max_width_roi])
                elif profile_type == 'lorentzian':
                    p0 = [initial_guesses['amp'], initial_guesses['cen'], initial_guesses['gamma']]
                    bounds = ([0, min_wl_roi, 1e-6], [max_amp_roi, max_wl_roi, max_width_roi])
                elif profile_type == 'pseudo_voigt':
                    p0 = [initial_guesses['amp'], initial_guesses['cen'], initial_guesses['fwhm'], initial_guesses['eta']]
                    bounds = ([0, min_wl_roi, 1e-6, 0], [max_amp_roi, max_wl_roi, max_width_roi, 1])
                elif profile_type == 'asymmetric_gaussian':
                    p0 = [initial_guesses['amp'], initial_guesses['cen'], initial_guesses['sigma_l'], initial_guesses['sigma_r']]
                    bounds = ([0, min_wl_roi, 1e-6, 1e-6], [max_amp_roi, max_wl_roi, max_width_roi, max_width_roi])

                # --- Override with user bounds ---
                profile_bounds_user = fitting_bounds_config.get(profile_type)
                if profile_bounds_user:
                    if isinstance(profile_bounds_user, tuple) and len(profile_bounds_user)==2 and len(profile_bounds_user[0])==len(p0) and len(profile_bounds_user[1])==len(p0):
                         bounds = profile_bounds_user; log.debug(f"Using user bounds for {profile_type}.")
                    else: log.warning(f"Invalid user bounds format for {profile_type}, using defaults.")

                log.debug(f"Fitting {profile_type}: p0={p0}, bounds={bounds}")
                if p0 is None: raise ValueError("p0 not defined")

                fit_success = False
                with warnings.catch_warnings():
                     warnings.simplefilter("ignore", category=RuntimeWarning)
                     warnings.simplefilter("ignore", category=FutureWarning) # Can occur with numpy/scipy interactions
                     popt, pcov = curve_fit(fit_func, x_roi, y_roi_corrected, p0=p0, bounds=bounds, maxfev=max_fit_iterations, method='trf', ftol=1e-6, xtol=1e-6, gtol=1e-6)
                fit_success = True

            except RuntimeError as fit_err: log.warning(f"Fit failed for {profile_type}: {fit_err}"); fit_results[profile_type] = {'status': 'Fit Failed', 'message': str(fit_err)}
            except ValueError as val_err: log.warning(f"Fit failed {profile_type} (input error): {val_err}"); fit_results[profile_type] = {'status': 'Input Error', 'message': str(val_err)}
            except Exception as e: log.error(f"Unexpected fit error {profile_type}: {e}", exc_info=True); fit_results[profile_type] = {'status': 'Unexpected Error', 'message': str(e)}

            if fit_success:
                y_pred = fit_func(x_roi, *popt)
                n_fit_params = len(popt); metrics = calculate_goodness_of_fit(y_roi_corrected, y_pred, n_fit_params, len(x_roi))
                try: perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)
                except (ValueError, np.linalg.LinAlgError): perr = np.full(len(popt), np.nan)
                param_names = ['amplitude', 'center'] + list(fit_func.__code__.co_varnames[3:len(popt)+1]) # Get names dynamically +1 needed? Check arg count
                fitted_params_dict = dict(zip(param_names[:len(popt)], popt.tolist())) # Ensure zip stops correctly
                param_errors_dict = dict(zip(param_names[:len(popt)], perr.tolist()))
                fit_results[profile_type] = {'status': 'Success', 'params_list': popt.tolist(), 'params_dict': fitted_params_dict,
                                             'param_errors': param_errors_dict, 'covariance': pcov.tolist() if pcov is not None else None,
                                             'metrics': metrics, 'bounds_used': bounds}
                log.debug(f"Fit success ({profile_type}): R2={metrics['r_squared']:.4f}, AIC={metrics['aic']:.2f}")

        result['all_fit_results'] = fit_results

        # --- 5. Select Best Model ---
        selection_criterion = params.get('selection_criterion', 'aic').lower()
        if selection_criterion not in ['aic', 'bic']: log.warning(f"Invalid criterion '{selection_criterion}', defaulting to 'aic'."); selection_criterion = 'aic'
        best_fit_type = None; best_criterion_value = np.inf
        successful_fits = {ptype: res for ptype, res in fit_results.items() if res['status'] == 'Success'}
        if not successful_fits: raise PeakFittingError("No profile types fitted successfully.")

        for ptype, res in successful_fits.items():
            criterion_value = res['metrics'].get(selection_criterion)
            if criterion_value is not None and np.isfinite(criterion_value) and criterion_value < best_criterion_value:
                 best_criterion_value = criterion_value; best_fit_type = ptype

        if best_fit_type is None: # Fallback using R2 if AIC/BIC fail
            best_r2 = -np.inf
            for ptype, res in successful_fits.items():
                 r2 = res['metrics'].get('r_squared', -np.inf);
                 if r2 > best_r2: best_r2 = r2; best_fit_type = ptype
            if best_fit_type is None: raise PeakFittingError(f"Could not determine best fit using '{selection_criterion}' or R2.")
            else: log.warning(f"{selection_criterion} failed; selected best fit via R2: {best_fit_type}")

        result['best_fit_type'] = best_fit_type
        result['best_params'] = successful_fits[best_fit_type]['params_dict']
        result['best_metrics'] = successful_fits[best_fit_type]['metrics']
        result['final_bounds'] = successful_fits[best_fit_type]['bounds_used']
        log.info(f"Best fit model selected: {best_fit_type} ({selection_criterion}={best_criterion_value:.2f})")

        # --- 6. Quality Control ---
        min_r_squared = params.get('min_r_squared', 0.90)
        quality_flags = []; best_fit_result = successful_fits[best_fit_type]
        if best_fit_result['status'] == 'Success': quality_flags.append('Fit Converged')
        else: quality_flags.append('Fit Failed')
        if result['best_metrics']['r_squared'] >= min_r_squared: quality_flags.append('Good R2')
        else: quality_flags.append('Poor R2'); log.warning(f"Fit R2 ({result['best_metrics']['r_squared']:.3f}) < threshold ({min_r_squared}).")

        # Check residuals skewness
        best_y_pred = FIT_FUNCTIONS[best_fit_type](x_roi, *best_fit_result['params_list'])
        residuals = y_roi_corrected - best_y_pred
        try:
            if np.std(residuals) > 1e-9:
                skewness = skew(residuals)
                if abs(skewness) > params.get('max_residual_skewness', 0.8):
                    quality_flags.append('Residuals Skewed')
                    if best_fit_type != 'asymmetric_gaussian': quality_flags.append('Asymmetry Warning')
            else: skewness = 0.0
            log.debug(f"Residual skewness = {skewness:.2f}")
        except ValueError: pass

        # Check bounds hit
        best_popt = best_fit_result['params_list']; lower_bounds, upper_bounds = result['final_bounds']
        for i, p_val in enumerate(best_popt):
            # Check proximity to bounds with a small tolerance
            if abs(p_val - lower_bounds[i]) < 1e-6 * (abs(p_val) + 1) or abs(p_val - upper_bounds[i]) < 1e-6 * (abs(p_val) + 1):
                 param_name = list(result['best_params'].keys())[i]
                 quality_flags.append('Param At Bound'); log.warning(f"Param {i} ('{param_name}') hit bounds."); break

        result['quality_flags'] = list(set(quality_flags))
        result['status'] = 'Success'
        result['message'] = f'Successfully fitted peak. Best model: {best_fit_type}.'

    except (PeakFittingError, ConfigurationError, DataNotFoundError, ValueError) as e: result['message'] = f"Fitting failed: {e}"; log.error(result['message'])
    except Exception as e: result['message'] = f"Unexpected fitting error: {e}"; log.exception(result['message'])

    log.info(f"--- Finished fitting peak index {peak_index}. Status: {result['status']} ---")
    return result


# --- Main Dispatcher Function ---
def fit_peaks_detailed(processed_data: pd.DataFrame, peaks_df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[int, Dict[str, Any]]:
    if processed_data is None or not all(col in processed_data.columns for col in ['wavelength', 'intensity']): raise DataNotFoundError("Processed data invalid for fitting.")
    if peaks_df is None or peaks_df.empty or 'index' not in peaks_df.columns: log.warning("Peak list empty or invalid."); return {}
    params = params or {}
    if 'profile_types' not in params: params['profile_types'] = list(FIT_FUNCTIONS.keys())

    wavelengths = processed_data['wavelength'].values; intensities = processed_data['intensity'].values
    all_peak_indices = peaks_df['index'].astype(int).tolist()
    all_results: Dict[int, Dict[str, Any]] = {}; total_peaks = len(peaks_df)
    log.info(f"Starting detailed fitting for {total_peaks} peaks...")

    for i, (_, peak_row) in enumerate(peaks_df.iterrows()):
        peak_index = int(peak_row['index']); log.info(f"Processing peak {i+1}/{total_peaks} (Index: {peak_index})...")
        peak_info_series = peak_row.copy()
        # Add FWHM guess fallback
        if 'width_fwhm_wl' not in peak_info_series or pd.isna(peak_info_series['width_fwhm_wl']) or peak_info_series['width_fwhm_wl'] <= 0:
            if len(wavelengths) > 1: peak_info_series['width_fwhm_wl'] = np.median(np.diff(wavelengths)) * 3
            else: peak_info_series['width_fwhm_wl'] = 0.1
            log.debug(f"Added fallback FWHM guess ({peak_info_series['width_fwhm_wl']:.3f}) for peak {peak_index}.")

        single_peak_result = fit_single_peak(wavelengths, intensities, peak_info_series, all_peak_indices, params)
        all_results[peak_index] = single_peak_result

    log.info(f"Finished detailed fitting for {total_peaks} peaks.")
    return all_results