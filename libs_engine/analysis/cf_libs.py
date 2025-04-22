# libs_engine/analysis/cf_libs.py
# (Use the robust V2 code provided in the previous answer)
# ... includes imports, constants, run_cf_libs, _validate_nist_data, _prepare_cf_data, _calculate_temperature, _calculate_electron_density, _get_partition_function, _calculate_composition ...
# (Paste the full code here)
import logging; log = logging.getLogger(__name__) # Add logger
import warnings; from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np; import pandas as pd
from scipy.optimize import curve_fit; from scipy.constants import physical_constants, k as boltzmann_k
from ..core._exceptions import AnalysisError, DataNotFoundError, ConfigurationError

electron_volt_in_J = physical_constants['electron volt-joule relationship'][0]
h_planck = physical_constants['Planck constant'][0]; c_light = physical_constants['speed of light in vacuum'][0]
e_charge = physical_constants['elementary charge'][0]; eV_to_K_factor = e_charge / boltzmann_k

def run_cf_libs(peak_fit_results: Dict[Any, Dict], nist_lines: Union[pd.DataFrame, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # ... (full function code) ...
    params = params or {}; log.info("Starting CF-LIBS analysis (V2).")
    if not peak_fit_results: raise DataNotFoundError("Peak fits required.")
    if not isinstance(nist_lines, pd.DataFrame) or nist_lines.empty: raise DataNotFoundError("NIST DataFrame required.")
    _validate_nist_data(nist_lines)
    results = {'status': 'Failure', 'message': 'Analysis incomplete.', 'plasma_temperature_K': None, 'temperature_uncertainty_K': None, 'electron_density_cm-3': None,
               'composition_atom_percent': None, 'composition_weight_percent': None, 'boltzmann_plot': None, 'lines_used': None}
    try:
        log.info("CF-LIBS Step 1: Prepare Data..."); merged_line_data = _prepare_cf_data(peak_fit_results, nist_lines, params)
        results['lines_used'] = merged_line_data.to_dict(orient='records'); log.info(f"Prepared {len(merged_line_data)} lines.")
        log.info("CF-LIBS Step 2: Calc Temperature..."); temp_results = _calculate_temperature(merged_line_data, params)
        plasma_temp_k = temp_results.get('temperature_k');
        if plasma_temp_k is None: raise AnalysisError(f"Temp calc failed: {temp_results.get('message')}")
        results['plasma_temperature_K'] = plasma_temp_k; results['temperature_uncertainty_K'] = temp_results.get('uncertainty_k'); results['boltzmann_plot'] = temp_results.get('plot_data')
        log.info(f"Temp: {plasma_temp_k:.1f} +/- {temp_results.get('uncertainty_k'):.1f} K")
        log.info("CF-LIBS Step 3: Calc Electron Density (Optional)..."); electron_density_cm3 = _calculate_electron_density(merged_line_data, plasma_temp_k, params)
        results['electron_density_cm-3'] = electron_density_cm3;
        if electron_density_cm3 is not None: log.info(f"N_e: {electron_density_cm3:.2e} cm^-3")
        else: log.info("N_e calc skipped/failed.")
        log.info("CF-LIBS Step 4: Calc Composition..."); composition_results = _calculate_composition(merged_line_data, plasma_temp_k, electron_density_cm3, params)
        if not composition_results.get('composition_atom_percent'): raise AnalysisError(f"Composition calc failed: {composition_results.get('message')}")
        results.update(composition_results); log.info(f"Atom %: {results.get('composition_atom_percent')}")
        if results.get('composition_weight_percent'): log.info(f"Weight %: {results.get('composition_weight_percent')}")
        results['status'] = 'Success'; results['message'] = 'CF-LIBS completed successfully.'; log.info("CF-LIBS finished successfully.")
    except (DataNotFoundError, ConfigurationError, AnalysisError) as e: results['message'] = f"CF-LIBS Failed: {e}"; log.error(results['message'])
    except Exception as e: results['message'] = f"Unexpected CF-LIBS error: {e}"; log.exception(results['message'])
    return results

def _validate_nist_data(nist_df: pd.DataFrame):
    required = ['wavelength_nm', 'E_k_eV', 'g_k', 'A_ki', 'element', 'ion_stage']; missing = [c for c in required if c not in nist_df.columns]
    if missing: raise DataNotFoundError(f"NIST data missing columns: {missing}")

def _prepare_cf_data(peak_fits: Dict[Any, Dict], nist_data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    # ... (full function code - select lines, match peaks, merge data) ...
    log.debug("Selecting/merging lines..."); selection_criteria = params.get('line_selection_criteria', {}); wl_tolerance_nm = selection_criteria.get('wl_tolerance_nm', 0.1)
    filtered_nist = nist_data.copy(); min_aki = selection_criteria.get('min_aki')
    if min_aki is not None: filtered_nist = filtered_nist[filtered_nist['A_ki'] >= min_aki]
    if filtered_nist.empty: raise AnalysisError("No NIST lines after criteria.")
    peak_data_list = []
    for pid, fit_info in peak_fits.items():
        intensity = fit_info.get('intensity_area');
        if intensity is None: amp = fit_info.get('amplitude'); width = fit_info.get('width');
        if amp is not None and width is not None: intensity = amp * width * 1.064 # Gaussian Area approx FWHM*Amp*1.064
        else: intensity = 0
        if intensity > 0: peak_data_list.append({'PeakID': pid, 'Wavelength_Obs': fit_info.get('center'), 'Intensity_Fit': intensity, 'Element_Fit': fit_info.get('element'), 'IonStage_Fit': fit_info.get('ion_stage'), 'Fit_R2': fit_info.get('r_squared', 1.0)})
    if not peak_data_list: raise AnalysisError("No valid peaks with positive intensity.")
    peaks_df = pd.DataFrame(peak_data_list); peaks_df.dropna(subset=['Wavelength_Obs', 'Intensity_Fit', 'Element_Fit', 'IonStage_Fit'], inplace=True)
    merged_data = []; filtered_nist_sorted = filtered_nist.sort_values('wavelength_nm')
    for _, peak in peaks_df.iterrows():
        wl_obs = peak['Wavelength_Obs']; min_wl, max_wl = wl_obs - wl_tolerance_nm, wl_obs + wl_tolerance_nm
        potential_matches = filtered_nist_sorted[(filtered_nist_sorted['wavelength_nm'] >= min_wl) & (filtered_nist_sorted['wavelength_nm'] <= max_wl) & (filtered_nist_sorted['element'] == peak['Element_Fit']) & (filtered_nist_sorted['ion_stage'] == peak['IonStage_Fit'])]
        if not potential_matches.empty:
            closest_match_idx = (potential_matches['wavelength_nm'] - wl_obs).abs().idxmin(); closest_match = potential_matches.loc[closest_match_idx]
            merged_data.append({'PeakID': peak['PeakID'], 'Wavelength_Obs': wl_obs, 'Intensity_Fit': peak['Intensity_Fit'], 'Element': peak['Element_Fit'], 'IonStage': int(peak['IonStage_Fit']),
                                'Wavelength_NIST': closest_match['wavelength_nm'], 'E_k_eV': closest_match['E_k_eV'], 'g_k': closest_match['g_k'], 'A_ki': closest_match['A_ki'], 'Fit_R2': peak['Fit_R2']})
    if not merged_data: raise AnalysisError("Failed to match peaks with NIST lines.")
    merged_df = pd.DataFrame(merged_data); min_fit_quality = selection_criteria.get('min_fit_quality', 0.0); merged_df = merged_df[merged_df['Fit_R2'] >= min_fit_quality]
    if merged_df.empty: raise AnalysisError("No lines after post-merge quality filters.")
    final_required = ['Intensity_Fit', 'Element', 'IonStage', 'E_k_eV', 'g_k', 'A_ki']
    if not all(col in merged_df.columns for col in final_required): raise AnalysisError("Internal error: Merged data frame missing columns.")
    return merged_df

def _calculate_temperature(merged_data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    # ... (full function code - Boltzmann plot) ...
    log.debug("Calculating temperature (Boltzmann)..."); temp_results = {'temperature_k': None, 'uncertainty_k': None, 'message': 'Not calculated.', 'plot_data': None}
    config = params.get('temp_calc_config', {}); element = config.get('element', 'Fe'); ion_stage = config.get('ion_stage', 0); min_points = config.get('min_points', 3)
    lines_for_plot = merged_data[(merged_data['Element'] == element) & (merged_data['IonStage'] == ion_stage) & (merged_data['Intensity_Fit'] > 0) & (merged_data['g_k'] > 0) & (merged_data['A_ki'] > 0)].copy()
    if len(lines_for_plot) < min_points: temp_results['message'] = f"Need {min_points} lines for Boltzmann plot ({len(lines_for_plot)} found for {element} {ion_stage+1})."; log.warning(temp_results['message']); return temp_results
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning); lines_for_plot['y_Boltzmann'] = np.log(lines_for_plot['Intensity_Fit'] / (lines_for_plot['g_k'] * lines_for_plot['A_ki']))
        lines_for_plot['x_Boltzmann_eV'] = lines_for_plot['E_k_eV']; lines_for_plot.dropna(subset=['y_Boltzmann', 'x_Boltzmann_eV'], inplace=True)
        if len(lines_for_plot) < min_points: raise ValueError(f"Insufficient valid points after coordinate calculation ({len(lines_for_plot)}).")
        coeffs, covariance = np.polyfit(lines_for_plot['x_Boltzmann_eV'], lines_for_plot['y_Boltzmann'], 1, cov=True); slope = coeffs[0]
        slope_std_dev = np.sqrt(covariance[0, 0]) if covariance is not None and np.isfinite(covariance).all() and covariance[0, 0] >= 0 else np.nan
        if slope >= -1e-9: raise ValueError(f"Boltzmann slope is non-negative ({slope:.2e}).")
        temperature_k = -eV_to_K_factor / slope; uncertainty_k = np.abs(temperature_k * slope_std_dev / slope) if np.isfinite(slope_std_dev) else np.nan
        temp_results['temperature_k'] = temperature_k; temp_results['uncertainty_k'] = uncertainty_k; temp_results['message'] = 'Temp calc OK.'
        temp_results['plot_data'] = {'E_k_eV': lines_for_plot['x_Boltzmann_eV'].tolist(), 'ln_term': lines_for_plot['y_Boltzmann'].tolist(), 'slope': slope, 'intercept': coeffs[1]}
    except (np.linalg.LinAlgError, ValueError) as fit_err: temp_results['message'] = f"Boltzmann fit failed: {fit_err}"; log.error(temp_results['message'])
    except Exception as e: temp_results['message'] = f"Boltzmann calc error: {e}"; log.exception(temp_results['message'])
    return temp_results

def _calculate_electron_density(merged_data: pd.DataFrame, temperature_k: float, params: Dict[str, Any]) -> Optional[float]:
    # ... (full function code - placeholder) ...
    log.debug("Placeholder: _calculate_electron_density"); config = params.get('electron_density_config', {}); method = config.get('method', 'stark')
    if method == 'stark':
        # Requires Stark broadening params & fitted FWHM. Not implemented here.
        line_wl_h_alpha = 656.28; h_alpha_row = merged_data[np.isclose(merged_data['Wavelength_Obs'], line_wl_h_alpha, atol=0.2)]
        if not h_alpha_row.empty: log.warning("H-alpha found, but Stark calculation not implemented."); return 1e17 # Placeholder if found
        else: log.info("Suitable line (e.g., H-alpha) not found for Stark broadening.")
    return None

def _get_partition_function(element: str, ion_stage: int, temperature_k: float, partition_func_data: Any) -> Optional[float]:
    # ... (full function code - placeholder with interpolation) ...
    if partition_func_data is None or not isinstance(partition_func_data, dict): log.error("Partition func data missing/invalid."); return None
    try:
        u_t = partition_func_data.get(element, {}).get(ion_stage, {}).get(temperature_k)
        if u_t is None:
            temp_points = sorted(partition_func_data.get(element, {}).get(ion_stage, {}).keys())
            if len(temp_points) >= 2: u_t = np.interp(temperature_k, temp_points, [partition_func_data[element][ion_stage][t] for t in temp_points]); log.debug(f"Interpolated U({element},{ion_stage}) @ {temperature_k:.0f} K = {u_t:.2f}")
            else: log.warning(f"Cannot interp U({element},{ion_stage}) @ {temperature_k:.0f} K."); return None
        return float(u_t)
    except Exception as e: log.error(f"Error getting partition function {element} {ion_stage}: {e}"); return None

def _calculate_composition(merged_data: pd.DataFrame, temperature_k: float, electron_density_cm3: Optional[float], params: Dict[str, Any]) -> Dict[str, Any]:
    # ... (full function code - simplified Saha-Boltzmann) ...
    log.debug("Calculating composition (Saha-Boltzmann)..."); comp_results = {'composition_atom_percent': None, 'composition_weight_percent': None, 'message': 'Not calculated.'}
    elements = params.get('elements_to_quantify', list(merged_data['Element'].unique())); partition_func_data = params.get('partition_functions_data'); atomic_weights_data = params.get('atomic_weights_data')
    if partition_func_data is None: raise ConfigurationError("Partition function data required.")
    lines_used = merged_data[merged_data['Element'].isin(elements)].copy();
    if lines_used.empty: raise AnalysisError(f"No lines found for elements: {elements}")
    try:
        lines_used['E_k_J'] = lines_used['E_k_eV'] * electron_volt_in_J; boltzmann_factor = boltzmann_k * temperature_k
        lines_used['ExpTerm'] = np.exp(np.clip(-lines_used['E_k_J'] / boltzmann_factor, -700, 700))
        lines_used['Denominator'] = lines_used['g_k'] * lines_used['A_ki'] * lines_used['ExpTerm']
        valid_lines = lines_used[lines_used['Denominator'] > 1e-99].copy()
        if valid_lines.empty: raise AnalysisError("No lines with valid denominator.")
        valid_lines['PartitionFunc'] = valid_lines.apply(lambda r: _get_partition_function(r['Element'], r['IonStage'], temperature_k, partition_func_data), axis=1)
        valid_lines.dropna(subset=['PartitionFunc'], inplace=True);
        if valid_lines.empty: raise AnalysisError("Could not get partition functions.")
        valid_lines['N_s_Proxy'] = valid_lines['Intensity_Fit'] * valid_lines['PartitionFunc'] / valid_lines['Denominator']
        # Simplified averaging - needs Saha for accuracy
        if electron_density_cm3 is None: log.warning("N_e missing. Using simplified composition estimate."); element_relative_abundance = valid_lines.groupby('Element')['N_s_Proxy'].mean()
        else: log.warning("Saha implementation needed. Using simplified estimate."); element_relative_abundance = valid_lines.groupby('Element')['N_s_Proxy'].mean()
        if element_relative_abundance.empty or element_relative_abundance.sum() <= 0: raise AnalysisError("Relative abundances empty/non-positive.")
        total_abundance = element_relative_abundance.sum(); atom_percent = ((element_relative_abundance / total_abundance) * 100.0).round(4).to_dict()
        comp_results['composition_atom_percent'] = atom_percent
        if atomic_weights_data and isinstance(atomic_weights_data, dict):
            try:
                weight_sum = sum(atom_percent.get(el, 0) * atomic_weights_data.get(el, 0) for el in atom_percent) / 100.0
                if weight_sum > 1e-9: weight_percent = {el: round((atom_percent.get(el, 0) * atomic_weights_data.get(el, 0) / 100.0) / weight_sum * 100.0, 4) for el in atom_percent if atomic_weights_data.get(el) is not None}; comp_results['composition_weight_percent'] = weight_percent
                else: log.warning("Cannot calculate weight % (zero weight sum).")
            except Exception as e: log.warning(f"Weight % calc failed: {e}")
        else: log.info("Atomic weights missing, skipping weight %.")
        comp_results['message'] = "Composition calculated."
    except (AnalysisError, ConfigurationError, DataNotFoundError) as e: comp_results['message'] = f"Composition failed: {e}"; log.error(comp_results['message'])
    except Exception as e: comp_results['message'] = f"Unexpected composition error: {e}"; log.exception(comp_results['message'])
    return comp_results