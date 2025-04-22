# LIBS_Software/api_server.py

import os
import logging
from flask import Flask, request, jsonify, send_from_directory # type: ignore
from flask_cors import CORS # type: ignore
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from typing import Optional

from libs_engine import LIBSAnalysisEngine
from libs_engine.core._exceptions import (
    LIBSEngineError, ConfigurationError, DataLoadingError, DataNotFoundError,
    ProcessingError, PeakFindingError, PeakFittingError, AnalysisError, DatabaseError
)
from libs_engine.processing import peak_fitting # Import fitting module for function access

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'uploads'))
ALLOWED_EXTENSIONS = {'txt', 'csv', 'asc', 'spc'}
FRONTEND_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'frontend', 'build'))

engine: Optional[LIBSAnalysisEngine] = None
log = logging.getLogger('api_server')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_json_serializable(data):
    if isinstance(data, (np.ndarray, pd.Series)): return data.tolist()
    if isinstance(data, pd.DataFrame): return data.replace({np.nan: None}).to_dict(orient='list')
    if isinstance(data, dict): return {k: make_json_serializable(v) for k, v in data.items()}
    if isinstance(data, list): return [make_json_serializable(item) for item in data]
    if isinstance(data, (np.int64, np.int32)): return int(data)
    if isinstance(data, (np.float64, np.float32)):
        if np.isnan(data): return None
        if np.isinf(data): return str(data)
        return float(data)
    if isinstance(data, np.bool_): return bool(data)
    if isinstance(data, slice): return {'start': data.start, 'stop': data.stop, 'step': data.step}
    return data

def _calculate_fit_curves(x_coords, fit_func, popt):
    try:
        x_np = np.asarray(x_coords)
        y_fit = fit_func(x_np, *popt)
        y_fit_serializable = np.where(np.isfinite(y_fit), y_fit, None).tolist()
        return y_fit_serializable
    except Exception as e:
        log.warning(f"Could not calculate fit curve points: {e}")
        return None

def create_app():
    global engine
    if engine is None:
        try:
            engine = LIBSAnalysisEngine()
            log.info("LIBS Analysis Engine initialized.")
        except Exception as e:
            log.critical(f"CRITICAL: Failed to initialize LIBSAnalysisEngine: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize analysis engine.") from e

    app = Flask(__name__, static_folder=None)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    log.info("Flask app created with CORS enabled for /api/*.")

    # --- API Routes ---
    @app.route('/api/status', methods=['GET'])
    def get_status():
        if engine is None: return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
        try:
            state = engine.state.name
            status_info = {'engine_state': state, 'loaded_file': engine.data.source_filepath,
                           'has_peaks': engine.data.has_data('peaks'), 'has_fits': engine.data.has_data('fit_results')}
            return jsonify({'status': 'success', 'message': f'Engine state: {state}', 'data': status_info})
        except Exception as e:
            log.error(f"Status error: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Failed status check'}), 500

    @app.route('/api/load', methods=['POST'])
    def load_spectrum_data():
        if engine is None: return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
        if 'spectrumFile' not in request.files: return jsonify({'status': 'error', 'message': 'No file part'}), 400
        file = request.files['spectrumFile']
        if file.filename == '': return jsonify({'status': 'error', 'message': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath); log.info(f"Uploaded {filename}")
                load_params = {} # Extract from request.form if needed
                engine.load_data(filepath, **load_params)
                raw_spectrum_data = make_json_serializable(engine.data.raw_data)
                # os.remove(filepath) # Optional cleanup
                return jsonify({'status': 'success', 'message': f'Loaded {filename}.', 'data': {'spectrum': raw_spectrum_data}})
            except (DataLoadingError, ConfigurationError, FileNotFoundError, LIBSEngineError) as e:
                log.error(f"Load failed '{filename}': {e}");
                if os.path.exists(filepath): os.remove(filepath); return jsonify({'status': 'error', 'message': str(e)}), 400
            except Exception as e:
                log.exception(f"Unexpected load error '{filename}': {e}");
                if os.path.exists(filepath): os.remove(filepath); return jsonify({'status': 'error', 'message': 'Server load error.'}), 500
        else: return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

    @app.route('/api/smooth', methods=['POST'])
    def smooth_spectrum_data():
        if engine is None: return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
        if engine.state not in [engine.state.DATA_LOADED, engine.state.SMOOTHED]: return jsonify({'status': 'error', 'message': f'Invalid state ({engine.state.name}) for smoothing.'}), 400
        try:
            params = request.json; log.info(f"Smoothing: {params}")
            method = params.get('method', 'savitzky_golay')
            engine.smooth_data(method=method, params=params)
            processed_spectrum = make_json_serializable(engine.data.processed_data)
            return jsonify({'status': 'success', 'message': f'Smoothed ({method}).', 'data': {'smoothed_spectrum': processed_spectrum}})
        except (ConfigurationError, DataNotFoundError, ProcessingError, LIBSEngineError) as e:
            log.error(f"Smooth failed: {e}"); return jsonify({'status': 'error', 'message': str(e)}), 400
        except Exception as e: log.exception(f"Unexpected smooth error: {e}"); return jsonify({'status': 'error', 'message': 'Server smooth error.'}), 500
        log.info(f"Attempting smooth. Current state: {engine.state.name}")


    @app.route('/api/find_peaks', methods=['POST'])
    def find_spectrum_peaks():
        if engine is None: return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
        log.info(f"-> Received /api/find_peaks request. Current Engine State: {engine.state.name}")
        if engine is None: return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
        if engine.state not in [engine.state.DATA_LOADED, engine.state.SMOOTHED]: return jsonify({'status': 'error', 'message': f'Invalid state ({engine.state.name}) for peak finding.'}), 400
        try:
            params = request.json; log.info(f"Finding peaks: {params}")
            method = params.get('method', 'simple')
            engine.find_peaks(method=method, params=params, nist_data=engine.data.nist_lines)
            peaks_df = engine.data.peaks
            if isinstance(peaks_df, pd.DataFrame) and not peaks_df.empty:
                count = len(peaks_df)
            # Convert DataFrame to list of dictionaries for easier frontend use
                peaks_result = peaks_df.to_dict(orient='records')
            # Convert numpy types within the list of dicts just in case
                peaks_result = make_json_serializable(peaks_result)
            else:
                count = 0
                peaks_result = [] 
            return jsonify({'status': 'success', 'message': f'Found {count} peaks ({method}).', 'data': {'peaks': peaks_result if peaks_result is not None else []}})
        except (ConfigurationError, DataNotFoundError, PeakFindingError, LIBSEngineError) as e:
            log.error(f"Find peaks failed: {e}"); return jsonify({'status': 'error', 'message': str(e)}), 400
        except Exception as e: log.exception(f"Unexpected find peaks error: {e}"); return jsonify({'status': 'error', 'message': 'Server find peaks error.'}), 500

    @app.route('/api/fit_peaks', methods=['POST'])
    def fit_spectrum_peaks():
        if engine is None: return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
        log.info(f"-> Received /api/find_peaks request. Current Engine State: {engine.state.name}")
        if engine.state != engine.state.PEAKS_FOUND: return jsonify({'status': 'error', 'message': f'Invalid state ({engine.state.name}) for peak fitting.'}), 400
        try:
            params = request.json; log.info(f"Fitting peaks: {params}")
            engine.fit_peaks(params=params)
            detailed_results = engine.data.peak_fit_results or {}
            augmented_results = {}
            wavelengths = engine.data.processed_data['wavelength'].values if engine.data.processed_data is not None else None
            for peak_idx_str, fit_info in detailed_results.items():
                 peak_idx = int(peak_idx_str) # Ensure int key
                 augmented_info = make_json_serializable(fit_info.copy())
                 if fit_info.get('status') == 'Success':
                     fit_type = fit_info.get('best_fit_type')
                     popt = fit_info.get('params_list')
                     fit_func = peak_fitting.FIT_FUNCTIONS.get(fit_type)
                     roi_slice_info = fit_info.get('roi_slice')
                     x_fit_curve = None
                     if roi_slice_info and wavelengths is not None:
                         slc = slice(roi_slice_info['start'], roi_slice_info['stop'])
                         roi_wl = wavelengths[slc]
                         if len(roi_wl) > 1: x_fit_curve = np.linspace(roi_wl[0], roi_wl[-1], 100)

                     if fit_func and popt and x_fit_curve is not None:
                         y_fit_curve = _calculate_fit_curves(x_fit_curve, fit_func, popt)
                         augmented_info['fit_curve_x'] = x_fit_curve.tolist()
                         augmented_info['fit_curve_y'] = y_fit_curve
                 augmented_results[peak_idx] = augmented_info

            log.info(f"Peak fitting complete for {len(augmented_results)} peaks.")
            return jsonify({'status': 'success', 'message': 'Peak fitting complete.', 'data': {'fit_results': augmented_results}})
        except (ConfigurationError, DataNotFoundError, PeakFittingError, LIBSEngineError) as e:
            log.error(f"Fit peaks failed: {e}"); return jsonify({'status': 'error', 'message': str(e)}), 400
        except Exception as e: log.exception(f"Unexpected fit peaks error: {e}"); return jsonify({'status': 'error', 'message': 'Server fit peaks error.'}), 500

    @app.route('/api/run_auto', methods=['POST'])
    def run_auto_analysis_route():
        if engine is None: return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500
        if engine.state not in [engine.state.DATA_LOADED, engine.state.SMOOTHED]: return jsonify({'status': 'error', 'message': f'Invalid state ({engine.state.name}) for auto analysis.'}), 400
        try:
            params = request.json; log.info(f"Auto Analysis: {params}")
            elements = params.get('elements', [])
            auto_result_dict = engine.run_auto_analysis(elements=elements)
            response_data = {
                'status': auto_result_dict.get('status', 'Failed'),
                'message': auto_result_dict.get('message', 'Auto analysis finished.'),
                'data': {
                    'analysis_type': auto_result_dict.get('analysis_type'),
                    'output': make_json_serializable(auto_result_dict.get('output')),
                    'spectrum': make_json_serializable(engine.data.processed_data),
                    'peaks': make_json_serializable(engine.data.peaks),
                    # Add fits if needed: 'fits': make_json_serializable(engine.data.peak_fit_results)
                }
            }
            log.info(f"Auto analysis status: {response_data['status']}")
            status_code = 200 if response_data['status'] == 'Success' else 400
            return jsonify(response_data), status_code
        except (ConfigurationError, DataNotFoundError, ProcessingError, PeakFindingError, PeakFittingError, AnalysisError, DatabaseError, LIBSEngineError) as e:
            log.error(f"Auto analysis failed: {e}"); return jsonify({'status': 'error', 'message': str(e)}), 400
        except Exception as e: log.exception(f"Unexpected auto analysis error: {e}"); return jsonify({'status': 'error', 'message': 'Server auto analysis error.'}), 500

    # Add routes for quantify, cf_libs, ml...

    # --- Serve React App Route ---
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_react_app(path):
        build_dir = FRONTEND_BUILD_DIR
        if path != "" and os.path.exists(os.path.join(build_dir, path)):
            return send_from_directory(build_dir, path)
        elif os.path.exists(os.path.join(build_dir, 'index.html')):
            return send_from_directory(build_dir, 'index.html')
        else: return jsonify({"status": "error", "message": "UI not built or found."}), 404

    return app

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
     log = logging.getLogger('api_server_run')
     try:
         app = create_app()
         log.info("Starting Flask development server...")
         app.run(host='127.0.0.1', port=5000, debug=True)
     except Exception as e:
         log.critical(f"Failed to create or run Flask app: {e}", exc_info=True)

