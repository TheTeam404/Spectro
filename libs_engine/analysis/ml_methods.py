# libs_engine/analysis/ml_methods.py
# (Use the robust V2 code provided in the previous answer)
# ... includes imports, checks, apply_model, _prepare_features, _fit_simple_model, _apply_specific_model ...
# (Paste the full code here)
import logging; log = logging.getLogger(__name__) # Add logger
import os; from typing import Optional, Dict, Any, Union, List
import numpy as np; import pandas as pd
try: import joblib
except ImportError: joblib = None
try: from sklearn.base import BaseEstimator; from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA; from sklearn.cross_decomposition import PLSRegression # type: ignore
except ImportError: BaseEstimator = None; StandardScaler = PCA = PLSRegression = None
from ..core._exceptions import AnalysisError, ConfigurationError, DataNotFoundError

def apply_model(data: Union[pd.DataFrame, np.ndarray, Dict[Any, Any]], model_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # ... (full function code) ...
    if data is None: raise DataNotFoundError(f"Input data for ML '{model_type}' is None.")
    if joblib is None and (params.get('model_path') or params.get('scaler_path')): raise ConfigurationError("joblib needed to load models/scalers.")
    if BaseEstimator is None and model_type not in ['custom']: raise ConfigurationError(f"scikit-learn needed for '{model_type}'.")
    params = params or {}; log.info(f"Applying ML: '{model_type}', params={params}")
    results = {'status': 'Failure', 'message': 'ML application incomplete.'}
    try:
        log.debug("Preparing features..."); input_features = _prepare_features(data, params.get('features'))
        if input_features is None: raise DataNotFoundError("Feature prep failed.")
        log.debug(f"Features shape: {input_features.shape}")
        scaler_path = params.get('scaler_path')
        if scaler_path:
            log.info(f"Loading scaler: {scaler_path}")
            if not os.path.exists(scaler_path): raise ConfigurationError(f"Scaler file not found: '{scaler_path}'")
            try: scaler = joblib.load(scaler_path); input_features = scaler.transform(input_features); log.debug("Scaler applied.")
            except Exception as e: raise AnalysisError(f"Scaler error '{scaler_path}': {e}") from e
        else: log.info("No scaler used.")
        model = None; model_path = params.get('model_path'); fit_if_missing = params.get('fit_if_missing', False)
        if model_path:
            log.info(f"Loading model: {model_path}")
            if not os.path.exists(model_path): raise ConfigurationError(f"Model file not found: '{model_path}'")
            try: model = joblib.load(model_path)
            except Exception as e: raise AnalysisError(f"Model load error '{model_path}': {e}") from e
        elif fit_if_missing: log.warning(f"Attempting fit for '{model_type}'."); model = _fit_simple_model(model_type, input_features, params);
        if model is None: raise ConfigurationError(f"Model '{model_type}' not loaded or fitted.")
        log.debug(f"Applying model '{model_type}'..."); model_output = _apply_specific_model(model, model_type, input_features, params)
        results.update(model_output); results['status'] = 'Success'; results['message'] = f"ML '{model_type}' applied successfully."; log.info(results['message'])
    except (ConfigurationError, DataNotFoundError, AnalysisError) as e: results['message'] = f"ML application failed: {e}"; log.error(results['message'])
    except Exception as e: results['message'] = f"Unexpected ML error: {e}"; log.exception(results['message'])
    return results

def _prepare_features(data: Union[pd.DataFrame, np.ndarray, Dict[Any, Any]], features_param: Optional[List[str]]) -> Optional[np.ndarray]:
    # ... (full function code) ...
    input_features = None
    if isinstance(data, np.ndarray): input_features = data; log.debug("Using numpy array input.")
    elif isinstance(data, pd.DataFrame):
        if features_param:
            missing = [c for c in features_param if c not in data.columns];
            if missing: raise ConfigurationError(f"Feature columns missing: {missing}")
            try: input_features = data[features_param].select_dtypes(include=np.number).values; log.debug(f"Extracted features: {features_param}")
            except Exception as e: raise DataNotFoundError(f"Error extracting features: {e}") from e
        else:
            log.warning("Using all numeric columns as features.");
            try: input_features = data.select_dtypes(include=np.number).values
            except Exception as e: raise DataNotFoundError(f"Could not auto-extract numeric features: {e}") from e
    else: raise DataNotFoundError(f"Unsupported ML data type: {type(data).__name__}")
    if input_features is None or input_features.size == 0: raise DataNotFoundError("Features empty after extraction.")
    if not isinstance(input_features, np.ndarray): raise AnalysisError(f"Features not numpy array: {type(input_features).__name__}.")
    if input_features.ndim == 1: input_features = input_features.reshape(1, -1)
    elif input_features.ndim != 2: raise DataNotFoundError(f"Features must be 2D, shape {input_features.shape}")
    if not np.issubdtype(input_features.dtype, np.number): raise DataNotFoundError("Features have non-numeric data.")
    if np.isnan(input_features).any(): log.warning("Features contain NaN values.")
    return input_features

def _fit_simple_model(model_type: str, X: np.ndarray, params: Dict[str, Any]) -> Optional[BaseEstimator]: # type: ignore
    # ... (full function code) ...
    model = None; model_type_lower = model_type.lower()
    if model_type_lower == 'pca' and PCA is not None:
        n_components = params.get('n_components');
        if n_components is None: raise ConfigurationError("'n_components' required to fit PCA.")
        try: model = PCA(n_components=n_components); model.fit(X); log.info(f"Fitted PCA ({model.n_components_}).")
        except Exception as e: log.error(f"PCA fit failed: {e}", exc_info=True); model = None
    elif model_type_lower == 'standardscaler' and StandardScaler is not None:
         try: model = StandardScaler(); model.fit(X); log.info("Fitted StandardScaler.")
         except Exception as e: log.error(f"Scaler fit failed: {e}", exc_info=True); model = None
    else: log.warning(f"Fitting not supported/library missing for '{model_type}'.")
    return model

def _apply_specific_model(model: Any, model_type: str, X: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    # ... (full function code) ...
    output = {}; model_type_lower = model_type.lower()
    try:
        if model_type_lower == 'pca':
            if not isinstance(model, PCA): raise TypeError("Model not PCA.")
            output['transformed_data'] = model.transform(X); output['explained_variance_ratio'] = model.explained_variance_ratio_.tolist() if hasattr(model, 'explained_variance_ratio_') else None
        elif model_type_lower == 'pls_regression':
            if not isinstance(model, PLSRegression): raise TypeError("Model not PLSRegression.")
            output['predictions'] = model.predict(X)
        elif model_type_lower in ['ann_classifier', 'mlp_classifier', 'bdt_classifier', 'rf_classifier']:
            if not (hasattr(model, 'predict')): raise TypeError(f"Model '{model_type}' lacks 'predict'.")
            output['predictions'] = model.predict(X)
            if hasattr(model, 'predict_proba'):
                try: output['probabilities'] = model.predict_proba(X).tolist()
                except Exception as e: log.warning(f"Could not get probabilities: {e}")
        elif model_type_lower == 'standardscaler':
             if not isinstance(model, StandardScaler): raise TypeError("Model not StandardScaler.")
             output['transformed_data'] = model.transform(X)
        # Add other models...
        else: raise ConfigurationError(f"Apply logic missing for '{model_type}'.")
    except AttributeError as ae: raise AnalysisError(f"Error applying '{model_type}': incompatible model object. {ae}") from ae
    except Exception as e: raise AnalysisError(f"Error during '{model_type}' execution: {e}") from e
    return output