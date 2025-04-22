# libs_engine/data_import/loader.py
import logging
import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from ..core._exceptions import DataLoadingError

log = logging.getLogger(__name__) # Use logger

def load_spectrum(filepath: str, file_format: Optional[str] = None, **kwargs: Any) -> Optional[pd.DataFrame]:
    """
    Loads spectral data, adding more robust parsing and logging.
    """
    # --- Determine Format ---
    if file_format is None:
        # Attempt to guess format from extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        if ext == '.csv': file_format = 'csv'
        elif ext in ['.txt', '.asc']: file_format = 'txt' # Treat asc as txt/csv flexible
        elif ext == '.spc': file_format = 'spc' # Placeholder
        else: raise DataLoadingError(f"Cannot determine file format for '{os.path.basename(filepath)}'. Please specify format.")
    else:
        file_format = file_format.lower()

    log.info(f"Attempting load: '{filepath}' (Format: {file_format})")
    log.debug(f"Loader kwargs received: {kwargs}")

    # --- Get Loader Parameters ---
    # Sensible defaults, allow overrides via kwargs
    default_column_names = ['wavelength', 'intensity']
    column_names = kwargs.get('column_names', default_column_names)
    skip_rows = kwargs.get('skip_rows', 0) # Default: assume no header unless specified
    separator = kwargs.get('separator') # Default: let pandas guess if not specified
    comment_char = kwargs.get('comment', '#')

    if len(column_names) != 2:
        raise DataLoadingError("Parameter 'column_names' must contain exactly two names.")
    if not isinstance(skip_rows, int) or skip_rows < 0:
         log.warning(f"Invalid skip_rows value ({skip_rows}), defaulting to 0.")
         skip_rows = 0

    log.debug(f"Read params: Separator='{separator or 'auto'}' SkipRows={skip_rows}, Comment='{comment_char}'")

    # --- Load Based on Format ---
    try:
        if file_format in ['csv', 'txt', 'asc']:
            # Let pandas try to guess separator if not provided
            try:
                 data = pd.read_csv(
                     filepath,
                     sep=separator, # Let pandas infer if None
                     header=None, # We handle skiprows explicitly
                     names=column_names,
                     skiprows=skip_rows,
                     comment=comment_char,
                     on_bad_lines='warn', # Warn about skipped lines
                     engine='python' # Often more flexible for weird separators
                 )
                 log.debug(f"Pandas read_csv finished. Initial shape: {data.shape}")
            except pd.errors.EmptyDataError:
                log.warning(f"File appears empty or contains only comments/header: {filepath}")
                return None # Explicitly return None for empty file after skipping
            except Exception as pd_err:
                 # Catch specific pandas errors maybe? For now, generic.
                 raise DataLoadingError(f"Pandas read_csv failed for '{filepath}': {pd_err}")


            # --- Data Cleaning and Validation ---
            if data.empty:
                 log.warning(f"DataFrame is empty after loading (before cleaning): {filepath}")
                 return None # No data read

            # Check initial types
            log.debug(f"Initial dtypes: {data.dtypes.to_dict()}")

            # Attempt numeric conversion (more robust)
            wl_col, int_col = column_names[0], column_names[1]
            data[wl_col] = pd.to_numeric(data[wl_col], errors='coerce')
            data[int_col] = pd.to_numeric(data[int_col], errors='coerce')

            # Check how many NaNs were introduced
            nan_counts = data.isna().sum()
            log.debug(f"NaN counts after to_numeric: Wavelength={nan_counts[wl_col]}, Intensity={nan_counts[int_col]}")

            # Drop rows with *any* NaN in wavelength or intensity column
            initial_rows = len(data)
            data.dropna(subset=[wl_col, int_col], inplace=True)
            rows_after_dropna = len(data)
            log.debug(f"Rows after dropna: {rows_after_dropna} (dropped {initial_rows - rows_after_dropna})")

            # Standardize column names *after* cleaning
            data.rename(columns={wl_col: 'wavelength', int_col: 'intensity'}, inplace=True)

            if data.empty:
                log.error(f"Data is empty after numeric conversion and NaN removal for '{filepath}'. Check file contents and skip_rows setting.")
                return None # Return None if *all* rows were invalid

            # Final check
            if not all(col in data.columns for col in ['wavelength', 'intensity']):
                 raise DataLoadingError("Internal error: Final DataFrame missing required columns.")

            log.info(f"Successfully loaded and cleaned {len(data)} data points from {filepath}.")
            return data[['wavelength', 'intensity']] # Return only the essential columns

        # Placeholder for other formats
        elif file_format == 'spc':
            log.warning(f"SPC file format loader not implemented yet.")
            # Add library call here, e.g., pip install spectrochempy
            # import spectrochempy as scp
            # nd = scp.read_spc(filepath)
            # df = pd.DataFrame({'wavelength': nd.x.values, 'intensity': nd.y.values})
            # return df
            return None

        else:
            # This case should have been caught by initial format check, but as safety
            raise DataLoadingError(f"Unsupported file format encountered internally: '{file_format}'")

    except FileNotFoundError:
        # Raise specific exception for API handling
        raise DataLoadingError(f"File not found at path: {filepath}") from None
    except DataLoadingError as e:
         log.error(f"DataLoadingError: {e}")
         raise # Re-raise specific loading errors
    except Exception as e:
        # Catch any other unexpected errors during loading/processing
        log.exception(f"An unexpected error occurred during loading of '{filepath}': {e}")
        raise DataLoadingError(f"Unexpected error processing file '{filepath}': {e}") from e