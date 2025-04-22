# libs_engine/processing/__init__.py
from .smoothing import smooth_spectrum
from .peak_detection import find_peaks
from .peak_fitting import fit_peaks_detailed, FIT_FUNCTIONS # Expose fit functions if needed externally

__all__ = ['smooth_spectrum', 'find_peaks', 'fit_peaks_detailed', 'FIT_FUNCTIONS']