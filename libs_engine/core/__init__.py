# libs_engine/core/__init__.py
from .engine import LIBSAnalysisEngine, EngineState
from .data_manager import DataManager
from ._exceptions import * # Make exceptions available within core

__all__ = [
    'LIBSAnalysisEngine', 'EngineState',
    'DataManager',
    # Export specific exceptions or use * from _exceptions
    'LIBSEngineError', 'ConfigurationError', 'DataLoadingError', 'DataNotFoundError',
    'ProcessingError', 'PeakFindingError', 'PeakFittingError', 'AnalysisError', 'DatabaseError',
]