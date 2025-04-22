# libs_engine/__init__.py
import logging
from .core.engine import LIBSAnalysisEngine
from .core._exceptions import * # Expose exceptions
from .external.nist_fetcher import get_nist_data # Expose fetcher if available
from .analysis import * # Expose analysis functions/loaders
from .processing import * # Expose processing functions

__version__ = "0.1.0"
logging.getLogger(__name__).addHandler(logging.NullHandler()) # Avoid warnings if no handler set