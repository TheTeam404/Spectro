# libs_engine/external/__init__.py
import logging; log = logging.getLogger(__name__)
nist_fetcher_available = False
try:
    from .nist_fetcher import get_nist_data
    nist_fetcher_available = True; log.debug("NIST Fetcher loaded.")
except ImportError as e:
    log.warning(f"Could not import NIST Fetcher: {e}")
    def get_nist_data(*args, **kwargs): raise ImportError("NIST fetcher unavailable (missing 'astroquery'?).")

log.info("External sub-package initialized.")
__all__ = ['get_nist_data'] if nist_fetcher_available else []