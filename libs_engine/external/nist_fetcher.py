# libs_engine/external/nist_fetcher.py
# (Use the robust code provided in the previous answer)
# ... includes imports, constants, get_nist_data ...
# (Paste the full code here)
import logging; log = logging.getLogger(__name__) # Add logger
from typing import Optional, List, Union
try: from astroquery.nist import Nist; from astropy.table import Table; from astropy import units as u; ASTROQUERY_AVAILABLE = True
except ImportError: ASTROQUERY_AVAILABLE = False; Nist = Table = u = None
from ..core._exceptions import DatabaseError, ConfigurationError

DEFAULT_WAVELENGTH_UNIT = u.nm if u else None; QUERY_WAVELENGTH_UNIT = u.AA if u else None

def get_nist_data(elements: List[str], min_wl: float = 200.0, max_wl: float = 900.0, wavelength_unit:u.Unit = DEFAULT_WAVELENGTH_UNIT, timeout: int = 60) -> Table: # type: ignore
    # ... (full function code) ...
    if not ASTROQUERY_AVAILABLE: raise ImportError("'astroquery' needed for NIST fetcher.")
    if not elements or not isinstance(elements, list): raise ConfigurationError("Need list of elements.")
    if not isinstance(min_wl, (int, float)) or not isinstance(max_wl, (int, float)) or min_wl >= max_wl or min_wl < 0: raise ConfigurationError(f"Invalid wavelength range: {min_wl}-{max_wl}.")
    if wavelength_unit is None or not isinstance(wavelength_unit, u.Unit) or not wavelength_unit.is_equivalent(u.m): raise ConfigurationError("Invalid wavelength unit (needs Astropy unit).")
    if not isinstance(timeout, int) or timeout <= 0: raise ConfigurationError("Timeout must be positive int.")
    try: min_wl_query = (min_wl * wavelength_unit).to(QUERY_WAVELENGTH_UNIT); max_wl_query = (max_wl * wavelength_unit).to(QUERY_WAVELENGTH_UNIT)
    except u.UnitConversionError as e: raise ConfigurationError(f"Wavelength unit conversion failed: {e}") from e
    query_payload = {'minwav': min_wl_query.value, 'maxwav': max_wl_query.value, 'unit': QUERY_WAVELENGTH_UNIT.to_string(), 'element': "|".join(elements), 'line_type': 'all', 'wavelength_type': 'observed', 'energy_level_unit': 'eV'}
    log.info(f"Querying NIST: {elements} {min_wl:.2f}-{max_wl:.2f} {wavelength_unit.to_string()}...")
    log.debug(f"NIST query payload (subset): min={query_payload['minwav']:.2f}A, max={query_payload['maxwav']:.2f}A, el='{query_payload['element']}'")
    try:
        Nist.TIMEOUT = timeout
        result_table = Nist.query(minwav=min_wl_query, maxwav=max_wl_query, linename="|".join(elements), energy_level_unit=query_payload['energy_level_unit'], wavelength_type=query_payload['wavelength_type'])
        if result_table is None: raise DatabaseError("NIST query returned None.")
        if len(result_table) == 0: log.warning(f"NIST query returned 0 lines for criteria.")
        else: log.info(f"NIST query fetched {len(result_table)} lines.")
        return result_table
    except ImportError as e: log.error(f"Astroquery import failed: {e}"); raise ImportError("NIST fetcher unavailable.") from e
    except Exception as e: error_message = f"NIST query failed: {e}"; log.exception(error_message); raise DatabaseError(error_message) from e