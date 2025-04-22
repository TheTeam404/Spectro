# libs_engine/utils/helpers.py
import logging
log = logging.getLogger(__name__)

# Example utility function
def format_value(value, precision=3):
    """Formats a numeric value for display."""
    if value is None: return "N/A"
    try: return f"{float(value):.{precision}g}"
    except (ValueError, TypeError): return str(value)

# Add other general-purpose helper functions here