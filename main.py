# LIBS_Software/main.py

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)-25s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('main')

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    log.debug(f"Added project root to sys.path: {project_root}")

try:
    from waitress import serve
    WAITRESS_AVAILABLE = True
except ImportError:
    WAITRESS_AVAILABLE = False
    serve = None

try:
    from api_server import create_app
except ImportError as e:
    log.critical(f"Failed to import create_app from api_server: {e}", exc_info=True)
    sys.exit(1)
except RuntimeError as e:
    log.critical(f"RuntimeError during app creation import: {e}")
    sys.exit(1)

if __name__ == '__main__':
    log.info("=========================================")
    log.info(" LIBS Software Application - Main Entry ")
    log.info("=========================================")

    try:
        flask_app = create_app()
        if flask_app is None: raise RuntimeError("create_app() returned None.")
        log.info("Flask app created successfully.")
    except Exception as e:
        log.critical(f"Failed to create the Flask app instance: {e}", exc_info=True)
        sys.exit(1)

    # Configuration for server run
    USE_WAITRESS = False # Set True for production mode
    HOST = '127.0.0.1'
    PORT = 5000
    is_debug_mode = not USE_WAITRESS

    if USE_WAITRESS and WAITRESS_AVAILABLE:
        log.info(f"Starting production server (Waitress) on http://{HOST}:{PORT}")
        try:
            serve(flask_app, host=HOST, port=PORT, threads=8)
        except Exception as e: log.critical(f"Waitress server failed: {e}", exc_info=True); sys.exit(1)
    elif USE_WAITRESS and not WAITRESS_AVAILABLE:
        log.warning("Waitress not installed. Falling back to Flask dev server."); is_debug_mode = True

    if not USE_WAITRESS or not WAITRESS_AVAILABLE:
        log.info(f"Starting Flask dev server on http://{HOST}:{PORT} (Debug: {is_debug_mode})")
        log.warning("Flask dev server is NOT suitable for production.")
        try:
            flask_app.run(host=HOST, port=PORT, debug=is_debug_mode)
        except Exception as e: log.critical(f"Flask dev server failed: {e}", exc_info=True); sys.exit(1)

    log.info("Application server stopped.")