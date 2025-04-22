# libs_engine/core/_exceptions.py
class LIBSEngineError(Exception): pass
class ConfigurationError(LIBSEngineError): pass
class DataLoadingError(LIBSEngineError): pass
class DataNotFoundError(LIBSEngineError): pass
class ProcessingError(LIBSEngineError): pass
class PeakFindingError(ProcessingError): pass
class PeakFittingError(ProcessingError): pass
class AnalysisError(LIBSEngineError): pass
class DatabaseError(LIBSEngineError): pass