# VECCHIO (sostituisci tutto):
from .base_model import BaseTimeSeriesModel
from .timesfm import TimesFMModel
from .lag_llama import LagLlamaModel
from .transformer import TimeSeriesTransformer

__all__ = [
    "BaseTimeSeriesModel",
    "TimesFMModel",
    "LagLlamaModel",
    "TimeSeriesTransformer"
]

# NUOVO (sostituisci con questo):
from .base_model import BaseTimeSeriesModel
from .hf_base_model import HuggingFaceTimeSeriesModel

# Import condizionali (non generano errore se non disponibili)
try:
    from .timesfm import TimesFMModel
except ImportError:
    TimesFMModel = None

try:
    from .lag_llama import LagLlamaModel
except ImportError:
    LagLlamaModel = None

try:
    from .transformer import TimeSeriesTransformer
except ImportError:
    TimeSeriesTransformer = None

try:
    from .chronos import ChronosModel
except ImportError:
    ChronosModel = None

try:
    from .patchtst import PatchTSTModel
except ImportError:
    PatchTSTModel = None

try:
    from .tirex import TiRexModel
except ImportError:
    TiRexModel = None

try:
    from .moirai import MoiraiModel
except ImportError:
    MoiraiModel = None

__all__ = [
    "BaseTimeSeriesModel",
    "HuggingFaceTimeSeriesModel",
]

# Aggiungi dinamicamente se disponibili
if TimesFMModel:
    __all__.append("TimesFMModel")
if LagLlamaModel:
    __all__.append("LagLlamaModel")
if TimeSeriesTransformer:
    __all__.append("TimeSeriesTransformer")
if ChronosModel:
    __all__.append("ChronosModel")
if PatchTSTModel:
    __all__.append("PatchTSTModel")
if TiRexModel:
    __all__.append("TiRexModel")
if MoiraiModel:
    __all__.append("MoiraiModel")
