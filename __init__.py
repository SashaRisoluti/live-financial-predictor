"""
Financial Time Series Forecasting Framework

Un framework modulare per previsioni finanziarie usando modelli
di time series da Hugging Face.
"""

from .predictor import FinancialPredictor
from .data.fetcher import FinancialDataFetcher
from .models.base_model import BaseTimeSeriesModel

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    "FinancialPredictor",
    "FinancialDataFetcher",
    "BaseTimeSeriesModel"
]
