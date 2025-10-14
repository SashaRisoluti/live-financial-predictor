"""Utilities per gestione timestamps (Toto, Moirai)."""

import pandas as pd
import numpy as np
import torch
from typing import Tuple


def dates_to_unix_timestamps(
    dates: pd.DatetimeIndex,
    market_open_hour: int = 9,
    market_open_minute: int = 30
) -> np.ndarray:
    """
    Converti date pandas a unix timestamps.
    
    Args:
        dates: DatetimeIndex
        market_open_hour: Ora apertura mercato (default 9 AM)
        market_open_minute: Minuto apertura (default 30)
    
    Returns:
        Unix timestamps in secondi
    """
    if dates.tz is None:
        # Assumi US/Eastern se non specificato
        dates = dates.tz_localize('US/Eastern')
    
    # Se sono solo date (no time), aggiungi market open time
    if not hasattr(dates, 'hour') or dates.hour.sum() == 0:
        dates = dates + pd.Timedelta(
            hours=market_open_hour,
            minutes=market_open_minute
        )
    
    # Converti a unix timestamp
    unix_times = dates.astype(np.int64) // 10**9
    
    return unix_times


def compute_time_interval(
    dates: pd.DatetimeIndex
) -> float:
    """
    Calcola intervallo temporale medio in secondi.
    
    Args:
        dates: DatetimeIndex
    
    Returns:
        Intervallo in secondi (es. 86400 per daily)
    """
    if len(dates) < 2:
        # Default: daily
        return 86400.0
    
    # Calcola differenza media
    diffs = np.diff(dates.astype(np.int64) // 10**9)
    
    # Usa mediana per robustezza (ignora weekend, festività)
    interval = np.median(diffs)
    
    return float(interval)


def pad_timestamps(
    timestamps: np.ndarray,
    target_length: int,
    interval: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Padda timestamps a lunghezza target.
    
    Args:
        timestamps: Array unix timestamps
        target_length: Lunghezza desiderata
        interval: Intervallo temporale
    
    Returns:
        (padded_timestamps, padding_mask)
    """
    current_length = len(timestamps)
    
    if current_length >= target_length:
        return timestamps[-target_length:], np.ones(target_length, dtype=bool)
    
    # Padding
    pad_length = target_length - current_length
    
    # Crea timestamps backward
    first_timestamp = timestamps[0]
    padded_timestamps = np.array([
        first_timestamp - (pad_length - i) * interval
        for i in range(pad_length)
    ] + list(timestamps))
    
    # Mask: False per padding, True per dati reali
    mask = np.array([False] * pad_length + [True] * current_length)
    
    return padded_timestamps, mask
