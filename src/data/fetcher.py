import numpy as np
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf


class FinancialDataFetcher:
    """Recupera dati finanziari da varie API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Chiave API per servizi premium (Alpha Vantage, ecc.)
        """
        self.api_key = api_key
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        source: str = "yahoo"
    ) -> pd.DataFrame:
        """
        Recupera dati storici di un'azione.
        
        Args:
            ticker: Simbolo dell'azione (es. 'AAPL', 'GOOGL')
            start_date: Data inizio (formato 'YYYY-MM-DD')
            end_date: Data fine (formato 'YYYY-MM-DD')
            interval: Intervallo temporale ('1d', '1h', '1wk', ecc.)
            source: Fonte dati ('yahoo', 'alpha_vantage')
        
        Returns:
            DataFrame con colonne [Open, High, Low, Close, Volume]
        """
        if source == "yahoo":
            return self._fetch_yahoo(ticker, start_date, end_date, interval)
        elif source == "alpha_vantage":
            return self._fetch_alpha_vantage(ticker, start_date, end_date)
        else:
            raise ValueError(f"Fonte {source} non supportata")
    
    def _fetch_yahoo(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str
    ) -> pd.DataFrame:
        """Recupera dati da Yahoo Finance."""
        
        # Default: ultimi 2 anni
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📊 Scaricando {ticker} da Yahoo Finance...")
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        if data.empty:
            raise ValueError(f"Nessun dato trovato per {ticker}")
        
        print(f"✓ Scaricati {len(data)} record")
        return data
    
    def _fetch_alpha_vantage(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Recupera dati da Alpha Vantage (richiede API key)."""
        
        if not self.api_key:
            raise ValueError("API key richiesta per Alpha Vantage")
        
        import requests
        
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            raise ValueError(f"Errore nel recupero dati: {data.get('Note', 'Unknown error')}")
        
        # Converti in DataFrame
        ts_data = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df = df.sort_index()
        
        # Filtra per date
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df
    
    def fetch_crypto_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Recupera dati di criptovalute.
        
        Args:
            symbol: Simbolo crypto (es. 'BTC-USD', 'ETH-USD')
        """
        return self._fetch_yahoo(symbol, start_date, end_date, interval)
    
    def prepare_for_model(
        self,
        df: pd.DataFrame,
        target_column: str = "Close",
        features: Optional[list] = None
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Prepara dati per il modello.
        
        Args:
            df: DataFrame con dati finanziari
            target_column: Colonna da prevedere
            features: Lista di feature aggiuntive
        
        Returns:
            (data_array, dates) - Array numpy e indice date
        """
        if features is None:
            # Usa solo la colonna target
            data = df[target_column].values.reshape(-1, 1)
        else:
            # Usa multiple features
            data = df[features].values
        
        return data, df.index
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge indicatori tecnici comuni."""
        
        # Moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatilità
        df['Volatility'] = df['Close'].rolling(window=30).std()
        
        # Rimuovi NaN
        df = df.dropna()
        
        return df
