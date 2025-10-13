import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from .models.base_model import BaseTimeSeriesModel
from .models.timesfm import TimesFMModel
from .models.lag_llama import LagLlamaModel
from .models.transformer import TimeSeriesTransformer
from .data.fetcher import FinancialDataFetcher


class FinancialPredictor:
    """
    Framework principale per previsioni finanziarie con modelli time series.
    
    Esempio:
        predictor = FinancialPredictor(
            model_type="timesfm",
            mode="zero-shot"
        )
        
        predictions = predictor.predict(
            ticker="AAPL",
            horizon=30
        )
    """
    
    AVAILABLE_MODELS = {
        "timesfm": TimesFMModel,
        "lag_llama": LagLlamaModel,
        "transformer": TimeSeriesTransformer,
    }
    
    def __init__(
        self,
        model_type: str = "timesfm",
        mode: str = "zero-shot",
        api_key: Optional[str] = None,
        device: str = "auto",
        **model_kwargs
    ):
        """
        Inizializza il predictor.
        
        Args:
            model_type: Tipo di modello ('timesfm', 'lag_llama', 'transformer')
            mode: Modalità ('zero-shot' o 'fine-tuned')
            api_key: Chiave API per dati finanziari
            device: Device di computazione ('cuda', 'cpu', 'auto')
            **model_kwargs: Parametri aggiuntivi per il modello
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Modello {model_type} non supportato. "
                f"Scegli tra: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        if mode not in ["zero-shot", "fine-tuned"]:
            raise ValueError("mode deve essere 'zero-shot' o 'fine-tuned'")
        
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Inizializza componenti
        self.model_type = model_type
        self.mode = mode
        self.data_fetcher = FinancialDataFetcher(api_key=api_key)
        
        # Carica modello
        print(f"🚀 Inizializzo {model_type} in modalità {mode}...")
        model_class = self.AVAILABLE_MODELS[model_type]
        self.model: BaseTimeSeriesModel = model_class(
            mode=mode,
            device=device,
            **model_kwargs
        )
        
        self.historical_data = None
        self.predictions = None
        self.ticker = None
    
    def predict(
        self,
        ticker: str,
        horizon: int = 30,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_technical_indicators: bool = False,
        data_source: str = "yahoo",
        return_confidence: bool = False,
        **predict_kwargs
    ) -> Dict:
        """
        Predice valori futuri per un'azione.
        
        Args:
            ticker: Simbolo dell'azione (es. 'AAPL', 'TSLA')
            horizon: Giorni da prevedere
            start_date: Data inizio dati storici
            end_date: Data fine dati storici
            use_technical_indicators: Se True, aggiunge indicatori tecnici
            data_source: Fonte dati ('yahoo', 'alpha_vantage')
            return_confidence: Se True, include intervalli di confidenza
            
        Returns:
            Dict con 'predictions', 'dates', 'historical_data', 'metrics'
        """
        self.ticker = ticker
        
        # 1. Scarica dati
        print(f"\n📈 Analizzando {ticker}...")
        df = self.data_fetcher.fetch_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            source=data_source
        )
        
        # 2. Aggiungi indicatori tecnici se richiesto
        if use_technical_indicators:
            print("🔧 Aggiungendo indicatori tecnici...")
            df = self.data_fetcher.add_technical_indicators(df)
        
        # 3. Prepara dati
        data, dates = self.data_fetcher.prepare_for_model(df, target_column="Close")
        self.historical_data = data
        self.historical_dates = dates
        
        # 4. Genera previsioni
        print(f"🔮 Generando previsioni per {horizon} giorni...")
        predictions = self.model.predict(
            historical_data=data,
            horizon=horizon,
            **predict_kwargs
        )
        
        # 5. Crea date future
        last_date = dates[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        self.predictions = predictions
        self.future_dates = future_dates
        
        # 6. Calcola metriche su validation set
        metrics = self._calculate_metrics(data, predictions)
        
        result = {
            'predictions': predictions,
            'dates': future_dates,
            'historical_data': data,
            'historical_dates': dates,
            'metrics': metrics,
            'ticker': ticker,
            'model': self.model_type,
            'mode': self.mode
        }
        
        print(f"✓ Previsioni completate!")
        print(f"  - Ultimo prezzo: ${data[-1, 0]:.2f}")
        print(f"  - Previsione a {horizon}gg: ${predictions[-1]:.2f}")
        print(f"  - Variazione: {((predictions[-1] / data[-1, 0] - 1) * 100):.2f}%")
        
        return result
    
    def fine_tune_and_predict(
        self,
        ticker: str,
        horizon: int = 30,
        train_ratio: float = 0.8,
        epochs: int = 10,
        **kwargs
    ) -> Dict:
        """
        Fine-tune del modello e previsione.
        
        Args:
            ticker: Simbolo dell'azione
            horizon: Giorni da prevedere
            train_ratio: Rapporto dati training/validation
            epochs: Epoche di training
        """
        # Scarica dati
        df = self.data_fetcher.fetch_stock_data(ticker=ticker)
        data, dates = self.data_fetcher.prepare_for_model(df)
        
        # Split train/val
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Fine-tuning
        print(f"\n🎯 Fine-tuning su {ticker}...")
        metrics = self.model.fine_tune(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            **kwargs
        )
        
        # Predici
        return self.predict(ticker=ticker, horizon=horizon)
    
    def plot_predictions(
        self,
        show_history_days: int = 90,
        figsize: Tuple[int, int] = (14, 7),
        save_path: Optional[str] = None
    ) -> None:
        """Visualizza previsioni con dati storici."""
        
        if self.predictions is None:
            raise ValueError("Esegui prima predict()")
        
        # Prendi ultimi N giorni di storia
        hist_data = self.historical_data[-show_history_days:]
        hist_dates = self.historical_dates[-show_history_days:]
        
        plt.figure(figsize=figsize)
        
        # Dati storici
        plt.plot(hist_dates, hist_data[:, 0], 
                label='Storico', linewidth=2, color='#2E86AB')
        
        # Previsioni
        plt.plot(self.future_dates, self.predictions,
                label=f'Previsione ({self.model_type})', 
                linewidth=2, color='#A23B72', linestyle='--')
        
        # Punto di connessione
        plt.plot([hist_dates[-1], self.future_dates[0]],
                [hist_data[-1, 0], self.predictions[0]],
                color='gray', linestyle=':', alpha=0.5)
        
        plt.axvline(x=hist_dates[-1], color='gray', 
                   linestyle=':', alpha=0.3, label='Oggi')
        
        plt.title(f'{self.ticker} - Previsione {self.mode}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Prezzo ($)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Grafico salvato in {save_path}")
        
        plt.show()
    
    def _calculate_metrics(
        self,
        historical_data: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calcola metriche di valutazione."""
        
        # Usa ultimi N punti come validation
        val_size = min(len(predictions), 30)
        if len(historical_data) < val_size:
            return {}
        
        actual = historical_data[-val_size:, 0]
        
        # Semplici metriche
        metrics = {
            'mean_historical': float(np.mean(historical_data[:, 0])),
            'std_historical': float(np.std(historical_data[:, 0])),
            'last_price': float(historical_data[-1, 0]),
            'predicted_mean': float(np.mean(predictions)),
            'predicted_trend': 'up' if predictions[-1] > predictions[0] else 'down'
        }
        
        return metrics
    
    def save_predictions(self, path: str) -> None:
        """Salva previsioni in CSV."""
        if self.predictions is None:
            raise ValueError("Nessuna previsione da salvare")
        
        df = pd.DataFrame({
            'Date': self.future_dates,
            'Predicted_Close': self.predictions,
            'Ticker': self.ticker,
            'Model': self.model_type,
            'Mode': self.mode
        })
        
        df.to_csv(path, index=False)
        print(f"💾 Previsioni salvate in {path}")
    
    def compare_models(
        self,
        ticker: str,
        models: List[str],
        horizon: int = 30,
        **kwargs
    ) -> Dict:
        """Confronta multiple modelli sullo stesso ticker."""
        
        results = {}
        
        for model_type in models:
            print(f"\n{'='*50}")
            print(f"Testing {model_type}...")
            print('='*50)
            
            predictor = FinancialPredictor(
                model_type=model_type,
                mode=self.mode,
                **kwargs
            )
            
            result = predictor.predict(ticker=ticker, horizon=horizon)
            results[model_type] = result
        
        return results
