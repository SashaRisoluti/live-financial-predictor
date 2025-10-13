# 📈 Financial Time Series Forecasting Framework

Framework flessibile e modulare per previsioni finanziarie usando modelli di time series all'avanguardia da Hugging Face.

## 🎯 Caratteristiche

- **Multiple architetture**: TimesFM (Google), Lag-Llama, Time Series Transformer
- **Modalità flessibili**: Zero-shot o Fine-tuned
- **Dati in tempo reale**: Integrazione con Yahoo Finance e Alpha Vantage
- **Indicatori tecnici**: RSI, MACD, Moving Averages, Volatility
- **Visualizzazioni**: Grafici professionali e comparazioni tra modelli
- **Facile da estendere**: Architettura modulare per aggiungere nuovi modelli

## 🚀 Quick Start

### Installazione

```bash
# Clone repository
git clone https://github.com/your-username/financial-forecast.git
cd financial-forecast

# Installa dipendenze
pip install -r requirements.txt

# Installa TimesFM (Google)
pip install git+https://github.com/google-research/timesfm.git

# Installa Lag-Llama
pip install git+https://github.com/time-series-foundation-models/lag-llama.git
```

### Uso Base

```python
from src.predictor import FinancialPredictor

# Inizializza predictor
predictor = FinancialPredictor(
    model_type="timesfm",  # o "lag_llama", "transformer"
    mode="zero-shot"       # o "fine-tuned"
)

# Genera previsioni
result = predictor.predict(
    ticker="AAPL",
    horizon=30  # giorni da prevedere
)

# Visualizza risultati
predictor.plot_predictions()
```

## 📚 Esempi Completi

### 1. Previsione Zero-Shot

```python
from src.predictor import FinancialPredictor

# Zero-shot: usa il modello pre-addestrato senza training
predictor = FinancialPredictor(
    model_type="timesfm",
    mode="zero-shot"
)

result = predictor.predict(
    ticker="AAPL",
    horizon=30,
    use_technical_indicators=True  # Aggiunge RSI, MACD, ecc.
)

# Salva previsioni
predictor.save_predictions("predictions_aapl.csv")

# Visualizza
predictor.plot_predictions(
    show_history_days=90,
    save_path="aapl_forecast.png"
)
```

### 2. Fine-Tuning su Dati Specifici

```python
# Fine-tuning: addestra il modello sui tuoi dati
predictor = FinancialPredictor(
    model_type="lag_llama",
    mode="fine-tuned"
)

result = predictor.fine_tune_and_predict(
    ticker="TSLA",
    horizon=30,
    epochs=10,
    learning_rate=1e-4,
    train_ratio=0.8  # 80% training, 20% validation
)
```

### 3. Analisi Portfolio

```python
tickers = ["AAPL", "GOOGL", "MSFT", "NVDA"]

predictor = FinancialPredictor(
    model_type="timesfm",
    mode="zero-shot"
)

results = {}
for ticker in tickers:
    result = predictor.predict(
        ticker=ticker,
        horizon=30,
        use_technical_indicators=True
    )
    results[ticker] = result
    
    # Stampa sommario
    last_price = result['historical_data'][-1, 0]
    pred_price = result['predictions'][-1]
    change_pct = ((pred_price / last_price - 1) * 100)
    
    print(f"{ticker}:")
    print(f"  Attuale: ${last_price:.2f}")
    print(f"  Previsione (30gg): ${pred_price:.2f} ({change_pct:+.2f}%)")
```

### 4. Criptovalute

```python
predictor = FinancialPredictor(
    model_type="timesfm",
    mode="zero-shot"
)

result = predictor.predict(
    ticker="BTC-USD",  # Bitcoin
    horizon=14,
    use_technical_indicators=True
)

# Funziona anche con: ETH-USD, BNB-USD, etc.
```

### 5. Confronto Modelli

```python
predictor = FinancialPredictor(
    model_type="timesfm",
    mode="zero-shot"
)

results = predictor.compare_models(
    ticker="AAPL",
    models=["timesfm", "lag_llama"],
    horizon=30
)

# Visualizza confronto
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
for model_name, result in results.items():
    plt.plot(result['dates'], result['predictions'], 
             label=model_name, linewidth=2)
plt.legend()
plt.show()
```

## 🎨 Struttura del Progetto

```
financial-forecast/
├── src/
│   ├── models/
│   │   ├── base_model.py      # Classe astratta base
│   │   ├── timesfm.py         # TimesFM di Google
│   │   ├── lag_llama.py       # Lag-Llama
│   │   └── transformer.py     # Time Series Transformer
│   ├── data/
│   │   ├── fetcher.py         # Recupero dati da API
│   │   └── preprocessor.py    # Preprocessing
│   └── predictor.py           # Orchestratore principale
├── examples/
│   └── basic_usage.py         # Esempi di utilizzo
├── configs/
│   └── model_config.yaml      # Configurazioni
└── tests/
    └── test_models.py         # Unit tests
```

## 🔧 Modelli Supportati

### TimesFM (Google Research)
- **Contesto**: 512 time steps
- **Orizzonte**: Fino a 128 step
- **Caratteristiche**: Addestrato su miliardi di time series
- **Migliore per**: Previsioni zero-shot, alta accuracy

### Lag-Llama
- **Contesto**: 32 time steps
- **Caratteristiche**: Usa lag features, architettura basata su LLaMA
- **Migliore per**: Fine-tuning rapido, interpretabilità

### Time Series Transformer
- **Architettura**: Encoder-decoder transformer
- **Caratteristiche**: Attenzione temporale
- **Migliore per**: Dati multivariate

## 📊 Output del Modello

Il metodo `predict()` restituisce un dizionario con:

```python
{
    'predictions': np.ndarray,      # Previsioni future
    'dates': pd.DatetimeIndex,      # Date delle previsioni
    'historical_data': np.ndarray,  # Dati storici usati
    'historical_dates': pd.DatetimeIndex,
    'metrics': dict,                # Metriche di valutazione
    'ticker': str,
    'model': str,
    'mode': str
}
```

## 🎯 Opzioni Avanzate

### Indicatori Tecnici Automatici

```python
result = predictor.predict(
    ticker="AAPL",
    horizon=30,
    use_technical_indicators=True  # Aggiunge automaticamente:
                                   # - Moving Averages (7, 30 giorni)
                                   # - RSI (14 giorni)
                                   # - MACD
                                   # - Volatility
)
```

### Personalizzazione Modello

```python
predictor = FinancialPredictor(
    model_type="timesfm",
    mode="zero-shot",
    context_length=512,    # Lunghezza contesto
    horizon=128,           # Orizzonte default
    device="cuda"          # Usa GPU
)
```

### Dati Personalizzati

```python
result = predictor.predict(
    ticker="AAPL",
    start_date="2020-01-01",  # Inizio periodo storico
    end_date="2024-12-31",    # Fine periodo storico
    data_source="yahoo"        # o "alpha_vantage"
)
```

## 🧪 Testing

```bash
# Esegui tests
python -m pytest tests/

# Esegui esempi
python examples/basic_usage.py
```

## 📝 Aggiungere un Nuovo Modello

1. Crea una classe che eredita da `BaseTimeSeriesModel`:

```python
from src.models.base_model import BaseTimeSeriesModel

class MyCustomModel(BaseTimeSeriesModel):
    def __init__(self, mode="zero-shot", **kwargs):
        super().__init__(
            model_name="your-model-name",
            mode=mode,
            **kwargs
        )
        self.load_model()
    
    def load_model(self):
        # Carica il tuo modello
        pass
    
    def predict(self, historical_data, horizon, **kwargs):
        # Implementa la logica di previsione
        pass
    
    def fine_tune(self, train_data, val_data, epochs, **kwargs):
        # Implementa il fine-tuning
        pass
```

2. Registra il modello in `FinancialPredictor`:

```python
AVAILABLE_MODELS = {
    "timesfm": TimesFMModel,
    "lag_llama": LagLlamaModel,
    "my_model": MyCustomModel  # Aggiungi qui
}
```

## 🤝 Contribuire

Contributi benvenuti! Per aggiungere nuovi modelli o features:

1. Fork del repository
2. Crea un branch (`git checkout -b feature/nuova-feature`)
3. Commit delle modifiche (`git commit -am 'Aggiunge nuova feature'`)
4. Push al branch (`git push origin feature/nuova-feature`)
5. Apri una Pull Request

## 📄 Licenza

MIT License - vedi [LICENSE](LICENSE) per dettagli

## 🙏 Ringraziamenti

- [Google Research](https://github.com/google-research/timesfm) per TimesFM
- [Time Series Foundation Models](https://github.com/time-series-foundation-models/lag-llama) per Lag-Llama
- [Hugging Face](https://huggingface.co/) per l'ecosistema di modelli

## 📧 Contatti

Per domande o supporto, apri una issue su GitHub.

## ⚠️ Disclaimer

Questo framework è solo per scopi educativi e di ricerca. Le previsioni finanziarie non costituiscono consulenza finanziaria. Investire comporta rischi e si possono perdere capitali.

---

**Made with ❤️ for the financial ML community**
