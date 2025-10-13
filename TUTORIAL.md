# 🚀 Quick Start Tutorial

Questa guida ti porterà dall'installazione alla tua prima previsione in 5 minuti.

## 1. Installazione (2 minuti)

```bash
# Clone il repository
git clone https://github.com/yourusername/financial-forecast.git
cd financial-forecast

# Crea ambiente virtuale (opzionale ma consigliato)
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt

# Installa modelli specifici
pip install git+https://github.com/google-research/timesfm.git
```

## 2. Prima Previsione (1 minuto)

Crea un file `my_first_forecast.py`:

```python
from src.predictor import FinancialPredictor

# Inizializza con TimesFM (zero-shot)
predictor = FinancialPredictor(
    model_type="timesfm",
    mode="zero-shot"
)

# Prevedi Apple per 30 giorni
result = predictor.predict(
    ticker="AAPL",
    horizon=30
)

# Visualizza
predictor.plot_predictions()
```

Esegui:
```bash
python my_first_forecast.py
```

## 3. Comprendere l'Output

L'oggetto `result` contiene:

```python
{
    'predictions': array([180.5, 181.2, 182.0, ...]),  # 30 valori
    'dates': DatetimeIndex(['2025-01-15', ...]),
    'historical_data': array(...),                      # Dati usati
    'metrics': {
        'last_price': 179.80,
        'predicted_mean': 182.50,
        'predicted_trend': 'up'
    },
    'ticker': 'AAPL',
    'model': 'timesfm',
    'mode': 'zero-shot'
}
```

## 4. Esempi Progressivi

### Livello 1: Modifica il Ticker

```python
# Prova con altri ticker
result = predictor.predict(ticker="GOOGL", horizon=30)
result = predictor.predict(ticker="TSLA", horizon=30)
result = predictor.predict(ticker="BTC-USD", horizon=14)  # Crypto!
```

### Livello 2: Aggiungi Indicatori Tecnici

```python
result = predictor.predict(
    ticker="AAPL",
    horizon=30,
    use_technical_indicators=True  # RSI, MACD, Moving Averages
)
```

### Livello 3: Cambia Modello

```python
# Prova Lag-Llama
predictor = FinancialPredictor(
    model_type="lag_llama",  # Diverso modello
    mode="zero-shot"
)
result = predictor.predict(ticker="NVDA", horizon=30)
```

### Livello 4: Fine-Tuning

```python
# Addestra il modello sui tuoi dati specifici
predictor = FinancialPredictor(
    model_type="lag_llama",
    mode="fine-tuned"
)

result = predictor.fine_tune_and_predict(
    ticker="AAPL",
    horizon=30,
    epochs=10  # Più epoche = migliore accuracy (ma più lento)
)
```

### Livello 5: Analisi Portfolio

```python
from src.predictor import FinancialPredictor
import pandas as pd

predictor = FinancialPredictor(
    model_type="timesfm",
    mode="zero-shot"
)

# Analizza multiple azioni
portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
results = []

for ticker in portfolio:
    result = predictor.predict(ticker=ticker, horizon=30)
    
    last_price = result['historical_data'][-1, 0]
    pred_price = result['predictions'][-1]
    change_pct = ((pred_price / last_price - 1) * 100)
    
    results.append({
        'Ticker': ticker,
        'Current': f"${last_price:.2f}",
        'Predicted': f"${pred_price:.2f}",
        'Change': f"{change_pct:+.2f}%"
    })

df = pd.DataFrame(results)
print("\n📊 Portfolio Forecast Summary:")
print(df.to_string(index=False))
```

## 5. Salva e Condividi

```python
# Salva previsioni in CSV
predictor.save_predictions("my_forecast.csv")

# Salva grafico
predictor.plot_predictions(
    show_history_days=90,
    save_path="my_forecast.png"
)
```

## 6. Confronta Modelli

```python
predictor = FinancialPredictor(model_type="timesfm", mode="zero-shot")

results = predictor.compare_models(
    ticker="AAPL",
    models=["timesfm", "lag_llama"],
    horizon=30
)

# Vedi quale performa meglio!
```

## 🎯 Tips & Best Practices

1. **Zero-shot prima**: Inizia sempre con zero-shot per vedere risultati rapidi
2. **GPU consigliata**: Per fine-tuning, usa GPU se disponibile (`device="cuda"`)
3. **Orizzonte realistico**: 7-30 giorni è ottimale. 60+ giorni è più incerto
4. **Indicatori tecnici**: Aiutano ma rallentano. Usali per analisi dettagliate
5. **Validazione**: Usa train_ratio=0.8 per avere un 20% di validazione

## ⚠️ Troubleshooting

**Errore: "No data found"**
```python
# Il ticker potrebbe essere sbagliato. Verifica su Yahoo Finance
# Esempio: usa "AAPL" non "Apple"
```

**Errore: "CUDA out of memory"**
```python
# Riduci context_length o usa CPU
predictor = FinancialPredictor(
    model_type="timesfm",
    context_length=256,  # Invece di 512
    device="cpu"
)
```

**Previsioni strane/piatte**
```python
# Prova fine-tuning o aumenta i dati storici
result = predictor.predict(
    ticker="AAPL",
    start_date="2020-01-01",  # Più dati storici
    horizon=30
)
```

## 📚 Prossimi Passi

1. Esplora `examples/basic_usage.py` per esempi avanzati
2. Leggi `README.md` per documentazione completa
3. Modifica `configs/model_config.yaml` per personalizzare parametri
4. Contribuisci aggiungendo il tuo modello!

## 🆘 Supporto

- Issues: https://github.com/yourusername/financial-forecast/issues
- Documentazione: [README.md](README.md)
- Esempi: [examples/](examples/)

---

**Happy forecasting! 📈**
