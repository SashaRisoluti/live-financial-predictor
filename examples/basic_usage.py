"""
Esempi di utilizzo del Financial Time Series Forecasting Framework
"""

from src.predictor import FinancialPredictor


def example_1_zero_shot_prediction():
    """Esempio 1: Previsione zero-shot con TimesFM."""
    
    print("=" * 70)
    print("ESEMPIO 1: Previsione Zero-Shot con TimesFM")
    print("=" * 70)
    
    # Inizializza predictor
    predictor = FinancialPredictor(
        model_type="timesfm",
        mode="zero-shot"
    )
    
    # Genera previsioni
    result = predictor.predict(
        ticker="AAPL",
        horizon=30,  # 30 giorni
        use_technical_indicators=False
    )
    
    # Visualizza
    predictor.plot_predictions(show_history_days=90)
    
    # Salva risultati
    predictor.save_predictions("predictions_aapl_timesfm.csv")
    
    return result


def example_2_fine_tuned_prediction():
    """Esempio 2: Fine-tuning su dati specifici."""
    
    print("\n" + "=" * 70)
    print("ESEMPIO 2: Fine-Tuning su Tesla")
    print("=" * 70)
    
    predictor = FinancialPredictor(
        model_type="lag_llama",
        mode="fine-tuned"
    )
    
    # Fine-tune e predici
    result = predictor.fine_tune_and_predict(
        ticker="TSLA",
        horizon=30,
        epochs=10,
        learning_rate=1e-4,
        train_ratio=0.8
    )
    
    predictor.plot_predictions()
    
    return result


def example_3_multiple_stocks():
    """Esempio 3: Analisi di multiple azioni."""
    
    print("\n" + "=" * 70)
    print("ESEMPIO 3: Analisi Portfolio Tech")
    print("=" * 70)
    
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    predictor = FinancialPredictor(
        model_type="timesfm",
        mode="zero-shot"
    )
    
    results = {}
    for ticker in tickers:
        print(f"\n📊 Analizzando {ticker}...")
        result = predictor.predict(
            ticker=ticker,
            horizon=30,
            use_technical_indicators=True
        )
        results[ticker] = result
        
        # Stampa sommario
        last_price = result['historical_data'][-1, 0]
        pred_price = result['predictions'][-1]
        change = ((pred_price / last_price - 1) * 100)
        
        print(f"  Prezzo attuale: ${last_price:.2f}")
        print(f"  Previsione 30gg: ${pred_price:.2f}")
        print(f"  Variazione: {change:+.2f}%")
    
    return results


def example_4_crypto_prediction():
    """Esempio 4: Previsione criptovalute."""
    
    print("\n" + "=" * 70)
    print("ESEMPIO 4: Previsione Bitcoin")
    print("=" * 70)
    
    predictor = FinancialPredictor(
        model_type="timesfm",
        mode="zero-shot"
    )
    
    result = predictor.predict(
        ticker="BTC-USD",
        horizon=14,  # 2 settimane
        use_technical_indicators=True
    )
    
    predictor.plot_predictions(show_history_days=180)
    
    return result


def example_5_compare_models():
    """Esempio 5: Confronto tra modelli."""
    
    print("\n" + "=" * 70)
    print("ESEMPIO 5: Confronto TimesFM vs Lag-Llama")
    print("=" * 70)
    
    predictor = FinancialPredictor(
        model_type="timesfm",  # Verrà sovrascritto
        mode="zero-shot"
    )
    
    results = predictor.compare_models(
        ticker="AAPL",
        models=["timesfm", "lag_llama"],
        horizon=30
    )
    
    # Confronta risultati
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 7))
    
    for model_name, result in results.items():
        plt.plot(result['dates'], result['predictions'], 
                label=model_name, linewidth=2)
    
    plt.title("Confronto Modelli - AAPL", fontsize=16)
    plt.xlabel("Data")
    plt.ylabel("Prezzo ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results


def example_6_custom_options():
    """Esempio 6: Opzioni avanzate."""
    
    print("\n" + "=" * 70)
    print("ESEMPIO 6: Configurazione Avanzata")
    print("=" * 70)
    
    # Configurazione personalizzata
    predictor = FinancialPredictor(
        model_type="timesfm",
        mode="zero-shot",
        context_length=512,  # Parametro specifico del modello
        horizon=128,
        device="cuda"  # Usa GPU se disponibile
    )
    
    result = predictor.predict(
        ticker="NVDA",
        horizon=60,
        start_date="2023-01-01",  # Dati da gennaio 2023
        use_technical_indicators=True,
        data_source="yahoo"
    )
    
    predictor.plot_predictions(
        show_history_days=180,
        save_path="nvda_prediction.png"
    )
    
    return result


if __name__ == "__main__":
    # Esegui esempi
    
    # Esempio base
    example_1_zero_shot_prediction()
    
    # Decommentare per eseguire altri esempi:
    
    # example_2_fine_tuned_prediction()
    # example_3_multiple_stocks()
    # example_4_crypto_prediction()
    # example_5_compare_models()
    # example_6_custom_options()
