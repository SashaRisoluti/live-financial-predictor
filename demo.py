#!/usr/bin/env python3
"""
Demo Script - Financial Time Series Forecasting Framework

Questo script esegue una demo rapida del framework.
Usa dati reali di Yahoo Finance per mostrare le capacità.
"""

import sys
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.predictor import FinancialPredictor
import warnings
warnings.filterwarnings('ignore')


def print_banner(text):
    """Stampa un banner formattato."""
    width = 70
    print("\n" + "=" * width)
    print(f"{text:^{width}}")
    print("=" * width + "\n")


def demo_zero_shot():
    """Demo 1: Previsione zero-shot base."""
    print_banner("DEMO 1: Zero-Shot Prediction (TimesFM)")
    
    print("Inizializzo il modello TimesFM...")
    predictor = FinancialPredictor(
        model_type="timesfm",
        mode="zero-shot"
    )
    
    print("\n📊 Analizzando Apple (AAPL)...")
    result = predictor.predict(
        ticker="AAPL",
        horizon=30,
        use_technical_indicators=False
    )
    
    # Stampa risultati
    last_price = result['historical_data'][-1, 0]
    pred_price = result['predictions'][-1]
    change_pct = ((pred_price / last_price - 1) * 100)
    
    print("\n✓ Previsione completata!")
    print(f"  📈 Ticker: {result['ticker']}")
    print(f"  💵 Prezzo attuale: ${last_price:.2f}")
    print(f"  🔮 Previsione (30 giorni): ${pred_price:.2f}")
    print(f"  📊 Variazione attesa: {change_pct:+.2f}%")
    print(f"  🤖 Modello: {result['model']} ({result['mode']})")
    
    # Salva previsioni
    predictor.save_predictions("demo_predictions.csv")
    print("\n💾 Previsioni salvate in: demo_predictions.csv")
    
    return predictor


def demo_multiple_stocks():
    """Demo 2: Analisi di multiple azioni."""
    print_banner("DEMO 2: Portfolio Analysis")
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    predictor = FinancialPredictor(
        model_type="timesfm",
        mode="zero-shot"
    )
    
    print("📊 Analizzando portfolio tech (AAPL, MSFT, GOOGL)...\n")
    
    results_summary = []
    
    for ticker in tickers:
        print(f"  • Analizzando {ticker}...", end=" ")
        
        try:
            result = predictor.predict(
                ticker=ticker,
                horizon=30,
                use_technical_indicators=False
            )
            
            last_price = result['historical_data'][-1, 0]
            pred_price = result['predictions'][-1]
            change_pct = ((pred_price / last_price - 1) * 100)
            
            results_summary.append({
                'ticker': ticker,
                'current': last_price,
                'predicted': pred_price,
                'change': change_pct
            })
            
            print("✓")
            
        except Exception as e:
            print(f"✗ (Errore: {str(e)})")
    
    # Stampa summary
    print("\n📋 PORTFOLIO SUMMARY:")
    print("-" * 70)
    print(f"{'Ticker':<10} {'Attuale':<12} {'Previsto':<12} {'Variazione':<12}")
    print("-" * 70)
    
    for r in results_summary:
        print(f"{r['ticker']:<10} ${r['current']:<11.2f} ${r['predicted']:<11.2f} {r['change']:+.2f}%")
    
    print("-" * 70)


def demo_with_indicators():
    """Demo 3: Previsione con indicatori tecnici."""
    print_banner("DEMO 3: Technical Indicators Analysis")
    
    predictor = FinancialPredictor(
        model_type="timesfm",
        mode="zero-shot"
    )
    
    print("📊 Analizzando NVIDIA (NVDA) con indicatori tecnici...")
    result = predictor.predict(
        ticker="NVDA",
        horizon=30,
        use_technical_indicators=True  # RSI, MACD, MA
    )
    
    last_price = result['historical_data'][-1, 0]
    pred_price = result['predictions'][-1]
    change_pct = ((pred_price / last_price - 1) * 100)
    
    print("\n✓ Analisi completata con indicatori tecnici!")
    print(f"  📈 Ticker: NVDA")
    print(f"  💵 Prezzo attuale: ${last_price:.2f}")
    print(f"  🔮 Previsione (30 giorni): ${pred_price:.2f}")
    print(f"  📊 Variazione attesa: {change_pct:+.2f}%")
    print(f"  🔧 Indicatori: RSI, MACD, MA (7,30), Volatility")


def main():
    """Esegue tutte le demo."""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║       📈 FINANCIAL TIME SERIES FORECASTING FRAMEWORK 📈          ║
    ║                                                                   ║
    ║              Quick Demo - Zero-Shot Predictions                   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print("Questo script eseguirà 3 demo rapide:\n")
    print("  1. Previsione zero-shot base (AAPL)")
    print("  2. Analisi portfolio (AAPL, MSFT, GOOGL)")
    print("  3. Analisi con indicatori tecnici (NVDA)")
    print("\nNOTA: I dati vengono scaricati in tempo reale da Yahoo Finance.")
    print("      La prima esecuzione potrebbe richiedere qualche minuto.\n")
    
    try:
        # Demo 1
        demo_zero_shot()
        
        # Demo 2
        demo_multiple_stocks()
        
        # Demo 3
        demo_with_indicators()
        
        # Conclusione
        print_banner("DEMO COMPLETATA!")
        print("✓ Hai visto il framework in azione!")
        print("\n📚 Prossimi passi:")
        print("  • Leggi TUTORIAL.md per una guida completa")
        print("  • Esplora examples/basic_usage.py per esempi avanzati")
        print("  • Modifica configs/model_config.yaml per personalizzare")
        print("  • Prova il fine-tuning con mode='fine-tuned'")
        print("\n💡 Tip: Usa predictor.plot_predictions() per visualizzare grafici!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrotta dall'utente.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Errore durante la demo: {str(e)}")
        print("\nVerifica che:")
        print("  1. Hai installato tutte le dipendenze (pip install -r requirements.txt)")
        print("  2. Hai una connessione internet attiva (per Yahoo Finance)")
        print("  3. TimesFM sia installato correttamente")
        sys.exit(1)


if __name__ == "__main__":
    main()
