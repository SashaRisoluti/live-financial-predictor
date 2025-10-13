import numpy as np
import torch
from typing import Dict, Optional, List
from .base_model import BaseTimeSeriesModel


class LagLlamaModel(BaseTimeSeriesModel):
    """Implementazione di Lag-Llama per forecasting."""
    
    def __init__(
        self,
        mode: str = "zero-shot",
        context_length: int = 32,
        lags: Optional[List[int]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(
            model_name="time-series-foundation-models/Lag-Llama",
            mode=mode,
            device=device
        )
        self.context_length = context_length
        self.lags = lags or [1, 2, 3, 4, 5, 6, 7]
        self.load_model()
    
    def load_model(self) -> None:
        """Carica Lag-Llama da Hugging Face."""
        try:
            from lag_llama.gluon.estimator import LagLlamaEstimator
            from gluonts.torch import PyTorchPredictor
            
            self.estimator = LagLlamaEstimator(
                prediction_length=self.context_length,
                context_length=self.context_length,
                input_size=1,
                lags_seq=self.lags,
                num_layers=12,
                model_dim=256,
                num_heads=8,
            )
            
            # Carica checkpoint pre-addestrato
            checkpoint_path = "lag-llama.ckpt"  # da scaricare
            self.model = self.estimator.create_training_network()
            
            if self.mode == "zero-shot":
                # Carica pesi pre-addestrati
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['state_dict'])
            
            self.model = self.model.to(self.device)
            print(f"✓ Lag-Llama caricato in modalità {self.mode}")
            
        except ImportError:
            print("⚠ Installare: pip install lag-llama gluonts")
            raise
    
    def predict(
        self,
        historical_data: np.ndarray,
        horizon: int = 24,
        num_samples: int = 100,
        **kwargs
    ) -> np.ndarray:
        """
        Previsione probabilistica con Lag-Llama.
        
        Args:
            historical_data: Dati storici
            horizon: Orizzonte di previsione
            num_samples: Numero di sample per previsione probabilistica
        
        Returns:
            Mediana delle previsioni
        """
        if historical_data.ndim == 1:
            historical_data = historical_data.reshape(-1, 1)
        
        # Prepara input nel formato GluonTS
        from gluonts.dataset.common import ListDataset
        
        dataset = ListDataset(
            [{"target": historical_data[-self.context_length:, 0],
              "start": "2020-01-01"}],
            freq="D"
        )
        
        self.model.eval()
        with torch.no_grad():
            # Genera previsioni
            forecasts = []
            for _ in range(num_samples):
                forecast = self._generate_forecast(
                    historical_data[-self.context_length:],
                    horizon
                )
                forecasts.append(forecast)
            
            forecasts = np.array(forecasts)
            # Restituisci mediana
            return np.median(forecasts, axis=0)
    
    def _generate_forecast(
        self,
        context: np.ndarray,
        horizon: int
    ) -> np.ndarray:
        """Genera una singola previsione."""
        context_tensor = torch.FloatTensor(context).to(self.device)
        
        # Implementazione specifica del forward pass
        output = self.model(context_tensor.unsqueeze(0))
        
        return output.cpu().numpy().squeeze()[:horizon]
    
    def fine_tune(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, float]:
        """Fine-tuning di Lag-Llama."""
        
        if self.mode != "fine-tuned":
            self.mode = "fine-tuned"
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        metrics = {"train_loss": [], "val_loss": []}
        
        # Prepara dataset
        train_dataset = self._prepare_dataset(train_data)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch in self._create_batches(train_dataset, batch_size):
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch['past_values'])
                loss = criterion(predictions, batch['future_values'])
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_dataset)
            metrics["train_loss"].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        return metrics
    
    def _prepare_dataset(self, data: np.ndarray) -> List[Dict]:
        """Prepara dataset per training."""
        dataset = []
        for i in range(len(data) - self.context_length - 24):
            dataset.append({
                'past_values': data[i:i+self.context_length],
                'future_values': data[i+self.context_length:i+self.context_length+24]
            })
        return dataset
    
    def _create_batches(self, dataset: List[Dict], batch_size: int):
        """Crea batch per training."""
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            yield {
                'past_values': torch.FloatTensor(
                    [b['past_values'] for b in batch]
                ).to(self.device),
                'future_values': torch.FloatTensor(
                    [b['future_values'] for b in batch]
                ).to(self.device)
            }
