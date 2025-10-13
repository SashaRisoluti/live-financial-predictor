import numpy as np
import torch
from typing import Dict, Optional
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from .base_model import BaseTimeSeriesModel


class TimesFMModel(BaseTimeSeriesModel):
    """Implementazione di Google TimesFM-2.5."""
    
    def __init__(
        self,
        mode: str = "zero-shot",
        context_length: int = 512,
        horizon: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(
            model_name="google/timesfm-2.5-base",
            mode=mode,
            device=device
        )
        self.context_length = context_length
        self.default_horizon = horizon
        self.load_model()
    
    def load_model(self) -> None:
        """Carica TimesFM da Hugging Face."""
        try:
            from timesfm import TimesFm
            
            # TimesFM ha una implementazione specifica
            self.model = TimesFm(
                context_len=self.context_length,
                horizon_len=self.default_horizon,
                backend="gpu" if self.device == "cuda" else "cpu"
            )
            self.model.load_from_checkpoint()
            print(f"✓ TimesFM caricato in modalità {self.mode}")
            
        except ImportError:
            print("⚠ timesfm non trovato, uso implementazione alternativa")
            # Fallback con transformer generico
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name
            ).to(self.device)
    
    def predict(
        self,
        historical_data: np.ndarray,
        horizon: int = None,
        frequency: str = "D",
        **kwargs
    ) -> np.ndarray:
        """
        Previsione con TimesFM.
        
        Args:
            historical_data: Serie storica (1D o 2D array)
            horizon: Step da prevedere (usa default se None)
            frequency: Frequenza temporale ('D', 'H', 'M', ecc.)
        """
        if horizon is None:
            horizon = self.default_horizon
            
        # Normalizza input
        if historical_data.ndim == 1:
            historical_data = historical_data.reshape(-1, 1)
        
        # Usa solo l'ultimo context_length
        if len(historical_data) > self.context_length:
            historical_data = historical_data[-self.context_length:]
        
        with torch.no_grad():
            # TimesFM accetta batch di serie temporali
            forecast = self.model.forecast(
                inputs=[historical_data[:, 0]],  # Prima feature
                freq=[frequency],
                horizon=[horizon]
            )
            
        return np.array(forecast[0])
    
    def fine_tune(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, float]:
        """Fine-tuning di TimesFM."""
        
        if self.mode != "fine-tuned":
            print("⚠ Passando a modalità fine-tuned")
            self.mode = "fine-tuned"
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        metrics = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            # Training loop
            self.model.train()
            train_loss = 0.0
            
            # Qui implementa il training loop specifico
            # (dipende dalla struttura esatta di TimesFM)
            
            # Validazione
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self._compute_validation_loss(val_data, criterion)
                    metrics["val_loss"].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")
        
        self.is_trained = True
        return metrics
    
    def _compute_validation_loss(
        self,
        val_data: np.ndarray,
        criterion
    ) -> float:
        """Calcola loss su validation set."""
        # Implementazione specifica
        return 0.0
