from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch
import numpy as np


class BaseTimeSeriesModel(ABC):
    """Classe base per tutti i modelli di time series."""
    
    def __init__(
        self,
        model_name: str,
        mode: str = "zero-shot",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_name: Nome del modello da Hugging Face
            mode: 'zero-shot' o 'fine-tuned'
            device: Dispositivo di computazione
        """
        self.model_name = model_name
        self.mode = mode
        self.device = device
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def load_model(self) -> None:
        """Carica il modello pre-addestrato."""
        pass
    
    @abstractmethod
    def predict(
        self,
        historical_data: np.ndarray,
        horizon: int,
        **kwargs
    ) -> np.ndarray:
        """
        Predice valori futuri.
        
        Args:
            historical_data: Dati storici (shape: [seq_len, features])
            horizon: Numero di step da prevedere
            
        Returns:
            Previsioni (shape: [horizon, features])
        """
        pass
    
    @abstractmethod
    def fine_tune(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """
        Fine-tuning del modello.
        
        Args:
            train_data: Dati di training
            val_data: Dati di validazione
            epochs: Numero di epoche
            
        Returns:
            Metriche di training
        """
        pass
    
    def save_model(self, path: str) -> None:
        """Salva il modello fine-tuned."""
        if self.mode != "fine-tuned" or not self.is_trained:
            raise ValueError("Solo i modelli fine-tuned addestrati possono essere salvati")
        torch.save(self.model.state_dict(), path)
    
    def load_finetuned(self, path: str) -> None:
        """Carica un modello fine-tuned."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.is_trained = True
        self.mode = "fine-tuned"
