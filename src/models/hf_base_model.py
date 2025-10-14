"""Classe base per modelli HuggingFace standard."""

from abc import abstractmethod
from typing import Dict, Optional
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from .base_model import BaseTimeSeriesModel


class HuggingFaceTimeSeriesModel(BaseTimeSeriesModel):
    """
    Classe base per modelli caricabili da HuggingFace Hub.
    
    Sottoclassi devono implementare:
    - _preprocess_input()
    - _postprocess_output()
    """
    
    def __init__(
        self,
        model_name: str,
        mode: str = "zero-shot",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(model_name, mode, device)
        self.tokenizer = None
    
    def load_model(self) -> None:
        """Carica modello da HF Hub."""
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            ).to(self.device)
            
            # Alcuni modelli hanno tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name
                )
            except:
                self.tokenizer = None
            
            print(f"✓ {self.model_name} caricato in modalità {self.mode}")
            
        except Exception as e:
            raise ImportError(
                f"Errore caricamento {self.model_name}: {e}\n"
                f"Installa con: pip install transformers"
            )
    
    @abstractmethod
    def _preprocess_input(
        self,
        historical_data: np.ndarray,
        **kwargs
    ) -> Dict:
        """Preprocessa input nel formato specifico del modello."""
        pass
    
    @abstractmethod
    def _postprocess_output(
        self,
        model_output,
        horizon: int,
        **kwargs
    ) -> np.ndarray:
        """Postprocessa output nel formato standard [horizon]."""
        pass
    
    def predict(
        self,
        historical_data: np.ndarray,
        horizon: int,
        **kwargs
    ) -> np.ndarray:
        """Template method per previsione."""
        
        # Preprocess
        model_input = self._preprocess_input(historical_data, **kwargs)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            output = self._forward(model_input, horizon)
        
        # Postprocess
        predictions = self._postprocess_output(output, horizon)
        
        return predictions
    
    @abstractmethod
    def _forward(self, model_input: Dict, horizon: int):
        """Forward pass specifico del modello."""
        pass
