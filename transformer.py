import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from .base_model import BaseTimeSeriesModel


class TimeSeriesTransformer(BaseTimeSeriesModel):
    """
    Implementazione di un Transformer per time series.
    Basato sull'architettura standard encoder-decoder.
    """
    
    def __init__(
        self,
        mode: str = "zero-shot",
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        context_length: int = 96,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(
            model_name="time-series-transformer",
            mode=mode,
            device=device
        )
        
        self.d_model = d_model
        self.nhead = nhead
        self.context_length = context_length
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        self.load_model()
    
    def load_model(self) -> None:
        """Costruisce il modello Transformer."""
        
        # Input embedding
        self.embedding = nn.Linear(1, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, 1)
        
        self.model = nn.ModuleDict({
            'embedding': self.embedding,
            'pos_encoder': self.pos_encoder,
            'transformer': self.transformer,
            'output_projection': self.output_projection
        })
        
        self.model = self.model.to(self.device)
        
        if self.mode == "zero-shot":
            # Se hai checkpoint pre-addestrati, caricali qui
            try:
                checkpoint = torch.load(
                    "transformer_pretrained.pt",
                    map_location=self.device
                )
                self.model.load_state_dict(checkpoint)
                print("✓ Caricato checkpoint pre-addestrato")
            except FileNotFoundError:
                print("⚠ Nessun checkpoint trovato, uso inizializzazione random")
        
        print(f"✓ Transformer caricato in modalità {self.mode}")
    
    def predict(
        self,
        historical_data: np.ndarray,
        horizon: int = 24,
        **kwargs
    ) -> np.ndarray:
        """
        Previsione autoregressive con Transformer.
        
        Args:
            historical_data: Dati storici (shape: [seq_len, features])
            horizon: Step da prevedere
        
        Returns:
            Previsioni (shape: [horizon])
        """
        if historical_data.ndim == 1:
            historical_data = historical_data.reshape(-1, 1)
        
        # Usa solo l'ultimo context
        if len(historical_data) > self.context_length:
            historical_data = historical_data[-self.context_length:]
        
        # Normalizza
        mean = historical_data.mean()
        std = historical_data.std() + 1e-8
        historical_data_norm = (historical_data - mean) / std
        
        # Converti a tensor
        src = torch.FloatTensor(historical_data_norm).to(self.device)
        src = src.unsqueeze(0)  # [1, seq_len, 1]
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Encoding della sequenza di input
            src_embedded = self.model['embedding'](src)
            src_embedded = self.model['pos_encoder'](src_embedded)
            memory = self.model['transformer'].encoder(src_embedded)
            
            # Predizione autoregressive
            tgt = src[:, -1:, :]  # Ultimo valore come inizio
            
            for _ in range(horizon):
                # Embedding target
                tgt_embedded = self.model['embedding'](tgt)
                tgt_embedded = self.model['pos_encoder'](tgt_embedded)
                
                # Decoder forward
                output = self.model['transformer'].decoder(
                    tgt_embedded,
                    memory
                )
                
                # Proiezione output
                next_pred = self.model['output_projection'](output[:, -1:, :])
                predictions.append(next_pred.item())
                
                # Aggiungi al target per prossima iterazione
                tgt = torch.cat([tgt, next_pred], dim=1)
        
        # Denormalizza
        predictions = np.array(predictions) * std + mean
        
        return predictions
    
    def fine_tune(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        horizon: int = 24,
        **kwargs
    ) -> Dict[str, float]:
        """
        Fine-tuning del Transformer.
        
        Args:
            train_data: Dati di training
            val_data: Dati di validazione
            epochs: Numero di epoche
            learning_rate: Learning rate
            batch_size: Batch size
            horizon: Orizzonte di previsione
        """
        if self.mode != "fine-tuned":
            self.mode = "fine-tuned"
        
        # Prepara dataset
        train_dataset = self._create_sequences(
            train_data,
            self.context_length,
            horizon
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        metrics = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            for i in range(0, len(train_dataset), batch_size):
                batch = train_dataset[i:i+batch_size]
                
                if len(batch) == 0:
                    continue
                
                # Prepara batch
                src_batch = torch.FloatTensor(
                    [item['src'] for item in batch]
                ).to(self.device)
                
                tgt_batch = torch.FloatTensor(
                    [item['tgt'] for item in batch]
                ).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                src_embedded = self.model['embedding'](src_batch)
                src_embedded = self.model['pos_encoder'](src_embedded)
                
                tgt_embedded = self.model['embedding'](tgt_batch[:, :-1, :])
                tgt_embedded = self.model['pos_encoder'](tgt_embedded)
                
                # Transformer forward
                output = self.model['transformer'](src_embedded, tgt_embedded)
                predictions = self.model['output_projection'](output)
                
                # Loss
                loss = criterion(predictions, tgt_batch[:, 1:, :])
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / max(num_batches, 1)
            metrics["train_loss"].append(avg_train_loss)
            
            # Validazione
            if val_data is not None:
                val_loss = self._validate(val_data, criterion, horizon)
                metrics["val_loss"].append(val_loss)
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}")
        
        self.is_trained = True
        return metrics
    
    def _create_sequences(
        self,
        data: np.ndarray,
        context_length: int,
        horizon: int
    ) -> list:
        """Crea sequenze per training."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Normalizza
        mean = data.mean()
        std = data.std() + 1e-8
        data_norm = (data - mean) / std
        
        sequences = []
        for i in range(len(data) - context_length - horizon):
            src = data_norm[i:i+context_length]
            tgt = data_norm[i+context_length:i+context_length+horizon]
            
            # Aggiungi start token al target
            tgt_input = np.vstack([src[-1:], tgt])
            
            sequences.append({
                'src': src,
                'tgt': tgt_input
            })
        
        return sequences
    
    def _validate(
        self,
        val_data: np.ndarray,
        criterion: nn.Module,
        horizon: int
    ) -> float:
        """Valida il modello."""
        val_dataset = self._create_sequences(
            val_data,
            self.context_length,
            horizon
        )
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for item in val_dataset:
                src = torch.FloatTensor(item['src']).unsqueeze(0).to(self.device)
                tgt = torch.FloatTensor(item['tgt']).unsqueeze(0).to(self.device)
                
                src_embedded = self.model['embedding'](src)
                src_embedded = self.model['pos_encoder'](src_embedded)
                
                tgt_embedded = self.model['embedding'](tgt[:, :-1, :])
                tgt_embedded = self.model['pos_encoder'](tgt_embedded)
                
                output = self.model['transformer'](src_embedded, tgt_embedded)
                predictions = self.model['output_projection'](output)
                
                loss = criterion(predictions, tgt[:, 1:, :])
                total_loss += loss.item()
        
        return total_loss / len(val_dataset)


class PositionalEncoding(nn.Module):
    """Positional encoding per Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
