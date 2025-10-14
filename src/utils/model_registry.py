"""Registry centralizzato per tutti i modelli."""

from typing import Dict, Type, Optional
from ..models.base_model import BaseTimeSeriesModel


class ModelRegistry:
    """
    Registry per gestire tutti i modelli disponibili.
    
    Permette di:
    - Registrare nuovi modelli facilmente
    - Gestire dipendenze opzionali
    - Lazy loading
    """
    
    _registry: Dict[str, Dict] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[BaseTimeSeriesModel],
        category: str,
        requires: Optional[list] = None
    ):
        """Registra un modello."""
        cls._registry[name] = {
            'class': model_class,
            'category': category,  # 'hf_standard' | 'custom_api'
            'requires': requires or []
        }
    
    @classmethod
    def create(cls, name: str, mode: str, **kwargs) -> BaseTimeSeriesModel:
        """Factory method per creare modelli."""
        
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Modello '{name}' non trovato. "
                f"Disponibili: {available}"
            )
        
        model_info = cls._registry[name]
        
        # Verifica dipendenze
        cls._check_dependencies(name, model_info['requires'])
        
        # Crea modello
        model_class = model_info['class']
        return model_class(mode=mode, **kwargs)
    
    @classmethod
    def _check_dependencies(cls, name: str, requires: list):
        """Verifica che dipendenze siano installate."""
        missing = []
        
        for package in requires:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            raise ImportError(
                f"Modello '{name}' richiede: {', '.join(missing)}\n"
                f"Installa con: pip install {' '.join(missing)}"
            )
    
    @classmethod
    def list_models(cls) -> Dict:
        """Lista tutti i modelli disponibili."""
        result = {}
        
        for name, info in cls._registry.items():
            # Check se disponibile
            try:
                cls._check_dependencies(name, info['requires'])
                available = True
            except ImportError:
                available = False
            
            result[name] = {
                'category': info['category'],
                'available': available,
                'requires': info['requires']
            }
        
        return result


# Registrazione modelli
def register_all_models():
    """Registra tutti i modelli disponibili."""
    
    # HF Standard Models
    try:
        from ..models.chronos import ChronosModel
        ModelRegistry.register(
            name="chronos",
            model_class=ChronosModel,
            category="hf_standard",
            requires=["transformers"]
        )
    except ImportError:
        pass
    
    try:
        from ..models.patchtst import PatchTSTModel
        ModelRegistry.register(
            name="patchtst",
            model_class=PatchTSTModel,
            category="hf_standard",
            requires=["transformers"]
        )
    except ImportError:
        pass
    
    # Custom API Models
    try:
        from ..models.tirex import TiRexModel
        ModelRegistry.register(
            name="tirex",
            model_class=TiRexModel,
            category="custom_api",
            requires=["tirex"]
        )
    except ImportError:
        pass
    
    try:
        from ..models.moirai import MoiraiModel
        ModelRegistry.register(
            name="moirai",
            model_class=MoiraiModel,
            category="custom_api",
            requires=["transformers"]
        )
    except ImportError:
        pass
    
    try:
        from ..models.toto import TotoModel
        ModelRegistry.register(
            name="toto",
            model_class=TotoModel,
            category="custom_api",
            requires=["toto-ts"]
        )
    except ImportError:
        pass
    
    # Existing models
    try:
        from ..models.timesfm import TimesFMModel
        ModelRegistry.register(
            name="timesfm",
            model_class=TimesFMModel,
            category="custom_api",
            requires=["timesfm"]
        )
    except ImportError:
        pass
    
    try:
        from ..models.transformer import TimeSeriesTransformer
        ModelRegistry.register(
            name="transformer",
            model_class=TimeSeriesTransformer,
            category="builtin",
            requires=[]  # Nessuna dipendenza extra
        )
    except ImportError:
        pass
    
    try:
        from ..models.lag_llama import LagLlamaModel
        ModelRegistry.register(
            name="lag_llama",
            model_class=LagLlamaModel,
            category="custom_api",
            requires=["gluonts", "lag-llama"]
        )
    except ImportError:
        pass
