from farcaster_sybil_detection.models.base import IModel


class ModelFactory:
    """Factory for creating model instances"""

    def __init__(self):
        self._models = {}

    def register(self, name: str, model_class: type):
        """Register a new model type"""
        self._models[name] = model_class

    def create(self, name: str, **kwargs) -> IModel:
        """Create a new model instance"""
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
        return self._models[name](**kwargs)
