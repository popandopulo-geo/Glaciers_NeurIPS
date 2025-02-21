import json
from types import SimpleNamespace

class Config:
    """
    Configuration class that initializes attributes from a JSON file.
    """
    def __init__(self, config_path: str):
        """
        Loads configuration from the given JSON file.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        for key, value in config_data.items():
            setattr(self, key, self._convert_dict_to_attr(value))

    def _convert_dict_to_attr(self, data):
        """
        Recursively converts dictionaries into SimpleNamespace objects for attribute access.
        """
        if isinstance(data, dict):
            return SimpleNamespace(**{k: self._convert_dict_to_attr(v) for k, v in data.items()})
        return data
