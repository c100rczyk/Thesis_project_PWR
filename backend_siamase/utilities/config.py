import json
import os

class ConfigReader:
    def __init__(self, path):
        self.path = path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Zbuduj pełną ścieżkę do pliku config.json
        self.path = os.path.join(current_dir, path)
    def load_config(self):
        with open(self.path, "r") as file:
            return json.load(file)


class Config(dict):

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = Config(value)
        return value


embedding_layer = None
representatives = None