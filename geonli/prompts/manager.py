"""
Prompt template manager.
Loads YAML/JSON prompt banks and registers them so tasks can pull
templates by name instead of embedding strings in Python.
"""

import os
from typing import Dict
from geonli.core.registry import register_prompt


class PromptManager:
    """
    Loads prompt template files from a directory and registers them.
    """

    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self._load_all()

    def _load_all(self):
        if not os.path.isdir(self.template_dir):
            return
        for fname in os.listdir(self.template_dir):
            path = os.path.join(self.template_dir, fname)
            name = os.path.splitext(fname)[0]
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                import yaml
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    for key, text in data.items():
                        register_prompt(f"{name}/{key}", text)
                elif isinstance(data, str):
                    register_prompt(name, data)
            elif fname.endswith(".json"):
                import json
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for key, text in data.items():
                        register_prompt(f"{name}/{key}", text)
            elif fname.endswith(".txt"):
                with open(path, "r") as f:
                    text = f.read()
                register_prompt(name, text)

    @staticmethod
    def register(name: str, template: str):
        register_prompt(name, template)
