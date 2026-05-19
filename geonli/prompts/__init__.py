import os
from geonli.prompts.manager import PromptManager

# Auto-load built-in templates on first import
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
PromptManager(_TEMPLATES_DIR)
