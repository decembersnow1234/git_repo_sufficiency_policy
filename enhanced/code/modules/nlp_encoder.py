import json
import numpy as np
import spacy
from spacy.tokens import Token
from typing import Any

class NLPEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling non-serializable NLP objects."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Token):
            return {'text': obj.text, 'pos': obj.pos_, 'dep': obj.dep_}

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if hasattr(obj, '__dict__'):
            return obj.__dict__

        return super().default(obj)
