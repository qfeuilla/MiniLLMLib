"""JSON utility functions for MiniLLMLib."""
import json
from typing import Any, Dict

import json_repair


def to_dict(item: Any) -> Dict[Any, Any]:
    """Recursive function to turn objects into dict."""
    if isinstance(item, dict):
        data = {}
        for k, v in item.items():
            data[k] = to_dict(v)
        return data
    elif isinstance(item, (list, tuple)):
        return [to_dict(x) for x in item]
    elif hasattr(item, "__dict__"):
        data = {}
        for k, v in item.__dict__.items():
            # Exclude private attributes, help with infinite loops
            # (e.g. in chained list, put the parent as _parent)
            if not k.startswith("_"):
                data[k] = to_dict(v)
        return data
    else:
        return item

def extract_json_from_completion(completion: str) -> str:
    """Extract and parse JSON from completion string."""
    return json.dumps(json.loads(json_repair.repair_json(completion)))
