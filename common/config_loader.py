#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Any

try:
    import yaml
except Exception:
    yaml = None


def load_config_file(path: str) -> Dict[str, Any]:
    """Load YAML or JSON config. Returns empty dict on failure.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        ext = os.path.splitext(path)[1].lower()
        with open(path, 'r', encoding='utf-8') as f:
            if ext in ('.yml', '.yaml'):
                if yaml is None:
                    raise RuntimeError('PyYAML is not installed.')
                return yaml.safe_load(f) or {}
            elif ext == '.json':
                return json.load(f) or {}
            else:
                raise RuntimeError('Unsupported config extension: %s' % ext)
    except Exception:
        return {}



