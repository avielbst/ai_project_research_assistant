from pathlib import Path
from typing import Dict, Any
import yaml

_CONFIG: Dict[str, Any] | None = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root
    """
    return PROJECT_ROOT


def load_config(filename: str = "config.yml") -> Dict[str, Any]:
    """
    Load YAML config
    """
    global _CONFIG
    if _CONFIG is None:
        config_path = PROJECT_ROOT / "config" / filename

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)

    return _CONFIG
