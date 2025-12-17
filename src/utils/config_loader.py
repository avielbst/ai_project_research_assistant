from pathlib import Path
from typing import Dict, Any
import os
import yaml

_CONFIG: Dict[str, Any] | None = None

# Project root = repo root (works locally + in Docker)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root
    """
    return PROJECT_ROOT


def load_config(filename: str = "config.yml") -> Dict[str, Any]:
    """
    Load YAML config with optional environment overrides.
    """
    global _CONFIG
    if _CONFIG is None:

        # Allow Docker / CLI override of config location
        config_path = Path(
            os.getenv("CONFIG_PATH", PROJECT_ROOT / "config" / filename)
        )

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # -------------------------------
        # Environment overrides (Docker-friendly)
        # -------------------------------
        # Vector DB
        if "vector_store" in cfg:
            cfg["vector_store"]["db_dir"] = os.getenv(
                "LANCEDB_DIR",
                cfg["vector_store"].get("db_dir")
            )

        # LLM (llama.cpp)
        if "llm" in cfg:
            cfg["llm"]["model_path"] = os.getenv(
                "MODEL_PATH",
                cfg["llm"].get("model_path")
            )

            # Optional tuning via env
            if "N_THREADS" in os.environ:
                cfg["llm"]["n_threads"] = int(os.environ["N_THREADS"])

            if "N_CTX" in os.environ:
                cfg["llm"]["n_ctx"] = int(os.environ["N_CTX"])

        _CONFIG = cfg

    return _CONFIG
