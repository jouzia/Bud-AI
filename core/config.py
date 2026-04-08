"""
Configuration Management

Single source of truth for all runtime settings.
Reads openenv.yaml + environment variables.
Never import settings from multiple places — always go through Config.

Usage:
    from core.config import Config
    cfg = Config.load()
    print(cfg.gemini_api_key)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent


@dataclass
class AgentConfig:
    model_name:       str   = "gemini-1.5-flash"
    temperature:      float = 0.3
    max_output_tokens: int  = 1024
    episodic_window:  int   = 5
    retry_attempts:   int   = 3
    retry_min_wait:   int   = 2
    retry_max_wait:   int   = 12


@dataclass
class RewardConfig:
    completeness_weight: float = 0.50
    diversity_weight:    float = 0.30
    efficiency_weight:   float = 0.20
    repeat_penalty:      float = 0.30
    illegal_penalty:     float = 0.50
    empty_op_penalty:    float = 0.20


@dataclass
class ServerConfig:
    host:  str = "0.0.0.0"
    port:  int = 7860
    share: bool = False


@dataclass
class Config:
    gemini_api_key:   str
    agent:            AgentConfig   = field(default_factory=AgentConfig)
    reward:           RewardConfig  = field(default_factory=RewardConfig)
    server:           ServerConfig  = field(default_factory=ServerConfig)
    spec_version:     str           = "2.0.0"
    debug:            bool          = False

    @classmethod
    def load(cls, yaml_path: Optional[Path] = None) -> "Config":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        port    = int(os.environ.get("PORT", 7860))
        debug   = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

        spec_version = "2.0.0"
        if _HAS_YAML:
            spec_file = yaml_path or (_ROOT / "openenv.yaml")
            if spec_file.exists():
                with open(spec_file) as f:
                    doc = yaml.safe_load(f)
                    spec_version = doc.get("version", spec_version)

        return cls(
            gemini_api_key = api_key,
            agent          = AgentConfig(),
            reward         = RewardConfig(),
            server         = ServerConfig(port=port),
            spec_version   = spec_version,
            debug          = debug,
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key)
