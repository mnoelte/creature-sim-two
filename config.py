from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    HEADLESS = "headless"
    REALTIME = "realtime"


@dataclass
class SimulationConfig:
    mode: Mode
    tick_dt: float
    render_fps: int
    population_size: int
    max_parts: int
    food_respawn_interval: float
    food_cap: int
    speed_multiplier: float
    seed: int
    generations: int
    ticks_per_generation: int
