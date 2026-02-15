from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class BodyPartType(str, Enum):
    CORE = "core"
    THRUSTER = "thruster"
    SENSOR = "sensor"
    MOUTH = "mouth"
    REPRODUCTION = "reproduction"


PART_COLORS = {
    BodyPartType.CORE: (140, 104, 72),
    BodyPartType.THRUSTER: (196, 124, 68),
    BodyPartType.SENSOR: (92, 132, 92),
    BodyPartType.MOUTH: (176, 148, 96),
    BodyPartType.REPRODUCTION: (116, 84, 92),
}


@dataclass
class BodyPart:
    kind: BodyPartType
    size: float


@dataclass
class Genome:
    parts: List[BodyPart] = field(default_factory=list)


@dataclass
class Creature:
    position: np.ndarray
    velocity: np.ndarray
    energy: float
    genome: Genome
    fitness: float = 0.0
    alive: bool = True


@dataclass
class Food:
    position: np.ndarray
    value: float


@dataclass
class World:
    width: float
    height: float


@dataclass
class PartSummary:
    mass: float
    drag: float
    thrust: float
    sensor_range: float
    mouth_radius: float
    reproduction_capacity: float
    upkeep: float
