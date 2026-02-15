from __future__ import annotations

from typing import List

import numpy as np

from models import BodyPart, BodyPartType, Genome


def ensure_all_part_types(parts: List[BodyPart], rng: np.random.Generator, max_parts: int) -> List[BodyPart]:
    type_counts = {kind: 0 for kind in BodyPartType}
    for p in parts:
        type_counts[p.kind] = type_counts.get(p.kind, 0) + 1

    missing = [kind for kind, count in type_counts.items() if count == 0]
    for kind in missing:
        if len(parts) < max_parts:
            parts.append(BodyPart(kind=kind, size=float(rng.uniform(0.5, 1.5))))
        else:
            replace_candidates = [i for i, p in enumerate(parts) if type_counts[p.kind] > 1]
            if not replace_candidates:
                replace_candidates = list(range(len(parts)))
            idx = int(rng.choice(replace_candidates))
            type_counts[parts[idx].kind] -= 1
            parts[idx] = BodyPart(kind=kind, size=float(rng.uniform(0.5, 1.5)))
            type_counts[kind] = type_counts.get(kind, 0) + 1
    return parts


def random_genome(rng: np.random.Generator, max_parts: int) -> Genome:
    if max_parts < len(BodyPartType):
        raise ValueError("max_parts must be at least the number of body part types")

    parts: List[BodyPart] = []
    for kind in BodyPartType:
        parts.append(BodyPart(kind=kind, size=float(rng.uniform(0.5, 1.5))))
    while len(parts) < max_parts and rng.random() < 0.7:
        parts.append(
            BodyPart(
                kind=rng.choice(list(BodyPartType)),
                size=float(rng.uniform(0.5, 1.5)),
            )
        )
    parts = parts[:max_parts]
    parts = ensure_all_part_types(parts, rng, max_parts)
    return Genome(parts=parts)
