"""Standalone viewer to display a creature genome with labeled body-part callouts."""

from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from genome_utils import random_genome
from models import BodyPart, BodyPartType, Genome, PART_COLORS


def _coerce_part_type(kind: object) -> BodyPartType:
    if isinstance(kind, BodyPartType):
        return kind
    # Try enum by value (e.g., "core")
    try:
        return BodyPartType(str(kind))
    except ValueError:
        pass
    # Try enum by name (e.g., "CORE")
    s = str(kind)
    if s in BodyPartType.__members__:
        return BodyPartType[s]
    upper = s.split(".")[-1]  # handle strings like "BodyPartType.CORE"
    if upper in BodyPartType.__members__:
        return BodyPartType[upper]
    lower = upper.lower()
    for t in BodyPartType:
        if t.value == lower:
            return t
    raise ValueError(f"Unknown body part type: {kind}")


def _ensure_all_part_types(parts: list[BodyPart], rng: np.random.Generator, max_parts: int) -> list[BodyPart]:
    if not parts:
        parts = [BodyPart(kind=BodyPartType.CORE, size=1.0)]

    parts = [BodyPart(kind=_coerce_part_type(p.kind), size=float(p.size)) for p in parts]
    counts = Counter(p.kind for p in parts)

    for t in BodyPartType:
        if counts.get(t, 0) == 0:
            if len(parts) < max_parts:
                parts.append(BodyPart(kind=t, size=float(rng.uniform(0.6, 1.4))))
                counts[t] += 1
            else:
                replace_idx = next((i for i, p in enumerate(parts) if counts[p.kind] > 1), None)
                if replace_idx is not None:
                    counts[parts[replace_idx].kind] -= 1
                    parts[replace_idx] = BodyPart(kind=t, size=float(rng.uniform(0.6, 1.4)))
                    counts[t] += 1
    return parts[:max_parts]


def _normalize_genome(genome: Genome, rng: np.random.Generator, max_parts: int) -> Genome:
    parts = [BodyPart(kind=_coerce_part_type(p.kind), size=float(p.size)) for p in genome.parts]
    parts = _ensure_all_part_types(parts, rng, max_parts)
    return Genome(parts=parts)


def _creature_radius(genome: Genome, max_radius: float) -> float:
    total_size = sum(part.size for part in genome.parts) or 1.0
    target = max(40.0, total_size * 60.0)
    return min(target, max_radius)


def _draw_creature_with_callouts(screen: "pygame.Surface", genome: Genome) -> None:
    import pygame

    screen.fill((16, 16, 16))
    width, height = screen.get_size()
    center = (width // 2, height // 2)

    parts = genome.parts or [BodyPart(BodyPartType.CORE, size=1.0)]
    font = pygame.font.SysFont("consolas", 16)
    callout_color = (230, 230, 230)
    line_color = (120, 120, 120)

    labels = [
        (part, f"{part.kind.value} (size {part.size:.2f})", font.render(f"{part.kind.value} (size {part.size:.2f})", True, callout_color))
        for part in parts
    ]
    max_label_w = max((surf.get_width() for _, _, surf in labels), default=0)
    max_label_h = max((surf.get_height() for _, _, surf in labels), default=0)

    angle_step = (2.0 * math.pi) / len(parts)
    base_angle = -math.pi / 2.0
    margin = max(80, max_label_w // 2 + 40, max_label_h // 2 + 24)
    max_radius = max(20, min(width, height) / 2 - margin)
    radius = int(_creature_radius(genome, max_radius))
    inner_r = max(12, int(radius * 0.35))
    callout_extension = max(16, min(60, min(width, height) / 2 - radius - (max_label_w / 2 + 12)))

    for idx, (part, label, label_surf) in enumerate(labels):
        a0 = base_angle + idx * angle_step
        a1 = a0 + angle_step
        mid = 0.5 * (a0 + a1)
        outer = radius

        points = [
            center,
            (
                center[0] + int(inner_r * math.cos(a0)),
                center[1] + int(inner_r * math.sin(a0)),
            ),
            (
                center[0] + int(outer * math.cos(mid)),
                center[1] + int(outer * math.sin(mid)),
            ),
            (
                center[0] + int(inner_r * math.cos(a1)),
                center[1] + int(inner_r * math.sin(a1)),
            ),
        ]
        pygame.draw.polygon(screen, PART_COLORS.get(part.kind, (160, 140, 120)), points)

        label_w, label_h = label_surf.get_size()
        callout_start = (
            center[0] + int(outer * math.cos(mid)),
            center[1] + int(outer * math.sin(mid)),
        )
        callout_end = (
            center[0] + int((outer + callout_extension) * math.cos(mid)),
            center[1] + int((outer + callout_extension) * math.sin(mid)),
        )
        pygame.draw.line(screen, line_color, callout_start, callout_end, width=2)

        label_pos = [
            callout_end[0] - label_w // 2,
            callout_end[1] - label_h // 2,
        ]
        pad = 8
        label_pos[0] = max(pad, min(label_pos[0], width - label_w - pad))
        label_pos[1] = max(pad, min(label_pos[1], height - label_h - pad))

        pygame.draw.rect(screen, (32, 32, 32), (*label_pos, label_w, label_h))
        pygame.draw.rect(screen, line_color, (*label_pos, label_w, label_h), width=1)
        screen.blit(label_surf, label_pos)

    pygame.draw.circle(screen, (54, 44, 32), center, radius, width=2)


def view_genome(genome: Genome, window_size: int = 640) -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Creature Structure Viewer")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        _draw_creature_with_callouts(screen, genome)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def _summarize_genome(genome: Genome) -> str:
    return ", ".join(f"{_coerce_part_type(p.kind).value}:{p.size:.2f}" for p in genome.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a creature genome with labeled body parts")
    parser.add_argument("--max-parts", type=int, default=7, help="Maximum parts when generating a random genome")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed; omit for nondeterministic genome")
    parser.add_argument("--window-size", type=int, default=640, help="Viewer window size in pixels")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    genome = _normalize_genome(random_genome(rng, args.max_parts), rng, args.max_parts)
    print(f"Viewer seed={args.seed if args.seed is not None else 'random'}, parts=[{_summarize_genome(genome)}]")
    view_genome(genome, window_size=args.window_size)


if __name__ == "__main__":
    main()
