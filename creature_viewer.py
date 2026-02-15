"""Standalone viewer to display a creature genome with labeled body-part callouts."""

from __future__ import annotations

import argparse
import math
from typing import List

import numpy as np

from models import BodyPart, BodyPartType, Genome, PART_COLORS


def _random_genome(rng: np.random.Generator, max_parts: int) -> Genome:
    parts: List[BodyPart] = [BodyPart(BodyPartType.CORE, size=1.0)]
    while len(parts) < max_parts and rng.random() < 0.7:
        kind_value = rng.choice([k.value for k in BodyPartType])
        parts.append(
            BodyPart(
                kind=BodyPartType(kind_value),
                size=float(rng.uniform(0.5, 1.5)),
            )
        )
    return Genome(parts=parts)


def _creature_radius(genome: Genome) -> float:
    total_size = sum(part.size for part in genome.parts) or 1.0
    return max(40.0, total_size * 60.0)


def _draw_creature_with_callouts(screen: "pygame.Surface", genome: Genome) -> None:
    import pygame

    screen.fill((16, 16, 16))
    width, height = screen.get_size()
    center = (width // 2, height // 2)

    parts = genome.parts or [BodyPart(BodyPartType.CORE, size=1.0)]
    angle_step = (2.0 * math.pi) / len(parts)
    base_angle = -math.pi / 2.0
    radius = int(_creature_radius(genome))
    inner_r = max(12, int(radius * 0.35))

    font = pygame.font.SysFont("consolas", 16)
    callout_color = (230, 230, 230)
    line_color = (120, 120, 120)

    for idx, part in enumerate(parts):
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

        # Callout line and label
        label = f"{part.kind.value} (size {part.size:.2f})"
        label_surf = font.render(label, True, callout_color)
        label_w, label_h = label_surf.get_size()

        callout_start = (
            center[0] + int(outer * math.cos(mid)),
            center[1] + int(outer * math.sin(mid)),
        )
        callout_end = (
            center[0] + int((outer + 40) * math.cos(mid)),
            center[1] + int((outer + 40) * math.sin(mid)),
        )
        pygame.draw.line(screen, line_color, callout_start, callout_end, width=2)

        label_pos = (
            callout_end[0] - label_w // 2,
            callout_end[1] - label_h // 2,
        )
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
    return ", ".join(f"{p.kind.value}:{p.size:.2f}" for p in genome.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a creature genome with labeled body parts")
    parser.add_argument("--max-parts", type=int, default=7, help="Maximum parts when generating a random genome")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed; omit for nondeterministic genome")
    parser.add_argument("--window-size", type=int, default=640, help="Viewer window size in pixels")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    genome = _random_genome(rng, args.max_parts)
    print(f"Viewer seed={args.seed if args.seed is not None else 'random'}, parts=[{_summarize_genome(genome)}]")
    view_genome(genome, window_size=args.window_size)


if __name__ == "__main__":
    main()
