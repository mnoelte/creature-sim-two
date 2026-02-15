from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from config import SimulationConfig
from models import BodyPart, BodyPartType, Creature, Food, Genome, PartSummary, World, PART_COLORS


class Simulator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state = self._initial_state()

    def _initial_state(self) -> SimulationState:
        world = World(width=100.0, height=100.0)
        creatures = [self._spawn_creature() for _ in range(self.config.population_size)]
        food: List[Food] = [
            Food(
                position=self.rng.uniform([0.0, 0.0], [world.width, world.height]),
                value=float(self.rng.uniform(1.0, 3.0)),
            )
            for _ in range(max(5, self.config.food_cap // 4))
        ]
        return SimulationState(world=world, creatures=creatures, food=food, rng=self.rng)

    def _spawn_creature(self) -> Creature:
        genome = self._random_genome()
        pos = self.rng.uniform([0.0, 0.0], [100.0, 100.0])
        vel = np.zeros(2)
        energy = 10.0
        return Creature(position=pos, velocity=vel, energy=energy, genome=genome)

    def _random_genome(self) -> Genome:
        if self.config.max_parts < len(BodyPartType):
            raise ValueError("max_parts must be at least the number of body part types")

        parts: List[BodyPart] = []
        for kind in BodyPartType:
            parts.append(BodyPart(kind=kind, size=float(self.rng.uniform(0.5, 1.5))))
        while len(parts) < self.config.max_parts and self.rng.random() < 0.7:
            parts.append(
                BodyPart(
                    kind=self.rng.choice(list(BodyPartType)),
                    size=float(self.rng.uniform(0.5, 1.5)),
                )
            )
        return Genome(parts=self._ensure_all_part_types(parts))

    def step(self, tick_dt: float) -> None:
        self._respawn_food(tick_dt)
        self._update_creatures(tick_dt)
        self.state.sim_time += tick_dt
        self.state.ticks_in_generation += 1
        if self.state.ticks_in_generation >= self.config.ticks_per_generation:
            self._advance_generation()

    def _respawn_food(self, tick_dt: float) -> None:
        self.state.last_respawn_time += tick_dt
        if self.state.last_respawn_time < self.config.food_respawn_interval:
            return
        self.state.last_respawn_time = 0.0
        deficit = self.config.food_cap - len(self.state.food)
        if deficit <= 0:
            return
        spawn_count = max(1, math.floor(deficit / 2))
        for _ in range(spawn_count):
            position = self.rng.uniform([0.0, 0.0], [self.state.world.width, self.state.world.height])
            value = float(self.rng.uniform(1.0, 3.0))
            self.state.food.append(Food(position=position, value=value))

    def _update_creatures(self, tick_dt: float) -> None:
        alive_creatures: List[Creature] = []
        for creature in self.state.creatures:
            if not creature.alive:
                continue
            parts = self._summarize_parts(creature.genome)
            self._apply_behavior(creature, parts, tick_dt)
            self._handle_food_collisions(creature, parts)
            self._apply_energy(creature, parts, tick_dt)
            if creature.energy <= 0.0:
                creature.alive = False
            if creature.alive:
                alive_creatures.append(creature)
        self.state.creatures = alive_creatures

    def _summarize_parts(self, genome: Genome) -> PartSummary:
        mass = 0.0
        drag = 0.0
        thrust = 0.0
        sensor_range = 5.0
        mouth_radius = 2.0
        reproduction_capacity = 0.0
        upkeep = 0.0
        for part in genome.parts:
            size = max(0.1, part.size)
            mass += size * 1.0
            drag += 0.1 * size
            upkeep += 0.002 * size
            if part.kind == BodyPartType.THRUSTER:
                thrust += 5.0 * size
                mass += 0.5 * size
            elif part.kind == BodyPartType.SENSOR:
                sensor_range += 10.0 * size
            elif part.kind == BodyPartType.MOUTH:
                mouth_radius = max(mouth_radius, 1.5 * size)
            elif part.kind == BodyPartType.REPRODUCTION:
                reproduction_capacity += 0.5 * size
            elif part.kind == BodyPartType.CORE:
                mass += 0.5 * size
        mass = max(0.5, mass)
        return PartSummary(
            mass=mass,
            drag=drag,
            thrust=thrust,
            sensor_range=sensor_range,
            mouth_radius=mouth_radius,
            reproduction_capacity=reproduction_capacity,
            upkeep=upkeep,
        )

    def _apply_behavior(self, creature: Creature, parts: PartSummary, tick_dt: float) -> None:
        target_dir = self._sense_food_direction(creature, parts.sensor_range)
        wander_dir = self._wander_direction(creature)

        seek_weight = 1.0 if target_dir is not None else 0.0
        wander_weight = 0.85

        combined_dir = wander_weight * wander_dir
        if target_dir is not None:
            combined_dir += seek_weight * target_dir

        norm = np.linalg.norm(combined_dir)
        if norm > 0.0:
            combined_dir /= norm

        effective_thrust = max(parts.thrust, 3.0)
        creature.velocity += (effective_thrust / parts.mass) * combined_dir * tick_dt

        creature.velocity *= math.exp(-parts.drag * tick_dt)
        speed = float(np.linalg.norm(creature.velocity))
        max_speed = 20.0
        if speed > max_speed:
            creature.velocity *= max_speed / speed
        creature.position += creature.velocity * tick_dt
        creature.position = np.clip(
            creature.position,
            [0.0, 0.0],
            [self.state.world.width, self.state.world.height],
        )

    def _wander_direction(self, creature: Creature) -> np.ndarray:
        base = creature.velocity
        jitter = self.rng.normal(0.0, 1.3, size=2)
        combined = base * 0.15 + jitter
        norm = np.linalg.norm(combined)
        if norm == 0.0:
            return np.array([1.0, 0.0])
        return combined / norm

    def _sense_food_direction(self, creature: Creature, sensor_range: float) -> Optional[np.ndarray]:
        if not self.state.food:
            return None
        pos = creature.position
        best = None
        best_dist2 = sensor_range * sensor_range
        for food in self.state.food:
            d = food.position - pos
            dist2 = float(np.dot(d, d))
            if dist2 <= best_dist2:
                best = d
                best_dist2 = dist2
        if best is None:
            return None
        norm = np.linalg.norm(best)
        if norm == 0.0:
            return None
        return best / norm

    def _handle_food_collisions(self, creature: Creature, parts: PartSummary) -> None:
        if not self.state.food:
            return
        pos = creature.position
        mouth_r2 = parts.mouth_radius * parts.mouth_radius
        remaining_food: List[Food] = []
        for food in self.state.food:
            d = food.position - pos
            if float(np.dot(d, d)) <= mouth_r2:
                creature.energy += food.value
                creature.fitness += food.value
            else:
                remaining_food.append(food)
        self.state.food = remaining_food

    def _apply_energy(self, creature: Creature, parts: PartSummary, tick_dt: float) -> None:
        move_cost = 0.0005 * float(np.linalg.norm(creature.velocity))
        upkeep = parts.upkeep
        creature.energy -= (move_cost + upkeep) * tick_dt
        creature.fitness += 0.001 * tick_dt

    def _advance_generation(self) -> None:
        self.state.generation += 1
        self.state.ticks_in_generation = 0
        self._reproduce_population()
        self.state.food.clear()
        self.state.last_respawn_time = 0.0

    def _reproduce_population(self) -> None:
        parents_pool = self.state.creatures
        if not parents_pool:
            parents_pool = [self._spawn_creature() for _ in range(self.config.population_size)]
        new_creatures: List[Creature] = []
        while len(new_creatures) < self.config.population_size:
            p1 = self._tournament_pick(parents_pool)
            p2 = self._tournament_pick(parents_pool)
            child_genome = self._crossover_genomes(p1.genome, p2.genome)
            child_genome = self._mutate_genome(child_genome)
            pos = self.rng.uniform([0.0, 0.0], [self.state.world.width, self.state.world.height])
            child = Creature(
                position=pos,
                velocity=np.zeros(2),
                energy=10.0,
                genome=child_genome,
            )
            new_creatures.append(child)
        self.state.creatures = new_creatures

    def _tournament_pick(self, pool: Sequence[Creature]) -> Creature:
        k = min(4, len(pool))
        candidates = self.rng.choice(pool, size=k, replace=False)
        return max(candidates, key=lambda c: c.fitness)

    def _crossover_genomes(self, g1: Genome, g2: Genome) -> Genome:
        parts: List[BodyPart] = []
        for p1, p2 in zip(g1.parts, g2.parts):
            pick = p1 if self.rng.random() < 0.5 else p2
            parts.append(BodyPart(kind=pick.kind, size=pick.size))
        longer = g1.parts if len(g1.parts) > len(g2.parts) else g2.parts
        for extra in longer[len(parts) :]:
            if len(parts) >= self.config.max_parts:
                break
            parts.append(BodyPart(kind=extra.kind, size=extra.size))
        parts = parts[: self.config.max_parts]
        parts = self._ensure_all_part_types(parts)
        return Genome(parts=parts)

    def _mutate_genome(self, genome: Genome) -> Genome:
        parts = [BodyPart(kind=p.kind, size=p.size) for p in genome.parts]
        for p in parts:
            if self.rng.random() < 0.2:
                p.size = float(np.clip(p.size * self.rng.uniform(0.8, 1.2), 0.3, 2.0))
        if len(parts) < self.config.max_parts and self.rng.random() < 0.2:
            parts.append(
                BodyPart(
                    kind=self.rng.choice(list(BodyPartType)),
                    size=float(self.rng.uniform(0.5, 1.5)),
                )
            )
        type_counts = {kind: 0 for kind in BodyPartType}
        for p in parts:
            type_counts[p.kind] = type_counts.get(p.kind, 0) + 1

        removable = [i for i, p in enumerate(parts) if type_counts[p.kind] > 1]
        if removable and self.rng.random() < 0.1:
            drop_idx = int(self.rng.choice(removable))
            dropped_kind = parts[drop_idx].kind
            parts.pop(drop_idx)
            type_counts[dropped_kind] -= 1

        parts = parts[: self.config.max_parts]
        parts = self._ensure_all_part_types(parts)
        return Genome(parts=parts)

    def _ensure_all_part_types(self, parts: List[BodyPart]) -> List[BodyPart]:
        type_counts = {kind: 0 for kind in BodyPartType}
        for p in parts:
            type_counts[p.kind] = type_counts.get(p.kind, 0) + 1

        missing = [kind for kind, count in type_counts.items() if count == 0]
        for kind in missing:
            if len(parts) < self.config.max_parts:
                parts.append(BodyPart(kind=kind, size=float(self.rng.uniform(0.5, 1.5))))
            else:
                replace_candidates = [i for i, p in enumerate(parts) if type_counts[p.kind] > 1]
                if not replace_candidates:
                    replace_candidates = list(range(len(parts)))
                idx = int(self.rng.choice(replace_candidates))
                type_counts[parts[idx].kind] -= 1
                parts[idx] = BodyPart(kind=kind, size=float(self.rng.uniform(0.5, 1.5)))
                type_counts[kind] = type_counts.get(kind, 0) + 1
        return parts

    def run_headless(self) -> None:
        total_ticks = self.config.generations * self.config.ticks_per_generation
        for _ in range(total_ticks):
            self.step(self.config.tick_dt * self.config.speed_multiplier)

    def run_realtime(self) -> None:
        try:
            import pygame  # type: ignore
        except ImportError:
            raise SystemExit("pygame is required for realtime mode; install it via requirements.txt")

        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((800, 800))
        running = True
        paused = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_n:
                        self.step(self.config.tick_dt * self.config.speed_multiplier)
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.config.speed_multiplier = min(self.config.speed_multiplier * 2.0, 16.0)
                    elif event.key == pygame.K_MINUS:
                        self.config.speed_multiplier = max(self.config.speed_multiplier * 0.5, 0.0625)
                    elif event.key == pygame.K_m:
                        running = False

            if not paused:
                self.step(self.config.tick_dt * self.config.speed_multiplier)

            screen.fill((10, 10, 10))
            self._draw_food(screen)
            self._draw_creatures(screen)
            pygame.display.flip()
            clock.tick(self.config.render_fps)

        pygame.quit()

    def _draw_food(self, screen: "pygame.Surface") -> None:
        import pygame

        fill_color = (88, 132, 84)
        outline_color = (52, 88, 52)
        for food in self.state.food:
            x = int((food.position[0] / self.state.world.width) * screen.get_width())
            y = int((food.position[1] / self.state.world.height) * screen.get_height())
            pygame.draw.circle(screen, fill_color, (x, y), 4)
            pygame.draw.circle(screen, outline_color, (x, y), 4, width=1)

    def _draw_creatures(self, screen: "pygame.Surface") -> None:
        import pygame

        scale = min(
            screen.get_width() / self.state.world.width,
            screen.get_height() / self.state.world.height,
        )
        for creature in self.state.creatures:
            x = int((creature.position[0] / self.state.world.width) * screen.get_width())
            y = int((creature.position[1] / self.state.world.height) * screen.get_height())
            radius_world = self._creature_radius(creature)
            radius_px = max(2, int(radius_world * scale))
            self._draw_creature_body(screen, creature, (x, y), radius_px)

    def _creature_radius(self, creature: Creature) -> float:
        total_size = sum(part.size for part in creature.genome.parts)
        return max(1.5, total_size * 1.6)

    def _draw_creature_body(self, screen: "pygame.Surface", creature: Creature, pos: tuple[int, int], radius_px: int) -> None:
        import pygame

        parts = creature.genome.parts or [BodyPart(BodyPartType.CORE, size=1.0)]
        angle_step = (2.0 * math.pi) / len(parts)
        speed = float(np.linalg.norm(creature.velocity))
        base_angle = math.atan2(creature.velocity[1], creature.velocity[0]) if speed > 1e-3 else 0.0

        inner_r = max(1, int(radius_px * 0.35))
        outline_color = (56, 44, 32) if creature.alive else (72, 72, 72)

        for idx, part in enumerate(parts):
            a0 = base_angle + idx * angle_step
            a1 = a0 + angle_step
            mid = 0.5 * (a0 + a1)
            outer = radius_px

            points = [
                pos,
                (
                    pos[0] + int(inner_r * math.cos(a0)),
                    pos[1] + int(inner_r * math.sin(a0)),
                ),
                (
                    pos[0] + int(outer * math.cos(mid)),
                    pos[1] + int(outer * math.sin(mid)),
                ),
                (
                    pos[0] + int(inner_r * math.cos(a1)),
                    pos[1] + int(inner_r * math.sin(a1)),
                ),
            ]
            pygame.draw.polygon(screen, PART_COLORS.get(part.kind, (140, 120, 100)), points)

        pygame.draw.circle(screen, outline_color, pos, radius_px, width=2)


class SimulationState:
    def __init__(self, world: World, creatures: List[Creature], food: List[Food], rng: np.random.Generator) -> None:
        self.world = world
        self.creatures = creatures
        self.food = food
        self.rng = rng
        self.last_respawn_time: float = 0.0
        self.sim_time: float = 0.0
        self.generation: int = 0
        self.ticks_in_generation: int = 0
