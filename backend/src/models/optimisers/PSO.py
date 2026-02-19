import numpy as np
import copy
from typing import Callable
from tqdm import tqdm

class Particle:
    """
    Particle
    Represents a candidate solution in PSO. Each particle has:
    """
    def __init__(self, num_dimensions: int, rng: np.random.Generator, pos_min: float, pos_max: float):
        """
            :param num_dimensions: number of dimensions (total weights to optimise)
            :param rng: numpy random generator for reproducibility
            :param pos_min: minimum bound for position values
            :param pos_max: maximum bound for position values
        """
        self.position: np.ndarray = rng.uniform(pos_min, pos_max, size=num_dimensions) # Random initial position within bounds
        self.velocity: np.ndarray = np.zeros(num_dimensions) # Start stationary with zero velocity
        self.fitness: float = float("inf")  # Worse possible fitness to start as we are tying to minimise
        self.best_position: np.ndarray = self.position.copy()
        self.best_fitness: float = float("inf") # start with worst fitness for pbest as well
        


class PSO: 
    """
    Particle Swarm Optimisation.
    Minimises a fitness function by searching over a continuous space.
    The fitness function is injected externally so PSO can optimise any model.
    
    PSO maintains a swarm of particles, each representing a candidate solution (set of weights).
    Each particle has a position (current solution), velocity (direction of movement), and fitness (error).
    The swarm iteratively updates particle velocities and positions based on inertia, cognitive (personal best), and social (global best) components.
    PSO includes early stopping if no improvement is seen for a certain number of iterations.
    PSO also supports drift adaptation by reinitialising a fraction of particles to encourage exploration when the data distribution changes.
    """
    def __init__(
        self,
        num_dimensions: int,
        fitness_fn: Callable[[np.ndarray], float],
        num_particles: int = 30,
        max_iterations: int = 1000,
        inertia: float = 0.5,
        c1: float = 2.4,
        c2: float = 1.4,
        vel_max: float = 5.0,
        pos_min: float = -1.0,
        pos_max: float = 1.0,
        stopping_patience: int = 50,
        scatter_rate: float = 0.25,
        seed: int | None = 42,
    ):
        """
        :param num_dimensions: size of the search space (total weights)
        :param fitness_fn: function that takes a position vector and returns a scalar error
        :param num_particles: number of particles in the swarm
        :param max_iterations: maximum number of PSO iterations
        :param inertia: inertia weight (controls momentum)
        :param c1: cognitive coefficient (pull toward personal best)
        :param c2: social coefficient (pull toward global best)
        :param vel_max: maximum velocity (clamped to [-vel_max, vel_max])
        :param pos_min: minimum position bound
        :param pos_max: maximum position bound
        :param stopping_patience: stop if no improvement for this many iterations
        :param scatter_rate: fraction of particles to reinitialise on drift (0.0 to 1.0)
        :param seed: random seed for reproducibility
        """
        self.num_dimensions: int = num_dimensions
        self.fitness_fn: Callable[[np.ndarray], float] = fitness_fn
        self.num_particles: int = num_particles
        self.max_iterations: int = max_iterations
        self.inertia: float = inertia
        self.c1: float = c1
        self.c2: float = c2
        self.vel_max: float = vel_max
        self.pos_min: float = pos_min
        self.pos_max: float = pos_max
        self.stopping_patience: int = stopping_patience
        self.scatter_rate: float = scatter_rate
        self.rng = np.random.default_rng(seed)

        # swarm size and state
        self.particles: list[Particle] = []
        self.gbest_position: None = None
        self.gbest_fitness: float = float("inf")

        # Convergence tracking
        self.fitness_history: list[float] = []

    # Initialisation
    def _create_particles(self) -> None:
        """
        Create the initiial swarm with random positions
        """
        self.particles = [] # reset particles list
        # Find the best particle across the whole swarm to set initial gbest
        for _ in range(self.num_particles):
            p = Particle(self.num_dimensions, self.rng, self.pos_min, self.pos_max)
            p.fitness = self.fitness_fn(p.position)
            p.best_fitness = p.fitness
            p.best_position = p.position.copy()
            self.particles.append(p)
        
        # Return the best particle object to set initial gbest
        best_particle = min(self.particles, key=lambda p: p.fitness)
        self.gbest_position = best_particle.position.copy()
        self.gbest_fitness = best_particle.fitness

    # PSO Operations
    def _evaluate_fitness(self) -> None:
        """Evaluate fitness for all particles. update each particle's fitness attribute."""
        for p in self.particles:
            p.fitness = self.fitness_fn(p.position)
    
    def _update_pbest(self) -> None:
        """Update personal best for each particle. Look at each particle's current fitness and compare to its best_fitness. If current fitness is better, update best_position and best_fitness."""
        for p in self.particles:
            if p.fitness < p.best_fitness:
                p.best_position = p.position.copy()
                p.best_fitness = p.fitness

    def _update_gbest(self) -> None:
        """Update global best from all particles. Update gbest_position and gbest_fitness if any particle has better fitness."""
        for p in self.particles:
            if p.fitness < self.gbest_fitness:
                self.gbest_position = p.position.copy()
                self.gbest_fitness = p.fitness
                print(f"New gbest fitness: {self.gbest_fitness:.6f}")
                
    def _update_velocities(self) -> None:
        """Update velocities using inertia + cognitive + social components. Then clamp to vel_max."""
        for p in self.particles:
            # add random number between 0 and 1 one per dimension for the stochastic component of the cognitive and social terms
            r1 = self.rng.random(self.num_dimensions)
            r2 = self.rng.random(self.num_dimensions)

            # Inertia component keeps the particle moving in the same direction
            inertia_component = self.inertia * p.velocity
            # pull towards personal best position
            cognitive_component = self.c1 * r1 * (p.best_position - p.position)
            # pull towards swarm's global best position
            social_component = self.c2 * r2 * (self.gbest_position - p.position)

            # update the velocity 
            p.velocity = inertia_component + cognitive_component + social_component

            # Clamp velocity
            p.velocity = np.clip(p.velocity, -self.vel_max, self.vel_max)
            
    def _update_positions(self) -> None:
        """Update positions by adding velocity, then clamp to pos_min and pos_max."""
        for p in self.particles:
            p.position = p.position + p.velocity
            p.position = np.clip(p.position, self.pos_min, self.pos_max)
        
    def _check_stopping(self, no_improve_count: int) -> bool:
        """Check if we should stop early due to no improvement."""
        return no_improve_count >= self.stopping_patience
    
            
    # Training 
    def train(self) -> np.ndarray:
        """
        Run the PSO optimisation loop.

        :return: best position found (optimal weights)
        """
        self._create_particles()
        self.fitness_history = []
        no_improve_count = 0
        prev_gbest = self.gbest_fitness

        for i in tqdm(range(self.max_iterations), desc="PSO Iterations", ncols=100):
            self._evaluate_fitness()
            self._update_pbest()
            self._update_gbest()
            self._update_velocities()
            self._update_positions()

            self.fitness_history.append(self.gbest_fitness)

            # Check for improvement
            if self.gbest_fitness < prev_gbest:
                prev_gbest = self.gbest_fitness
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping
            if self._check_stopping(no_improve_count):
                print(f"PSO stopped early at iteration {i + 1} (no improvement for {self.stopping_patience} iterations)")
                break

        print(f"PSO complete — Best fitness: {self.gbest_fitness:.6f}")
        return self.gbest_position.copy()


    # Drift adaptation
    def scatter_particles(self, scatter_rate: float | None = None) -> None:
        """
        Reinitialise a fraction of particles for fresh exploration.
        Called when drift is detected. Keeps (1 - scatter_rate) of particles
        to retain prior knowledge, reinitialises the rest.
        """
        if scatter_rate is not None:
            self.scatter_rate = scatter_rate
        
        num_to_scatter = int(self.num_particles * self.scatter_rate)
        # Randomly choose which particles to reinitialise
        indices = self.rng.choice(self.num_particles, size=num_to_scatter, replace=False)

        for idx in indices:
            new_particle = Particle(self.num_dimensions, self.rng, self.pos_min, self.pos_max)
            new_particle.fitness = self.fitness_fn(new_particle.position)
            new_particle.best_fitness = new_particle.fitness
            new_particle.best_position = new_particle.position.copy()
            self.particles[idx] = new_particle

        # Re-evaluate gbest after scattering
        self.gbest_fitness = float("inf")
        for p in self.particles:
            if p.best_fitness < self.gbest_fitness:
                self.gbest_position = p.best_position.copy()
                self.gbest_fitness = p.best_fitness

        print(f"Scattered {num_to_scatter}/{self.num_particles} particles")

    def retrain(self) -> np.ndarray:
        """
        Retrain after drift detection.
        Scatters some particles then runs the PSO loop again.
        The swarm retains knowledge from surviving particles.

        :return: new best position found
        """
        self.scatter_particles()
        self.fitness_history = []
        no_improve_count = 0
        prev_gbest = self.gbest_fitness

        for i in range(self.max_iterations):
            self._evaluate_fitness()
            self._update_pbest()
            self._update_gbest()
            self._update_velocities()
            self._update_positions()

            self.fitness_history.append(self.gbest_fitness)

            if self.gbest_fitness < prev_gbest:
                prev_gbest = self.gbest_fitness
                no_improve_count = 0
            else:
                no_improve_count += 1

            if self._check_stopping(no_improve_count):
                print(f"PSO retrain stopped at iteration {i + 1}")
                break

        print(f"PSO retrain complete — Best fitness: {self.gbest_fitness:.6f}")
        return self.gbest_position.copy()
    
    # Utility
    def update_fitness_fn(self, new_fitness_fn: Callable[[np.ndarray], float]) -> None:
        """
        Update the fitness function (e.g. when training data changes after drift).
        :param new_fitness_fn: new fitness function
        """
        self.fitness_fn = new_fitness_fn
        