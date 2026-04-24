"""
genetic_algorithm.py
--------------------
Core Genetic Algorithm logic for Neural Architecture Search.

IMPROVEMENTS in this version:
  - Genome now carries activation function + dropout rate (not just layer sizes)
  - Expanded layer size pool including 1024
  - Tournament selection replaces pure top-k (more diversity)
  - Crossover now mixes activation and dropout too
  - Mutation also mutates activation and dropout rate
  - Max layers increased to 6
"""

import random

# ── Search Space ──────────────────────────────────────────────────────────────

# Layer neuron count options
LAYER_SIZE_OPTIONS = [16, 32, 64, 128, 256, 512, 1024]

# Activation function options — GA will evolve the best one
ACTIVATION_OPTIONS = ['relu', 'tanh', 'elu', 'selu']

# Dropout rate options — GA will evolve the best one
# Removed 0.4 and 0.5: high dropout cripples shallow networks and hurts
# fitness evaluations, causing GA to wrongly discard good architectures.
DROPOUT_OPTIONS = [0.0, 0.1, 0.2, 0.3]

# Optimizer options — GA also evolves the optimizer
OPTIMIZER_OPTIONS = ['adam', 'rmsprop', 'adamw']

# Architecture depth limits
MIN_LAYERS = 1
MAX_LAYERS = 6


# ── Genome Structure ──────────────────────────────────────────────────────────
# A genome is now a dict, not just a list:
# {
#   'layers':     [128, 64, 32],   ← list of hidden layer sizes
#   'activation': 'relu',          ← activation function for all hidden layers
#   'dropout':    0.2              ← dropout rate after each hidden layer
# }


def random_genome() -> dict:
    """
    Create one random genome — a full neural network blueprint.

    Returns:
        dict with keys: layers, activation, dropout, optimizer
    """
    num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    return {
        'layers':     [random.choice(LAYER_SIZE_OPTIONS) for _ in range(num_layers)],
        'activation': random.choice(ACTIVATION_OPTIONS),
        'dropout':    random.choice(DROPOUT_OPTIONS),
        'optimizer':  random.choice(OPTIMIZER_OPTIONS)  # ← NEW: evolve optimizer too
    }


def init_population(pop_size: int) -> list:
    """
    Initialize a population of random genomes.

    Args:
        pop_size: How many individuals in generation 1
    Returns:
        List of genome dicts
    """
    return [random_genome() for _ in range(pop_size)]


def tournament_selection(scored_population: list, top_k: int, tournament_size: int = 3) -> list:
    """
    Tournament selection — more diverse than pure top-k.

    How it works:
      Repeat top_k times:
        - Randomly pick tournament_size candidates from the population
        - The best one among them wins and is selected
    
    This gives weaker individuals a small chance of being selected,
    which keeps diversity high and prevents premature convergence.

    Args:
        scored_population: List of (genome, fitness_score) tuples
        top_k: How many survivors to select
        tournament_size: How many compete in each mini-tournament
    Returns:
        List of selected genomes (without scores)
    """
    selected = []
    for _ in range(top_k):
        # Randomly pick tournament_size candidates
        tournament = random.sample(
            scored_population,
            min(tournament_size, len(scored_population))
        )
        # Best candidate in this tournament wins
        winner = max(tournament, key=lambda x: x[1])
        selected.append(winner[0])
    return selected


def crossover(parent_a: dict, parent_b: dict) -> dict:
    """
    Combine two parent genomes to produce a child genome.

    Layer crossover:  randomly pick each layer from either parent
    Activation:       randomly pick from either parent
    Dropout:          randomly pick from either parent

    Args:
        parent_a: First parent genome dict
        parent_b: Second parent genome dict
    Returns:
        Child genome dict
    """
    layers_a = parent_a['layers']
    layers_b = parent_b['layers']

    # Child length = randomly chosen from either parent's length
    child_length = random.choice([len(layers_a), len(layers_b)])
    child_layers = []

    for i in range(child_length):
        options = []
        if i < len(layers_a):
            options.append(layers_a[i])
        if i < len(layers_b):
            options.append(layers_b[i])
        child_layers.append(random.choice(options))

    # Inherit activation, dropout, and optimizer from either parent randomly
    child_activation = random.choice([parent_a['activation'], parent_b['activation']])
    child_dropout    = random.choice([parent_a['dropout'],    parent_b['dropout']])
    child_optimizer  = random.choice([
        parent_a.get('optimizer', 'adam'),
        parent_b.get('optimizer', 'adam')
    ])

    return {
        'layers':     child_layers if child_layers else [random.choice(LAYER_SIZE_OPTIONS)],
        'activation': child_activation,
        'dropout':    child_dropout,
        'optimizer':  child_optimizer
    }


def mutate(genome: dict, mutation_rate: float = 0.3) -> dict:
    """
    Apply random mutations to a genome.

    Possible mutations:
      1. Change a layer size
      2. Add a new layer
      3. Remove a layer
      4. Change the activation function
      5. Change the dropout rate

    Args:
        genome: The genome dict to mutate
        mutation_rate: Probability that any mutation happens
    Returns:
        Mutated genome dict (original is not modified)
    """
    genome = {
        'layers':     genome['layers'].copy(),
        'activation': genome['activation'],
        'dropout':    genome['dropout'],
        'optimizer':  genome.get('optimizer', 'adam')
    }

    if random.random() < mutation_rate:
        # Pick one of 6 mutation types (added change_optimizer)
        mutation_type = random.choice([
            'change_layer', 'add_layer', 'remove_layer',
            'change_activation', 'change_dropout', 'change_optimizer'
        ])

        if mutation_type == 'change_layer' and genome['layers']:
            # Change a random layer's neuron count
            idx = random.randint(0, len(genome['layers']) - 1)
            genome['layers'][idx] = random.choice(LAYER_SIZE_OPTIONS)

        elif mutation_type == 'add_layer' and len(genome['layers']) < MAX_LAYERS:
            # Insert a new layer at a random position
            idx = random.randint(0, len(genome['layers']))
            genome['layers'].insert(idx, random.choice(LAYER_SIZE_OPTIONS))

        elif mutation_type == 'remove_layer' and len(genome['layers']) > MIN_LAYERS:
            # Remove a random layer
            idx = random.randint(0, len(genome['layers']) - 1)
            genome['layers'].pop(idx)

        elif mutation_type == 'change_activation':
            # Switch to a different activation function
            genome['activation'] = random.choice(ACTIVATION_OPTIONS)

        elif mutation_type == 'change_dropout':
            # Switch to a different dropout rate
            genome['dropout'] = random.choice(DROPOUT_OPTIONS)

        elif mutation_type == 'change_optimizer':
            # Switch to a different optimizer
            genome['optimizer'] = random.choice(OPTIMIZER_OPTIONS)

    return genome


def create_next_generation(
    scored_population: list,
    pop_size: int,
    top_k: int = 5,
    mutation_rate: float = 0.3,
    tournament_size: int = 3
) -> list:
    """
    Full evolution pipeline for one generation:
    Tournament Selection → Elitism → Crossover → Mutation

    Args:
        scored_population: List of (genome, fitness_score) tuples
        pop_size: Size of the next generation
        top_k: How many survivors (parents)
        mutation_rate: Probability of mutation
        tournament_size: Candidates per tournament round
    Returns:
        New population of genome dicts
    """
    # Step 1: Tournament selection for diverse parents
    elite = tournament_selection(scored_population, top_k, tournament_size)

    # Step 2: Always keep the single best genome unchanged (elitism)
    best_genome = max(scored_population, key=lambda x: x[1])[0]
    new_population = [best_genome]

    # Step 3: Also keep all selected elites
    for g in elite:
        if len(new_population) < pop_size:
            new_population.append(g)

    # Step 4: Fill rest with crossover + mutation children
    while len(new_population) < pop_size:
        parent_a = random.choice(elite)
        parent_b = random.choice(elite)
        child    = crossover(parent_a, parent_b)
        child    = mutate(child, mutation_rate)
        new_population.append(child)

    return new_population


def genome_to_label(genome: dict) -> str:
    """
    Convert a genome dict to a human-readable string for display.

    Example: "layers=[128, 64] | relu | drop=0.2 | adam"
    """
    opt = genome.get('optimizer', 'adam')
    return f"layers={genome['layers']} | {genome['activation']} | drop={genome['dropout']} | {opt}"
