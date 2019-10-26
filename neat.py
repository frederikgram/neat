""" """
import sys
import random
import json
from typing import List, Dict, Any
from argparse import ArgumentParser
from dataclasses import dataclass, field

innovation_number = 0
innovation_database = list()

species_number = 0
species_database = list()

@dataclass
class Node:
    # Unique Node Identifier in
    # genome network structure
    number: int

    # "input", "hidden" or "output"
    node_type: str 

@dataclass
class Connection:

    # Nodes are represented
    # by their Node.number value
    input_node: int
    output_node: int

    # Float in range of (-1, 1)
    weight: float
    enabled: bool

    # Connection innovation number
    innov: int = field(init=False, default=None)

    def __post_init__(self):
        """ Initialize, assign and update 
            local and global innovation number
        """

        global innovation_number
        global innovation_database      

        stored_innov_id = search_for_innovation(self)
        if stored_innov_id == None:
            self.innov = innovation_number + 1
            innovation_number += 1
            innovation_database.append(self)
        else:
            self.innov = stored_innov_id

@dataclass
class Genome:
    nodes: List[Node]
    connections: List[Connection]
    fitness: float = field(init=False, default=0)

@dataclass
class Species:

    id: int
    representative: Genome
    genomes: List[Genome]

def search_for_innovation(connection: Connection) -> int or None:
    """ Searches for identical connections through every connection
        ever created. If such a connection exists, return its innovation number
        otherwise, return None
    """

    for con in innovation_database:
        if con.input_node == connection.input_node and con.output_node == connection.output_node:
            return con.innov

    return None

def add_connection(genome: Genome) -> Genome:
    """ Sample two random nodes, and create a connection
        between them with a randomized weight.
    """

    nodes = random.sample(genome.nodes, 2)

    new_connection = Connection(
        input_node = nodes[0].number,
        output_node = nodes[1].number,
        
        weight = random.uniform(-1, 1),
        enabled = True,
    )

    genome.connections.append(new_connection)

    return genome

def add_node(genome: Genome) -> Genome:
    """ Add a new node in place of an old connection,
        disable the old connection, and create two new
        connections acting as input and output connections
        to and from your new node
    """

    connection = random.choice(genome.connections)
    connection.enabled = False

    new_node = Node(number = len(genome.nodes) + 1,
                    node_type = "hidden")

    new_input_connection = Connection(
        input_node = connection.input_node,
        output_node = new_node.number,

        weight = 1,
        enabled = True,
    )

    new_output_connection = Connection(
        input_node = new_node.number,
        output_node = connection.output_node,

        weight = random.uniform(-1, 1),
        enabled = True,
    )

    genome.nodes.append(new_node)
    genome.connections.extend([new_input_connection, new_output_connection])

    return genome

def number_of_disjoint_genes(genome_a: Genome, genome_b: Genome) -> int:
    """ Returns the number of genes that are both not shared by the
        two parental genomes, nor exists at the end of a genome
    """

    count = 0

    zipped_connections = list(zip(genome_a.connections, genome_b.connections))
    for enum, (con_a, con_b) in enumerate(zipped_connections):
        if con_a not in genome_b.connections or con_b not in genome_a.connections:
            if enum < len(genome_a.connections) - 1 and enum < len(genome_b.connections) - 1:
                count += 1

    return count

def number_of_excess_genes(genome_a: Genome, genome_b: Genome) -> int:
    """ Returns the number of genes that are only carried by 
        a single parental genome, and exists past the other
        parental genomes last gene
    """

    return abs(len(genome_a.connections) - len(genome_b.connections))

def average_weight_difference(genome_a: Genome, genome_b: Genome) -> float:
    """ Returns the average difference between all weights (including
        disabled connections) shared by the to parental genomes
    """

    weights = list()

    longest, shortest = max(genome_a, genome_b, key = lambda genome: len(genome.connections)), min(genome_a, genome_b, key = lambda genome: len(genome.connections))
    for con_s in shortest.connections:
        if any([con_l for con_l in longest.connections if con_l.innov == con_s.innov]):
            weights.append(con_s.weight)
            
    return sum(weights) / len(weights)

def crossover(genome_a: Genome, genome_b) -> Genome:
    """ """ 

    new_genome = Genome(
        nodes = list(),
        connections = list()
    )

    # Check which of the two parent
    # genomes is the fittest
    most_fit, least_fit = max(genome_a, genome_b, key=lambda genome: genome.fitness), min(genome_a, genome_b, key=lambda genome: genome.fitness)

    # Compare the connections of the parental genomes    
    zipped_connections = list(zip(genome_a.connections, genome_b.connections))
    for enum, (con_a, con_b) in enumerate(zipped_connections):
        
        # Handle excess genes
        if enum == len(zipped_connections) - 1:
            
            if len(genome_a.connections) - 1 > enum or len(genome_b.connections) - 1 > enum:
                # A parent genome has excess genes

                try:
                    new_genome.connections.extend(most_fit.connections[enum: ])
                except IndexError:
                    # Excess genes were carried by the least fit parent
                    pass

        # Handle identical genes
        elif con_a.innov == con_b.innov:

            # Randomly chose a connection
            # between [con_a and con_b]
            new_con = random.choice([con_a, con_b])

            # If either of the parental connections does 
            # not have the same state as the other
            # possibly mutate the new connection
            # to be either enabled or disabled 
            if con_a.enabled != con_b.enabled:

                # 25% chance of enabled / disabled mutation
                if random.randint(0, 100) < 25:
                    new_con.enabled = random.choice([True, False])

            new_genome.connections.append(new_con)
        
        elif any([con for con in genome_b.connections if con.innov == con_a.innov]) == False or \
             any([con for con in genome_a.connections if con.innov == con_b.innov]) == False:
            
            if enum < len(genome_a.connections) - 1 and enum < len(genome_b.connections) - 1:
                try:
                    new_genome.connections.append(most_fit.connections[enum])
                except IndexError:
                    # Disjoint gene is carried by the least fit parental genome
                    pass
            
    # Create and append neccesary nodes
    nodes = set()
    
    for node in most_fit.nodes:
        if node.number not in nodes:
            new_genome.nodes.append(node)
            nodes.add(node.number)

    return new_genome

def genome_distance(genome_a: Genome, genome_b: Genome) -> float:
    """ Returns the distance between two genomes
        based on their historical markers
    """

    longest, shortest = max(genome_a, genome_b, key = lambda genome: len(genome.connections)), min(genome_a, genome_b, key = lambda genome: len(genome.connections))

    N = len(longest.connections) + len(longest.nodes)
    d = (c1 * number_of_excess_genes(genome_a, genome_b) / N) + (c2 * number_of_disjoint_genes(genome_a, genome_b) / N) + c3 * average_weight_difference(genome_a, genome_b)

    return d
    
def generate_initial_population(num_inputs: int, num_outputs: int, population_size: int) -> List[Genome]:
    """ Generates an initial population of the given population size
        with every input node directly mapped to every output node 
    """

    population: List[Genome] = list()

    for _ in range(0, args.population):
        new_genome = Genome(
            nodes = [
                # Increase number of inputs by one to create an initial "bias" node
                Node(j, "input") if j <= num_inputs else Node(j, "output") for j in range(0, num_inputs + num_outputs)
            ],

            connections =  [
                Connection(
                    input_node = x,
                    output_node = y,

                    weight = random.uniform(-1, 1),
                    enabled = True

                  # Generate every possible combination of input/output connections
                ) for x in range(0, num_inputs) for y in range(num_inputs, num_inputs + num_outputs)
            ]
        )

        population.append(new_genome)

    return population

def generate_new_generation(current_generation: List[Genome], population_size: int, mutation_rate: int) -> List[Genome]:
    """ """

    # Establish 25% of each species
    # as survivors and let them live
    # for the new generation
    for species in species_database:
        sorted_species = list(sorted(
            species.genomes, key = lambda genome: genome.fitness, reverse=True
        ))

        species.genomes = species.genomes[:round(int(len(current_generation) * .25))]
        species.representative = random.choice(species.genomes)

    # Initialize new generation
    new_generation: List[Genomes] = list()

    # Fill new generation with the top 25% of the previous generation
    for species in species_database:
        new_generation.extend(species.genomes)

    # Reset species genome list
    for species in species_database:
        species.genomes = list()

    # Repopulate new generation 
    while len(new_generation) < population_size:

        # Sample two random parent genomes
        # from the new generation
        genome_a, genome_b = random.sample(new_generation, 2)
        
        # Create a new genome by crossing
        # over the chosen parent genomes
        new_genome = crossover(genome_a, genome_b)

        # Possibly mutate the new genome 
        if random.randint(0, 100) < mutation_rate:
            if random.randint(0, 1) == 0:
                new_genome = add_connection(new_genome)
            else:
                new_genome = add_node(new_genome)

        new_generation.append(new_genome)


    for genome in new_generation:
        species_was_assigned = False
        for species in species_database:
            if genome_distance(genome, species.representative) <= dt:
                species.genomes.append(genome)
                species_was_assigned = True
        if species_was_assigned == False:
            species_database.append(
                Species(
                    id = species_number,
                    representative = genome,
                    genomes = [genome]
                )
            )

            species_number += 1

    return new_generation

def genome_to_dict(genome: Genome) -> Dict[str, Any]:
    """ Returns a JSON compatible representation
        of a given genomes' structure
    """

    genome_json = {
        "nodes" : [],
        "connections": [],
        "fitness": genome.fitness
    }

    for node in genome.nodes:
        genome_json["nodes"].append(
            {"number": node.number,
             "node_type": node.node_type}
        )

    for connection in genome.connections:
        genome_json["connections"].append(
            {"input_node": connection.input_node,
             "output_node": connection.output_node,
             "weight": connection.weight,
             "enabled": connection.enabled,
             "innov": connection.innov}
        )

    return json.dumps(genome_json)

    """ Initializes and returns a Genome object
        and every Node and Connection Object
        it contains from a given json dictionary
    """

    genome_json = json.loads(genome_json)

    new_genome = Genome(
        nodes = list(),
        connections = list(),
        fitness = genome_json["fitness"] 
    )

    for node in genome_json["nodes"]:
        new_genome.nodes.append(
            Node(
                number = node["number"],
                node_type = node["node_type"]
            )
        )

    for connection in genome_json["connections"]:
        new_genome.connections.append(
            Connection(
                input_node = connection["input_node"],
                output_node = connection["output_node"],

                weight = connection["weight"],
                enabled = connection["enabled"],
                innov = connection["innov"]   
            )
        )

    return new_genome

if __name__ == "__main__":

    parser = ArgumentParser()

    # Required Arguments

    parser.add_argument("--inputs", metavar='i', type=int,
                        required=True, help="""how many input nodes to initialize,
                                               bias node should not be included""")

    parser.add_argument("--outputs", metavar='o', type=int,
                        required=True, help="how many output nodes to initialize")

    parser.add_argument("--population", metavar='p', type=int,
                        required=True, help="population size")

    # Optional Arguments
    
    parser.add_argument("--mutation_rate", metavar='m', type=int, default=10,
                        required=False, help="""chance of a genome mutating,
                                             must be in range (0, 100),
                                             default is set to 10%""")

    # Constants with predefined values
    
    parser.add_argument("-c1", metavar="c1", type=float, default=1.0,
                        required=False, help="See README.md # Constants")

    parser.add_argument("-c2", metavar="c2", type=float, default=1.0,
                        required=False, help="See README.md # Constants")
    
    parser.add_argument("-c3", metavar="c3", type=float, default=0.4,
                        required=False, help="See README.md # Constants")

    parser.add_argument("-dt", metavar="dt", type=float, default=3.0,
                        required=False, help="See README.md # Constants")


    args = parser.parse_args()

    # Initialize Constants 
    # removing parser abstraction

    c1 = args.c1
    c2 = args.c2
    c3 = args.c3

    # Genome distance Threshold
    dt = args.dt

    population = generate_initial_population(
            num_inputs = args.inputs,
            num_outputs = args.outputs,
            population_size = args.population,
        )
    
    species_database.append(
        Species(
            id = species_number,
            representative = random.choice(population),
            genomes = population
        )
    )

    species_number += 1

    # Evolution Loop
    while True:

        # Write population to assesor

        # Read fitness from assesor

        # Calculate and assign shared
        # fitness between species
        for genome_a in population:
            adjusted_fitness = (
                genome_a.fitness / (
                    sum([0 if genome_distance(genome_a, genome_b) > dt else 1 for genome_b in population if genome_b != genome_a])
                )
            )

            genome_a.fitness = adjusted_fitness

        # Generate new population
        population = generate_new_generation(
            current_generation = population,
            population_size = args.population,
            mutation_rate = args.mutation_rate,
        )

