""" """
import sys
import random
from typing import List
from argparse import ArgumentParser
from dataclasses import dataclass, field

innovation_number = 0
innovation_database = list()

@dataclass
class Node:
    # Unique Node Identifier in
    # genome network structure
    number: int

    # "sensor", "hidden" or "output"
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
    fitness: float = field(init=False, default=None)


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
    genome.connections.extend[new_input_connection, new_output_connection]

    return genome

def crossover(genome_a: Genome, genome_b) -> Genome:
    """ """

    new_genome = Genome(
        nodes = list(),
        conncetions = list()
    )

    zipped_genomes = zip(genome_a.connections, genome_b.connections)

    for enum, con_a, con_b in enumerate(zipped_genomes):
        
        # Handle excess genes
        if enum == len(zipped_genomes) - 1:
            # the new_genome will inherit excess genomes
            # if the carrier of the genomes is the
            # parent with the highest fitness value
            # otherwise the genes will be skipped

            if len(genome_a.connections) - 1 > enum or len(genome_b.connections) - 1 > enum:
                # A parent genome has excess genes

                most_fit, least_fit = max(genome_a, genome_b), min(genome_a, genome_b)

                try:
                    new_genome.connections.extend(most_fit.connections[enum: ])
                except IndexError:
                    # Excess genes were carried by the least fit parent
                    pass

        # Handle identical genes
        elif con_a.innov == con_b.innov:
            # For every identical gene in the parent
            # genomes the new genome will decide which
            # genome to inherit randomly

            new_genome.connections.append(
                random.choice([con_a, con_b])
            )
        
        # Handle disjoint genes

    return new_genome

def generate_initial_population(num_inputs: int, num_outputs: int, population_size: int) -> List[Genome]:
    """ Generates an initial population of the given population size
        with every input node directly mapped to every output node 
    """

    population: List[Genome] = list()

    for i in range(0, args.population - 1):
        new_genome = Genome(
            nodes = [
                # Increase number of inputs by one to create an initial "bias" node
                Node(j, "sensor") if j <= num_inputs else Node(j, "output") for j in range(0, args.num_inputs + args.num_outputs)
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


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--inputs', metavar='i', type=int,
                        required=True, help="""how many sensor nodes to initialize,
                                               bias node should not be included""")

    parser.add_argument('--outputs', metavar='o', type=int,
                        required=True, help='how many output nodes to initialize')

    parser.add_argument('--population', metavar='p', type=int,
                        required=True, help='population size')

    args = parser.parse_args()

    population = generate_initial_population(args.inputs, args.outputs, args.population)
