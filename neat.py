""" """
import random
from typing import List
from dataclasses import dataclass, field

@dataclass
class Node:
    # Unique Node Identifier in
    # genome network structure
    number: int

    # sensor, hidden, output
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

    pass

innovation_number = 0
innovation_database: List[Connection] = list()
