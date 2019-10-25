# NeuroEvolution of Augmented Topologies

Independent implementation of NEAT in Python




## Components

### Genome Structure

Genomes are in this case represented as a Tuple of two lists storing dataclasses of:
    
- Node Genes
  - Node Number (ID)
  - Node type (Sensor, Output, Hidden)

- Connection Genes
    - Input Node
    - Output Node
    - Weight
    - Enabled / Disabled
    - Innovation Number
  

### - Historical Markers 

info info info

### - Mutation

- Add Node
  - 1. Chose a connection
  - 2. Disable chosen connection
  - 3. Create a new Node
  - 4. Create a connection to and from the new node
        based on the in / out of the now disabled connection

    Each newly created node always has its connections initialized
    in the following way:

    The input connection is initialized with a starting weight of 1.0, and the output connection is randomized. 


- Add Connection
    When adding a connection, you simply sample two random nodes,
    and create a connection between them with a randomized weight.

