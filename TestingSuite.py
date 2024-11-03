from typing import List
from Node import Node
from TimelyOptimal import TimelyOptimalNode
import matplotlib.pyplot as plt
import pstats
import cProfile

def setup_network() -> List[Node]:
    """
    Set up the network with nodes and their neighbors.
    """
    # Create nodes with unique IDs
    node1 = Node(node_id=1, is_initiator=True)  # Initiator node
    node2 = Node(node_id=2, is_initiator=True)
    node3 = Node(node_id=3)
    node4 = Node(node_id=4)

    # Set up neighbors (assuming an undirected network)
    node1.neighbors = [node2, node3]
    node2.neighbors = [node1, node4]
    node3.neighbors = [node1, node4]
    node4.neighbors = [node2, node3]

    # List of all nodes in the network
    network = [node1, node2, node3, node4]
    return network


def setup_exaustive_network(numberOfNodes: int, connectivity: int, type: str) -> List[Node]:
    """
    Network used for profiling
    This function needs to be determinstic in the way it connects Nodes to each other
    connectivity: number of neighbors each node has
    """
    nodes = []
    for i in range(numberOfNodes):
        if type == "TimelyOptimal":
            nodes.append(TimelyOptimalNode(node_id=i+1, is_initiator=True if i == 0 else False))
        else:
            nodes.append(Node(node_id=i))

    for i in range(numberOfNodes):
        for j in range(i+1, i+connectivity+1):
            if j < numberOfNodes:
                nodes[i].neighbors.append(nodes[j])
                nodes[j].neighbors.append(nodes[i])

    return nodes

def profile_with_increasing_nodes(algorithmType: str):
    """
    Profile the algorithm with increasing number of nodes.
    """
    #dictionary of runtimes
    listOfRuntimes = []

    for i in range(1, 50):
        increasing_connectivity_profiler = cProfile.Profile()
        increasing_connectivity_profiler.enable()
        network = setup_exaustive_network(10 + i*i, 10, 'TimelyOptimal')
        if algorithmType == "TimelyOptimal":
            TimelyOptimalNode.run_algorithm(network, verbose=False)
        else:
            # Add other algorithms here
            pass
        increasing_connectivity_profiler.disable()

        stats = pstats.Stats(increasing_connectivity_profiler)

        # get total runtime and add it to the list
        total_runtime = stats.total_tt
        listOfRuntimes.append((total_runtime, 10 + i))

    # plot the runtimes of the algorithm
    # Extract x and y values
    x_values = [runtime[1] for runtime in listOfRuntimes]  # The 10 + i values
    y_values = [runtime[0] for runtime in listOfRuntimes]  # The total_runtime values

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', color='skyblue', linestyle='-')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Total Runtime")
    plt.title("Runtime Plot")
    plt.grid(True)
    plt.show()

def profile_with_increasing_connectivity(algorithmType: str):
    """
    Profile the algorithm with increasing connectivity.
    """
    #dictionary of runtimes
    listOfRuntimes = []
    for i in range(1, 50):
        increasing_connectivity_profiler = cProfile.Profile()
        increasing_connectivity_profiler.enable()
        network = setup_exaustive_network(200, 10 + i, 'TimelyOptimal')
        if algorithmType == "TimelyOptimal":
            TimelyOptimalNode.run_algorithm(network, verbose=False)
        else:
            # Add other algorithms here
            pass
        increasing_connectivity_profiler.disable()

        stats = pstats.Stats(increasing_connectivity_profiler)

        # get total runtime and add it to the list
        total_runtime = stats.total_tt
        listOfRuntimes.append((total_runtime, 10 + i))

    # plot the runtimes of the algorithm
    # Extract x and y values
    x_values = [runtime[1] for runtime in listOfRuntimes]  # The 10 + i values
    y_values = [runtime[0] for runtime in listOfRuntimes]  # The total_runtime values

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', color='skyblue', linestyle='-')
    plt.xlabel("Connectivity")
    plt.ylabel("Total Runtime")
    plt.title("Runtime Plot")
    plt.grid(True)
    plt.show()