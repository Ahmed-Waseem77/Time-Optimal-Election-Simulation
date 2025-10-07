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


def setup_exaustive_network(
    numberOfNodes: int, connectivity: int, type: str
) -> List[Node]:
    """
    Network used for profiling
    This function needs to be determinstic in the way it connects Nodes to each other
    connectivity: number of neighbors each node has
    """
    nodes = []
    for i in range(numberOfNodes):
        if type == "TimelyOptimal":
            nodes.append(
                TimelyOptimalNode(node_id=i + 1, is_initiator=True if i == 0 else False)
            )
        else:
            nodes.append(Node(node_id=i))

    for i in range(numberOfNodes):
        for j in range(i + 1, i + connectivity + 1):
            if j < numberOfNodes:
                nodes[i].neighbors.append(nodes[j])
                nodes[j].neighbors.append(nodes[i])

    return nodes


def profile_with_increasing_nodes(algorithmType: str):
    """
    Profile the algorithm with increasing number of nodes.
    """
    # Dictionaries to store runtime and message counts
    listOfRuntimes = []
    listOfMessageCounts = []

    for i in range(1, 50):
        # The number of nodes is 10 + i*i to have a more exponential-like growth
        numberOfNodes = 10 + i * i
        network = setup_exaustive_network(numberOfNodes, 10, "TimelyOptimal")
        if algorithmType == "TimelyOptimal":
            total_runtime, total_messages = TimelyOptimalNode.run_algorithm(
                network, verbose=False
            )
            listOfRuntimes.append((total_runtime, numberOfNodes))
            listOfMessageCounts.append((total_messages, numberOfNodes))
        else:
            # Add other algorithms here
            pass

    # Plot the runtimes of the algorithm
    x_values_runtime = [item[1] for item in listOfRuntimes]
    y_values_runtime = [item[0] for item in listOfRuntimes]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values_runtime, y_values_runtime, marker="o", color="red", linestyle="-")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Total Runtime (seconds)")
    plt.title("Runtime with Increasing Nodes")
    plt.grid(True)
    plt.show()

    # Plot the message counts of the algorithm
    x_values_messages = [item[1] for item in listOfMessageCounts]
    y_values_messages = [item[0] for item in listOfMessageCounts]

    plt.figure(figsize=(8, 5))
    plt.plot(
        x_values_messages, y_values_messages, marker="o", color="skyblue", linestyle="-"
    )
    plt.xlabel("Number of Nodes")
    plt.ylabel("Total Messages Sent")
    plt.title("Message Complexity with Increasing Nodes")
    plt.grid(True)
    plt.show()


def profile_with_increasing_connectivity(algorithmType: str):
    """
    Profile the algorithm with increasing connectivity.
    """
    listOfResults = []

    for i in range(1, 50):
        connectivity = 10 + i
        network = setup_exaustive_network(200, connectivity, "TimelyOptimal")
        if algorithmType == "TimelyOptimal":
            total_runtime, total_messages = TimelyOptimalNode.run_algorithm(
                network, verbose=False
            )
            listOfResults.append((total_runtime, total_messages, connectivity))
        else:
            pass

    # Plotting for Runtime
    runtime_x_values = [item[2] for item in listOfResults]  # Connectivity values
    runtime_y_values = [item[0] for item in listOfResults]  # Total runtime

    plt.figure(figsize=(8, 5))
    plt.plot(runtime_x_values, runtime_y_values, marker="o", color="red", linestyle="-")
    plt.xlabel("Connectivity")
    plt.ylabel("Total Runtime (seconds)")
    plt.title("Runtime with Increasing Connectivity")
    plt.grid(True)
    plt.show()

    # Plotting for Message Complexity
    message_x_values = [item[2] for item in listOfResults]  # Connectivity values
    message_y_values = [item[1] for item in listOfResults]  # Total messages

    plt.figure(figsize=(8, 5))
    plt.plot(
        message_x_values, message_y_values, marker="o", color="blue", linestyle="-"
    )
    plt.xlabel("Connectivity")
    plt.ylabel("Total Messages Sent")
    plt.title("Message Complexity with Increasing Connectivity")
    plt.grid(True)
    plt.show()
