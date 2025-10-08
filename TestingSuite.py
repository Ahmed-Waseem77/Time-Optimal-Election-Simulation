from typing import List
from Node import Node
from TimelyOptimal import TimelyOptimalNode
from Raft import RaftNode
from Bully import BullyNode
from Ring import RingNode
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
    numberOfNodes: int, connectivity: int, algorithmType: str
) -> List[Node]:
    """
    Network used for profiling
    This function needs to be determinstic in the way it connects Nodes to each other
    connectivity: number of neighbors each node has
    """
    nodes = []
    for i in range(numberOfNodes):
        if algorithmType == "TimelyOptimal":
            nodes.append(
                TimelyOptimalNode(node_id=i + 1, is_initiator=True if i == 0 else False)
            )
        elif algorithmType == "Raft":
            nodes.append(
                RaftNode(node_id=i + 1, is_initiator=True if i == 0 else False)
            )
        elif algorithmType == "Bully":
            nodes.append(BullyNode(node_id=i + 1))
        elif algorithmType == "Ring":
            nodes.append(
                RingNode(node_id=i + 1, is_initiator=True if i == 0 else False)
            )
        else:  # Default case, or if Node is the base type needed
            nodes.append(Node(node_id=i + 1))

    # Establish connections based on algorithm type
    if algorithmType == "Ring":
        # Create a directed ring: node i connects to node (i+1)%numberOfNodes
        for i in range(numberOfNodes):
            nodes[i].neighbors.append(nodes[(i + 1) % numberOfNodes])
    else:
        # General connectivity for other algorithms (TimelyOptimal, Raft, Bully)
        for i in range(numberOfNodes):
            # Ensure connectivity is within bounds
            actual_connectivity = min(connectivity, numberOfNodes - 1)
            # Connect to `actual_connectivity` subsequent nodes to form a general graph
            # The original logic connects to i+1 to i+connectivity.
            # Let's replicate that as much as possible, assuming a wrap-around.
            for k in range(1, actual_connectivity + 1):
                j = (i + k) % numberOfNodes
                if nodes[j] not in nodes[i].neighbors and nodes[i] != nodes[j]:
                    nodes[i].neighbors.append(nodes[j])
                    nodes[j].neighbors.append(
                        nodes[i]
                    )  # Assuming undirected for non-ring

    return nodes


def profile_with_increasing_nodes():
    """
    Profile various algorithms with an increasing number of nodes and plot results together.
    """
    ALGORITHMS = ["TimelyOptimal", "Raft", "Bully", "Ring"]
    results_runtimes = {algo: [] for algo in ALGORITHMS}
    results_messages = {algo: [] for algo in ALGORITHMS}

    # Reduce iterations for quicker overall benchmarking with multiple algorithms
    num_iterations = 20  # Original was 50, changed to 20 for faster testing

    for algorithmType in ALGORITHMS:
        print(
            f"--- Starting profiling for increasing nodes with {algorithmType} algorithm ---"
        )
        current_algo_runtimes = []
        current_algo_message_counts = []

        for i in range(1, num_iterations + 1):
            # The number of nodes is 10 + i*i to have a more exponential-like growth
            numberOfNodes = 10 + i * i

            # Connectivity parameter for setup_exaustive_network
            # For Ring, connectivity is fixed at 1. For others, use a reasonable default.
            connectivity_param = (
                1 if algorithmType == "Ring" else min(10, numberOfNodes - 1)
            )

            network = setup_exaustive_network(
                numberOfNodes, connectivity_param, algorithmType
            )

            total_runtime, total_messages = 0.0, 0

            if algorithmType == "TimelyOptimal":
                total_runtime, total_messages = TimelyOptimalNode.run_algorithm(
                    network, verbose=False
                )
            elif algorithmType == "Raft":
                total_runtime, total_messages = RaftNode.run_algorithm(
                    network, verbose=False
                )
            elif algorithmType == "Bully":
                total_runtime, total_messages = BullyNode.run_algorithm(
                    network, verbose=False
                )
            elif algorithmType == "Ring":
                total_runtime, total_messages = RingNode.run_algorithm(
                    network, verbose=False
                )

            current_algo_runtimes.append((total_runtime, numberOfNodes))
            current_algo_message_counts.append((total_messages, numberOfNodes))

        results_runtimes[algorithmType] = current_algo_runtimes
        results_messages[algorithmType] = current_algo_message_counts
        print(
            f"--- Finished profiling for increasing nodes with {algorithmType} algorithm ---"
        )

    # Plotting Runtimes for all algorithms on one graph
    plt.figure(figsize=(10, 6))
    for algorithmType, data in results_runtimes.items():
        x_values = [item[1] for item in data]
        y_values = [item[0] for item in data]
        plt.plot(x_values, y_values, marker="o", linestyle="-", label=algorithmType)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Total Runtime (seconds)")
    plt.title("Runtime with Increasing Nodes for Various Algorithms")
    plt.legend()
    plt.grid(True)
    plt.savefig("all_algorithms_runtime_increasing_nodes.png")
    plt.close()

    # Plotting Message Complexity for all algorithms on one graph
    plt.figure(figsize=(10, 6))
    for algorithmType, data in results_messages.items():
        x_values = [item[1] for item in data]
        y_values = [item[0] for item in data]
        plt.plot(x_values, y_values, marker="o", linestyle="-", label=algorithmType)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Total Messages Sent")
    plt.title("Message Complexity with Increasing Nodes for Various Algorithms")
    plt.legend()
    plt.grid(True)
    plt.savefig("all_algorithms_message_complexity_increasing_nodes.png")
    plt.close()


def profile_with_increasing_connectivity():
    """
    Profile various algorithms with increasing connectivity and plot results together.
    """
    ALGORITHMS = ["TimelyOptimal", "Raft", "Bully", "Ring"]
    NUMBER_OF_NODES_FOR_CONNECTIVITY_TEST = 100  # Reduced from 200 for faster testing
    num_iterations = 20  # Reduced from 50 for faster testing

    results_runtimes = {algo: [] for algo in ALGORITHMS}
    results_messages = {algo: [] for algo in ALGORITHMS}

    for algorithmType in ALGORITHMS:
        print(
            f"--- Starting profiling for increasing connectivity with {algorithmType} algorithm ---"
        )
        current_algo_results = []  # stores (runtime, messages, connectivity)

        if algorithmType == "Ring":
            # For Ring, connectivity is always 1, so run once and use 1 for plotting.
            network = setup_exaustive_network(
                NUMBER_OF_NODES_FOR_CONNECTIVITY_TEST, 1, algorithmType
            )
            total_runtime, total_messages = RingNode.run_algorithm(
                network, verbose=False
            )
            current_algo_results.append((total_runtime, total_messages, 1))
            print(
                f"--- Finished profiling for increasing connectivity with {algorithmType} algorithm (fixed connectivity) ---"
            )
        else:
            for i in range(1, num_iterations + 1):
                connectivity = 5 + i * 2  # Start from 5 and increase by 2 each step
                # Ensure connectivity does not exceed number of nodes - 1
                connectivity = min(
                    connectivity, NUMBER_OF_NODES_FOR_CONNECTIVITY_TEST - 1
                )

                network = setup_exaustive_network(
                    NUMBER_OF_NODES_FOR_CONNECTIVITY_TEST, connectivity, algorithmType
                )

                total_runtime, total_messages = 0.0, 0
                if algorithmType == "TimelyOptimal":
                    total_runtime, total_messages = TimelyOptimalNode.run_algorithm(
                        network, verbose=False
                    )
                elif algorithmType == "Raft":
                    total_runtime, total_messages = RaftNode.run_algorithm(
                        network, verbose=False
                    )
                elif algorithmType == "Bully":
                    total_runtime, total_messages = BullyNode.run_algorithm(
                        network, verbose=False
                    )
                current_algo_results.append(
                    (total_runtime, total_messages, connectivity)
                )
            print(
                f"--- Finished profiling for increasing connectivity with {algorithmType} algorithm ---"
            )

        results_runtimes[algorithmType] = [
            (item[0], item[2]) for item in current_algo_results
        ]
        results_messages[algorithmType] = [
            (item[1], item[2]) for item in current_algo_results
        ]

    # Plotting Runtimes for all algorithms on one graph
    plt.figure(figsize=(10, 6))
    for algorithmType, data in results_runtimes.items():
        # Sort data by connectivity for cleaner plots if not already sorted
        sorted_data = sorted(data, key=lambda x: x[1])
        x_values = [item[1] for item in sorted_data]
        y_values = [item[0] for item in sorted_data]
        plt.plot(x_values, y_values, marker="o", linestyle="-", label=algorithmType)
    plt.xlabel("Connectivity")
    plt.ylabel("Total Runtime (seconds)")
    plt.title(
        f"Runtime with Increasing Connectivity for Various Algorithms (Nodes={NUMBER_OF_NODES_FOR_CONNECTIVITY_TEST})"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig("all_algorithms_runtime_increasing_connectivity.png")
    plt.close()

    # Plotting Message Complexity for all algorithms on one graph
    plt.figure(figsize=(10, 6))
    for algorithmType, data in results_messages.items():
        # Sort data by connectivity for cleaner plots
        sorted_data = sorted(data, key=lambda x: x[1])
        x_values = [item[1] for item in sorted_data]
        y_values = [item[0] for item in sorted_data]
        plt.plot(x_values, y_values, marker="o", linestyle="-", label=algorithmType)
    plt.xlabel("Connectivity")
    plt.ylabel("Total Messages Sent")
    plt.title(
        f"Message Complexity with Increasing Connectivity for Various Algorithms (Nodes={NUMBER_OF_NODES_FOR_CONNECTIVITY_TEST})"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig("all_algorithms_message_complexity_increasing_connectivity.png")
    plt.close()
