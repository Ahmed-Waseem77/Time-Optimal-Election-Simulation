from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from Node import Node
import time


# Message types for Bully algorithm
ELECTION = "election"
OK = "ok"
COORDINATOR = "coordinator"


@dataclass
class BullyNode(Node):
    is_up: bool = True  # Represents if the node is active
    is_coordinator: bool = False
    coordinator_id: int = None
    election_in_progress: bool = False
    responded_to_election: bool = False
    timeout: float = 0.5  # Timeout for waiting for OK/Coordinator messages

    def __post_init__(self):
        # A higher ID initiates an election if it doesn't know the coordinator
        # or if it suspects the coordinator is down.
        # For simulation, we can have one or more nodes initially trigger an election.
        pass

    def start_election(self):
        """Initiates an election."""
        print(f"Node {self.node_id} starts an election.")
        self.election_in_progress = True
        self.coordinator_id = None  # Clear previous coordinator
        self.responded_to_election = False

        # Send ELECTION message to all nodes with higher IDs
        for neighbor in self.neighbors:
            if neighbor.node_id > self.node_id:
                message = {"type": ELECTION, "sender_id": self.node_id}
                self.outgoing_messages.append((neighbor.node_id, message))

        if not any(neighbor.node_id > self.node_id for neighbor in self.neighbors):
            # If no higher ID nodes, this node declares itself coordinator
            self.declare_coordinator()

    def declare_coordinator(self):
        """Declares itself as the coordinator and informs others."""
        self.is_coordinator = True
        self.coordinator_id = self.node_id
        self.election_in_progress = False
        print(f"Node {self.node_id} declares itself as the COORDINATOR.")

        # Send COORDINATOR message to all other nodes
        for neighbor in self.neighbors:
            message = {"type": COORDINATOR, "coordinator_id": self.node_id}
            self.outgoing_messages.append((neighbor.node_id, message))

    def prepare_messages(self):
        """
        Prepare messages based on the node's current state.
        This includes initiating elections or sending heartbeats/coordinator pings.
        """
        if not self.is_up:
            return

        # If an election is in progress and this node started it,
        # it will have sent messages in start_election,
        # but if it's waiting for responses, it won't send more election messages.

        # If not in an election and no coordinator is known (or suspected down), start one.
        # For simplicity, we assume an initial trigger or if coordinator_id becomes None.
        # In a real system, a node would periodically ping the coordinator.
        if (
            not self.election_in_progress
            and self.coordinator_id is None
            and not self.is_coordinator
        ):
            self.start_election()

        # If this node is the coordinator, it sends heartbeats (COORDINATOR messages)
        # to periodically assert its leadership.
        if self.is_coordinator:
            for neighbor in self.neighbors:
                message = {"type": COORDINATOR, "coordinator_id": self.node_id}
                self.outgoing_messages.append((neighbor.node_id, message))

    def receive_messages(self, messages: List[Tuple[int, Dict]]):
        """
        Receive messages from neighbors.
        """
        if not self.is_up:
            return
        self.incoming_messages.extend(messages)

    def process_messages(self):
        """
        Process all received messages and update the node's state.
        """
        if not self.is_up:
            self.incoming_messages.clear()
            return

        # Process incoming messages
        processed_messages_current_pulse = []
        for sender_id, message in self.incoming_messages:
            if message["type"] == ELECTION:
                self._handle_election(sender_id, message)
            elif message["type"] == OK:
                self._handle_ok(sender_id, message)
            elif message["type"] == COORDINATOR:
                self._handle_coordinator(sender_id, message)

            processed_messages_current_pulse.append((sender_id, message))

        self.incoming_messages.clear()

    def _handle_election(self, sender_id: int, message: Dict):
        """Handle an incoming ELECTION message."""
        print(f"Node {self.node_id} received ELECTION from {sender_id}.")

        # A node receiving an ELECTION message from a lower ID sends an OK message back
        if self.node_id > sender_id:
            response = {"type": OK, "sender_id": self.node_id}
            self.outgoing_messages.append((sender_id, response))
            print(f"Node {self.node_id} sent OK to {sender_id}.")
            self.responded_to_election = (
                True  # Mark that we've responded to an election
            )

        # Whether it sent an OK or not, it starts its own election if it hasn't already
        # and if it has a higher ID than the sender.
        if not self.election_in_progress and self.node_id > sender_id:
            self.start_election()
        elif (
            self.election_in_progress
            and self.node_id > sender_id
            and self.coordinator_id is not None
            and sender_id > self.coordinator_id
        ):  # A new election started by a higher node than previous coordinator
            # Reset and start new election from this node to ensure all nodes with higher ID are informed
            self.election_in_progress = False
            self.coordinator_id = None
            self.start_election()

    def _handle_ok(self, sender_id: int, message: Dict):
        """Handle an incoming OK message."""
        print(f"Node {self.node_id} received OK from {sender_id}.")
        # If this node is waiting for OK messages (i.e., it started an election)
        # and it receives an OK from a higher ID, it knows a higher node is alive
        # and will take over the election. It stops its own election process.
        if self.election_in_progress:
            print(
                f"Node {self.node_id} election stopped, higher node {sender_id} is active."
            )
            self.election_in_progress = False
            # It will now wait for a COORDINATOR message from the new leader.
            # No need to reset coordinator_id yet, as it might still be waiting
            # for the new coordinator to be declared.

    def _handle_coordinator(self, coordinator_id: int, message: Dict):
        """Handle an incoming COORDINATOR message."""
        print(f"Node {self.node_id} received COORDINATOR from {coordinator_id}.")
        # Acknowledge the new coordinator
        self.coordinator_id = coordinator_id
        self.is_coordinator = False  # Ensure this node isn't confused about its role
        self.election_in_progress = False
        self.responded_to_election = False
        print(
            f"Node {self.node_id} acknowledges Node {coordinator_id} as the coordinator."
        )

    @staticmethod
    def run_algorithm(network: List[Node], verbose: bool = False):
        """
        Simulates one run of the Bully election algorithm.
        """
        pulse = 0
        bully_nodes = [node for node in network if isinstance(node, BullyNode)]

        # Ensure only active nodes participate
        active_nodes = [node for node in bully_nodes if node.is_up]

        # Reset states for a new run
        for node in bully_nodes:
            node.messages_sent = 0
            node.is_coordinator = False
            node.coordinator_id = None
            node.election_in_progress = False
            node.responded_to_election = False
            node.outgoing_messages.clear()
            node.incoming_messages.clear()

        start_time = time.time()

        # Initial trigger: the highest ID node among active nodes will eventually become coordinator.
        # If no node starts an election, we can pick one to start.
        # For this simulation, let's assume the highest active node (if not already coordinator)
        # or a node that detects failure will start an election.

        # To simulate a typical scenario, we can force the highest ID node to "start"
        # by making it initially not know the coordinator, or an arbitrary node.
        # Let's say the node with ID 1 (or any, for now) starts an election if it doesn't hear from a coordinator.

        # For initial trigger, let's make an arbitrary node start an election.
        # Or more realistically, if no coordinator is known, the first node that "wakes up"
        # or detects a failure might start one.

        # A simpler approach for simulation: every node checks if there's a coordinator.
        # If not, and it hasn't started an election, it starts one.
        # The node with the highest ID among the active ones will eventually win.

        # Let's add a mechanism for one node to initiate.
        # For testing, let's make the node with the highest ID the initial "initiator"
        # if the goal is to quickly converge to a leader.
        # Or, we can simulate a failure and a lower ID node initiating.

        # Let's try to ensure convergence. The simulation runs for a fixed number of pulses
        # or until a stable coordinator is found.

        max_pulses = 100
        stable_coordinator_threshold = (
            5  # Number of pulses a coordinator must be stable
        )
        stable_coordinator_pulses = 0
        current_coordinator = None

        while pulse < max_pulses:
            pulse += 1
            if verbose:
                print(f"\n--- Pulse {pulse} ---")

            # Phase 1: Each node prepares messages.
            # This is where elections are started if needed, or heartbeats are sent.
            for node in active_nodes:
                node.prepare_messages()

            # Phase 2: Collect and deliver messages
            messages_to_deliver: Dict[int, List[Tuple[int, Dict]]] = {
                node.node_id: [] for node in active_nodes
            }
            for node in active_nodes:
                if not node.is_up:
                    continue
                node.messages_sent += len(node.outgoing_messages) * len(node.neighbors)
                for target_id, message in node.outgoing_messages:
                    # Only deliver if target is active
                    target_node = next(
                        (n for n in active_nodes if n.node_id == target_id), None
                    )
                    if target_node and target_node.is_up:
                        messages_to_deliver[target_id].append((node.node_id, message))
                node.reset_outgoing_messages()

            # Phase 3: Deliver messages
            for node in active_nodes:
                if node.is_up:
                    node.receive_messages(messages_to_deliver[node.node_id])

            # Phase 4: Process messages
            for node in active_nodes:
                if node.is_up:
                    node.process_messages()

            # Check for coordinator stability
            current_pulse_coordinator = None
            coordinator_count = 0
            for node in active_nodes:
                if node.is_coordinator:
                    current_pulse_coordinator = node.node_id
                    coordinator_count += 1

            if coordinator_count == 1:
                if current_coordinator == current_pulse_coordinator:
                    stable_coordinator_pulses += 1
                else:
                    stable_coordinator_pulses = 1
                    current_coordinator = current_pulse_coordinator
            else:
                stable_coordinator_pulses = 0
                current_coordinator = None

            if stable_coordinator_pulses >= stable_coordinator_threshold:
                if verbose:
                    print(
                        f"Coordinator {current_coordinator} stable for {stable_coordinator_threshold} pulses. Election completed."
                    )
                break

            # If after a few pulses, no election has started, or if nodes with lower IDs
            # haven't received an OK from a higher node they sent an ELECTION to,
            # they should timeout and assume they won the election.
            # This simulation uses a fixed number of pulses to simplify timeouts.
            # In a real system, nodes would manage their own timers.

            # To ensure an election starts if no one has, let's make the highest ID active node
            # initiate if no election is in progress after a few pulses.
            if pulse == 1 and not any(n.election_in_progress for n in active_nodes):
                highest_id_node = max(active_nodes, key=lambda n: n.node_id)
                if not highest_id_node.is_coordinator:
                    print(
                        f"Pulse {pulse}: Node {highest_id_node.node_id} is forced to start an election."
                    )
                    highest_id_node.start_election()  # Force start if no activity

            # A node that started an election and received no OK messages from higher nodes
            # by the end of this pulse (or after its internal timeout) should declare itself coordinator.
            for node in active_nodes:
                if node.election_in_progress and not node.responded_to_election:
                    # If this node sent election messages to higher nodes, but didn't receive an OK
                    # from ANY higher node that it sent to, it would declare itself coordinator.
                    # This check is a simplification.
                    higher_neighbors = [
                        n for n in node.neighbors if n.node_id > node.node_id
                    ]
                    if not higher_neighbors:  # No higher neighbors to send to
                        node.declare_coordinator()
                    else:
                        # This part is tricky to simulate accurately without proper message tracking.
                        # For now, we rely on the overall pulse-based progression.
                        pass

        end_time = time.time()
        total_runtime = end_time - start_time

        final_coordinator = None
        for node in active_nodes:
            if node.is_coordinator:
                final_coordinator = node
                break

        if verbose:
            print("\n--- Final States (Bully) ---")
            for node in active_nodes:
                print(
                    f"Node {node.node_id}: Is Up={node.is_up}, Is Coordinator={node.is_coordinator}, Known Coordinator={node.coordinator_id}"
                )
            if final_coordinator:
                print(
                    f"\nElection successful. Coordinator is Node {final_coordinator.node_id}."
                )
            else:
                print(
                    "\nElection inconclusive or no stable coordinator found within simulation pulses."
                )

        total_messages = sum(node.messages_sent for node in bully_nodes)
        return total_runtime, total_messages
