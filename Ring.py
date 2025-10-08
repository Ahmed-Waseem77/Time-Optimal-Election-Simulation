from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from Node import Node
import time


# Message types for Ring algorithm
ELECTION = "election"
COORDINATOR = "coordinator"


@dataclass
class RingNode(Node):
    is_up: bool = True  # Represents if the node is active
    is_coordinator: bool = False
    coordinator_id: int = None
    is_initiator: bool = False
    # For Ring election: (election_id, initiator_id, max_id_seen)
    election_message_payload: Tuple[int, int, int] = field(
        default_factory=lambda: (0, 0, 0)
    )
    election_in_progress: bool = False
    has_sent_election_in_current_pulse: bool = (
        False  # To prevent sending multiple messages in one pulse
    )

    def __post_init__(self):
        # In a real ring, neighbors would be ordered circularly.
        # For this simulation, we'll assume `neighbors` is an ordered list representing the ring.
        # This means `neighbors[0]` is the next node in the ring for sending messages.
        pass

    def start_election(self):
        """Initiates an election by sending an ELECTION message with its ID."""
        if not self.is_up:
            return

        if not self.election_in_progress:
            print(f"Node {self.node_id} initiates an election.")
            self.election_in_progress = True
            # Initial message contains (sender_id, initiator_id, max_id_seen)
            self.election_message_payload = (self.node_id, self.node_id, self.node_id)
            self.send_election_message()
        else:
            # If already in election, ensure it re-sends its current highest if applicable
            pass  # The prepare_messages will handle re-sending current election state if it changed.

    def send_election_message(self):
        """Sends the current election message to the next node in the ring."""
        if not self.is_up or not self.election_message_payload:
            return

        # Assuming the first neighbor in the list is the "next" node in the ring.
        # In a strict ring, there's only one "next" neighbor.
        if self.neighbors:
            next_node = self.neighbors[0]  # Assumes `neighbors` is ordered for the ring
            message = {
                "type": ELECTION,
                "payload": self.election_message_payload,
                "sender_id": self.node_id,
            }
            self.outgoing_messages.append((next_node.node_id, message))
            self.has_sent_election_in_current_pulse = True
            # print(f"Node {self.node_id} sent ELECTION {self.election_message_payload} to Node {next_node.node_id}")

    def declare_coordinator(self):
        """Declares itself as the coordinator and informs others."""
        self.is_coordinator = True
        self.coordinator_id = self.node_id
        self.election_in_progress = False
        print(f"Node {self.node_id} declares itself as the COORDINATOR.")

        # Send COORDINATOR message to all other nodes in the ring
        if self.neighbors:
            next_node = self.neighbors[0]
            message = {
                "type": COORDINATOR,
                "coordinator_id": self.node_id,
                "sender_id": self.node_id,  # The actual sender of this message in the ring
            }
            self.outgoing_messages.append((next_node.node_id, message))

    def prepare_messages(self):
        """
        Prepares messages for the current pulse.
        For Ring, this mainly involves propagating election/coordinator messages.
        """
        if not self.is_up:
            return

        self.has_sent_election_in_current_pulse = False  # Reset for this pulse

        # If it has an election message payload to send and hasn't sent it yet this pulse
        if (
            self.election_in_progress
            and self.election_message_payload
            and not self.has_sent_election_in_current_pulse
        ):
            self.send_election_message()

        # If it's the coordinator, it sends the coordinator message around the ring
        if self.is_coordinator:
            # The coordinator message should circulate the ring to inform all.
            # It only originates from the coordinator.
            if (
                self.neighbors and self.coordinator_id == self.node_id
            ):  # Only send if *this* node is the coordinator
                next_node = self.neighbors[0]
                message = {
                    "type": COORDINATOR,
                    "coordinator_id": self.node_id,
                    "sender_id": self.node_id,
                }
                self.outgoing_messages.append((next_node.node_id, message))

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

        for sender_id, message in self.incoming_messages:
            if message["type"] == ELECTION:
                self._handle_election_message(sender_id, message)
            elif message["type"] == COORDINATOR:
                self._handle_coordinator_message(sender_id, message)

        self.incoming_messages.clear()

    def _handle_election_message(self, sender_id: int, message: Dict):
        """Handle an incoming ELECTION message."""
        election_id, initiator_id, max_id_seen = message["payload"]
        # print(f"Node {self.node_id} received ELECTION {message['payload']} from {sender_id}. My current: {self.election_message_payload}")

        if not self.election_in_progress:
            # First time seeing an election, adopt its state and propagate
            self.election_in_progress = True
            self.coordinator_id = None
            self.is_coordinator = False
            self.election_message_payload = (
                election_id,
                initiator_id,
                max(self.node_id, max_id_seen),
            )
            self.send_election_message()  # Immediately propagate the updated message
            print(
                f"Node {self.node_id} adopted election, propagating: {self.election_message_payload}"
            )

        elif initiator_id == self.node_id:
            # Message has completed a full circle back to the initiator
            # The highest ID seen in the payload is the new coordinator
            new_coordinator_id = max_id_seen
            print(
                f"Node {self.node_id} (initiator) received ELECTION message back. New coordinator: {new_coordinator_id}"
            )
            self.election_in_progress = False
            self.election_message_payload = (0, 0, 0)  # Reset

            if new_coordinator_id == self.node_id:
                self.declare_coordinator()
            else:
                # Inform others about the coordinator (this node will send the first COORDINATOR message)
                self.coordinator_id = new_coordinator_id
                self.send_coordinator_announcement(new_coordinator_id)

        elif (
            election_id != self.election_message_payload[0]
            or max_id_seen > self.election_message_payload[2]
        ):
            # Received an election message that is either new or has a higher max_id
            # Update internal state and propagate
            self.election_message_payload = (
                election_id,
                initiator_id,
                max(self.node_id, max_id_seen),
            )
            self.send_election_message()
            print(
                f"Node {self.node_id} updated election message, propagating: {self.election_message_payload}"
            )

        # If election_id is the same and max_id_seen is not higher, simply ignore (message is looping without new info)
        # This helps prevent infinite loops if multiple nodes start elections or if message passes self before update

    def _handle_coordinator_message(self, sender_id: int, message: Dict):
        """Handle an incoming COORDINATOR message."""
        coordinator_id = message["coordinator_id"]
        # print(f"Node {self.node_id} received COORDINATOR from {sender_id}. Leader is {coordinator_id}")

        if self.coordinator_id != coordinator_id:
            print(
                f"Node {self.node_id} acknowledges Node {coordinator_id} as the new coordinator."
            )
            self.coordinator_id = coordinator_id
            self.is_coordinator = self.node_id == coordinator_id
            self.election_in_progress = False
            self.election_message_payload = (0, 0, 0)  # Reset election state

            # Propagate the coordinator message if it hasn't made a full circle yet
            if (
                message["sender_id"] != self.node_id
            ):  # If I didn't originate this message
                self.send_coordinator_announcement(
                    coordinator_id
                )  # Send it to my next neighbor

    def send_coordinator_announcement(self, coordinator_id: int):
        """Helper to send a COORDINATOR message."""
        if not self.is_up:
            return

        if self.neighbors:
            next_node = self.neighbors[0]
            message = {
                "type": COORDINATOR,
                "coordinator_id": coordinator_id,
                "sender_id": self.node_id,  # This node is now the one sending it
            }
            self.outgoing_messages.append((next_node.node_id, message))
            # print(f"Node {self.node_id} propagated COORDINATOR {coordinator_id} to {next_node.node_id}")

    @staticmethod
    def run_algorithm(network: List[Node], verbose: bool = False):
        """
        Simulates one run of the Ring election algorithm (simplified for a directed ring).
        Assumes `network` is already ordered to form a ring (node i -> node i+1 -> ... -> node N -> node 1).
        The `neighbors` list of each node must contain *only* its successor in the ring.
        """
        pulse = 0
        ring_nodes = [node for node in network if isinstance(node, RingNode)]

        # Ensure only active nodes participate
        active_nodes = [node for node in ring_nodes if node.is_up]
        if not active_nodes:
            return 0.0, 0  # No active nodes

        # Reset states for a new run
        for node in ring_nodes:
            node.messages_sent = 0
            node.is_coordinator = False
            node.coordinator_id = None
            node.election_in_progress = False
            node.election_message_payload = (0, 0, 0)
            node.outgoing_messages.clear()
            node.incoming_messages.clear()
            node.has_sent_election_in_current_pulse = False

        start_time = time.time()

        max_pulses = (
            len(active_nodes) * 3
        )  # Max pulses to ensure election and coordinator propagation
        stable_coordinator_threshold = (
            2  # Number of pulses all nodes agree on coordinator
        )

        # Pick one node to initiate the election (e.g., the one with the smallest ID or a specific initiator)
        # For this simulation, we'll let the node with is_initiator=True start.
        # If none, or multiple, we just pick the first active one.
        initiator = next((n for n in active_nodes if n.is_initiator), active_nodes[0])
        print(f"Initial trigger: Node {initiator.node_id} will start the election.")
        initiator.start_election()
        initiator_election_id = initiator.node_id  # Use initiator's ID as election_id

        stable_coordinator_pulses = 0

        while pulse < max_pulses:
            pulse += 1
            if verbose:
                print(f"\n--- Pulse {pulse} ---")

            # Phase 1: Each node prepares messages (sends election/coordinator messages)
            for node in active_nodes:
                node.prepare_messages()

            # Phase 2: Collect and deliver messages
            messages_to_deliver: Dict[int, List[Tuple[int, Dict]]] = {
                node.node_id: [] for node in active_nodes
            }
            for node in active_nodes:
                if not node.is_up:
                    continue

                # Each outgoing message is sent to *one* neighbor (the next in the ring)
                # messages_sent count should reflect this.
                # A single outgoing_message element in `node.outgoing_messages`
                # already represents a message destined for a specific neighbor.
                # So, messages_sent increases by the count of items in outgoing_messages.
                # If a node sends a message to multiple neighbors, it would create multiple entries.

                node.messages_sent += len(node.outgoing_messages)

                for target_id, message in node.outgoing_messages:
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

            # Check for termination: All active nodes agree on the same coordinator
            all_agree_on_coordinator = True
            current_coordinator_id = None

            # Find a non-None coordinator_id from any node
            for node in active_nodes:
                if node.coordinator_id is not None:
                    current_coordinator_id = node.coordinator_id
                    break

            if current_coordinator_id is not None:
                for node in active_nodes:
                    if node.coordinator_id != current_coordinator_id:
                        all_agree_on_coordinator = False
                        break
            else:  # No node has a coordinator_id yet
                all_agree_on_coordinator = False

            if all_agree_on_coordinator:
                stable_coordinator_pulses += 1
                if stable_coordinator_pulses >= stable_coordinator_threshold:
                    if verbose:
                        print(
                            f"All nodes agree on coordinator {current_coordinator_id} for {stable_coordinator_threshold} pulses. Election completed."
                        )
                    break
            else:
                stable_coordinator_pulses = (
                    0  # Reset if disagreement or no coordinator yet
                )

        end_time = time.time()
        total_runtime = end_time - start_time

        final_coordinator = None
        for node in active_nodes:
            if node.is_coordinator:
                final_coordinator = node
                break

        if verbose:
            print("\n--- Final States (Ring) ---")
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

        total_messages = sum(node.messages_sent for node in ring_nodes)
        return total_runtime, total_messages
