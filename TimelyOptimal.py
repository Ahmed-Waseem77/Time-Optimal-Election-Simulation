from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from Node import Node

@dataclass
class TimelyOptimalNode(Node):
    is_initiator: bool = False
    is_candidate: bool = field(init=False)
    highest_candidate: int = field(init=False)
    pulse_number: int = 0
    max_distance: int = 0
    repetition_counter: int = 0
    completed: bool = False

    def __post_init__(self):
        self.is_candidate = self.is_initiator
        self.highest_candidate = self.node_id if self.is_initiator else 0

    def prepare_messages(self):
        if not self.completed:
            message = (self.highest_candidate, self.max_distance)
            self.outgoing_messages.append(message)

    def receive_messages(self, messages: List[Tuple[int, int]]):
        ""
        if not self.completed:
            self.incoming_messages.extend(messages)

    def process_messages(self):
        """
        Process all received messages and update the node's state.
        """
        if self.completed:
            return

        # Step 2: Increment pulse number
        self.pulse_number += 1

        # Step 3: Check for completion signal
        if any(distance == -1 for _, distance in self.incoming_messages):
            self.completed = True
            self.outgoing_messages.append((self.highest_candidate, -1))
            return

        if not self.incoming_messages:
            # No messages received, nothing to do
            return

        # Step 4: Determine the highest candidate among received messages
        highest_in_messages = max(candidate for candidate, _ in self.incoming_messages)
        filtered_messages = [
            (candidate, distance)
            for candidate, distance in self.incoming_messages
            if candidate >= highest_in_messages
        ]

        # Step 5: Update if a new candidate is heard
        if highest_in_messages > self.highest_candidate:
            self.is_candidate = False
            self.highest_candidate = highest_in_messages
            self.max_distance = self.pulse_number

        # Step 6: If not a candidate, no further processing
        if not self.is_candidate:
            self.incoming_messages.clear()
            return

        # Step 7: If y < x, set c =1 and continue
        if highest_in_messages < self.highest_candidate:
            self.repetition_counter = 1
            self.incoming_messages.clear()
            return

        # Step 8: Find max distance from remaining messages
        max_distance_in_messages = max(distance for _, distance in filtered_messages)

        # Step 9: Update max_distance and reset repetition counter if new distances are found
        if max_distance_in_messages > self.max_distance:
            self.max_distance = max_distance_in_messages
            self.repetition_counter = 0
        else:
            # Step 10: Increment repetition counter
            self.repetition_counter += 1

        # Step 11: Check if BFS is completed
        if self.repetition_counter > 1:
            # Step 12: Completion - send termination signal
            self.completed = True
            self.outgoing_messages.append((self.highest_candidate, -1))

        # Clear incoming messages after processing
        self.incoming_messages.clear()

    def run_algorithm(network: List[Node], verbose: bool = False):
        pulse = 0
        while True:
            pulse += 1
            print(f"--- Pulse {pulse} ---")
    
            # Phase 1: Prepare messages to send
            for node in network:
                node.prepare_messages()
    
            # Phase 2: Collect all outgoing messages
            messages_to_deliver: Dict[int, List[Tuple[int, int]]] = {node.node_id: [] for node in network}
            for node in network:
                for message in node.outgoing_messages:
                    for neighbor in node.neighbors:
                        messages_to_deliver[neighbor.node_id].append(message)
                node.reset_outgoing_messages()
    
            # Phase 3: Deliver messages to nodes
            for node in network:
                node.receive_messages(messages_to_deliver[node.node_id])
    
            # Phase 4: Process received messages
            for node in network:
                node.process_messages()
    
            # Phase 5: Check for termination
            all_completed = all(node.completed for node in network)
            if all_completed:
                print("All nodes have completed the algorithm.")
                break
            
        # Display results
        if verbose:
            print("\n--- Final States ---")
            for node in network:
                print(f"Node {node.node_id}:")
                print(f"  Highest Candidate Known: {node.highest_candidate}")
                print(f"  Max Distance Known: {node.max_distance}")
                print(f"  Pulse Number: {node.pulse_number}")
                print(f"  Repetition Counter: {node.repetition_counter}")
                print(f"  Is Candidate: {node.is_candidate}")
                print(f"  Completed: {node.completed}\n")
    
    