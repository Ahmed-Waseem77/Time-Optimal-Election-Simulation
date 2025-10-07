from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class Node:
    node_id: int
    neighbors: List["Node"] = field(default_factory=list)
    incoming_messages: List[Tuple[int, int]] = field(default_factory=list)
    outgoing_messages: List[Tuple[int, int]] = field(default_factory=list)
    messages_sent: int = 0

    def reset_outgoing_messages(self):
        """
        Clear outgoing messages after they have been sent.
        """
        self.outgoing_messages.clear()

    def receive_messages(self, messages: List[Tuple[int, int]]):
        """
        Receive messages from neighbors.
        """
        pass

    def prepare_messages(self):
        """
        Prepare messages to send to all neighbors based on the current state.
        """
        pass

    def run_algorithim(network: List["Node"], verbose: bool = False):
        pass
