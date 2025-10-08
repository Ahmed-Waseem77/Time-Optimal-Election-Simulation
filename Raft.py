from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from Node import Node
import time
import random

# Define Raft states
FOLLOWER = "follower"
CANDIDATE = "candidate"
LEADER = "leader"


@dataclass
class RaftNode(Node):
    state: str = FOLLOWER
    current_term: int = 0
    voted_for: int = None
    log: List[Tuple[int, str]] = field(default_factory=list)  # (term, command)
    commit_index: int = 0
    last_applied: int = 0
    next_index: Dict[int, int] = field(
        default_factory=dict
    )  # For leader, next log entry to send to each follower
    match_index: Dict[int, int] = field(
        default_factory=dict
    )  # For leader, highest log entry known to be replicated on each follower

    # Election timeout for followers and candidates
    election_timeout: float = 0.0
    last_heartbeat_time: float = (
        0.0  # Time when last message from leader or vote was received
    )

    # For candidates
    votes_received: int = 0

    # For debugging/simulation
    is_initiator: bool = False  # Used for initial setup, not part of core Raft

    def __post_init__(self):
        self.election_timeout = random.uniform(0.15, 0.30)  # 150ms to 300ms
        self.last_heartbeat_time = time.time()

    def set_election_timeout(self):
        self.election_timeout = random.uniform(0.15, 0.30)
        self.last_heartbeat_time = time.time()

    def become_follower(self, term):
        self.state = FOLLOWER
        self.current_term = term
        self.voted_for = None
        self.set_election_timeout()
        self.votes_received = 0

    def become_candidate(self):
        self.state = CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = 1  # Vote for self
        self.set_election_timeout()  # Reset election timeout
        # Request votes from all neighbors
        self.prepare_messages()

    def become_leader(self):
        self.state = LEADER
        # Reinitialize nextIndex and matchIndex for all neighbors
        for neighbor in self.neighbors:
            self.next_index[neighbor.node_id] = len(self.log) + 1
            self.match_index[neighbor.node_id] = 0
        # Send initial empty AppendEntries RPCs (heartbeats) to all followers
        self.prepare_messages()

    def prepare_messages(self):
        """
        Prepare messages based on the node's current state (Leader, Candidate, Follower).
        """
        if self.state == LEADER:
            # Send AppendEntries RPC (heartbeat or log entries)
            for neighbor in self.neighbors:
                prev_log_index = self.next_index.get(neighbor.node_id, 1) - 1
                prev_log_term = (
                    self.log[prev_log_index - 1][0] if prev_log_index > 0 else 0
                )
                entries_to_send = self.log[prev_log_index:]
                message = {
                    "type": "AppendEntries",
                    "term": self.current_term,
                    "leaderId": self.node_id,
                    "prevLogIndex": prev_log_index,
                    "prevLogTerm": prev_log_term,
                    "entries": entries_to_send,
                    "leaderCommit": self.commit_index,
                }
                self.outgoing_messages.append((neighbor.node_id, message))
        elif self.state == CANDIDATE:
            # Send RequestVote RPC
            last_log_index = len(self.log)
            last_log_term = self.log[-1][0] if self.log else 0
            message = {
                "type": "RequestVote",
                "term": self.current_term,
                "candidateId": self.node_id,
                "lastLogIndex": last_log_index,
                "lastLogTerm": last_log_term,
            }
            # Send to all neighbors for simplicity in simulation, in real Raft it's to all other servers
            for neighbor in self.neighbors:
                self.outgoing_messages.append((neighbor.node_id, message))
        # Followers don't initiate messages unless responding to RPCs, which is handled in process_messages

    def receive_messages(self, messages: List[Tuple[int, Dict]]):
        """
        Receive messages from neighbors.
        """
        self.incoming_messages.extend(messages)

    def process_messages(self):
        """
        Process all received messages and update the node's state.
        This also handles election timeouts for followers/candidates.
        """
        current_time = time.time()

        # Handle election timeout for followers and candidates
        if (
            self.state != LEADER
            and (current_time - self.last_heartbeat_time) > self.election_timeout
        ):
            print(
                f"Node {self.node_id} ({self.state}) election timeout. Becoming candidate."
            )
            self.become_candidate()
            # After becoming candidate, it will prepare_messages (RequestVote) in the next phase
            self.incoming_messages.clear()  # Clear any old messages before new term
            return  # Skip processing old messages for this pulse

        # Process incoming messages
        new_incoming_messages = []
        for sender_id, message in self.incoming_messages:
            if message["term"] > self.current_term:
                print(
                    f"Node {self.node_id} received message with higher term {message['term']} from {sender_id}. Becoming follower."
                )
                self.become_follower(message["term"])
            elif message["term"] < self.current_term:
                # Ignore messages with stale terms, except for specific responses
                if message["type"] not in [
                    "RequestVoteResponse",
                    "AppendEntriesResponse",
                ]:
                    continue  # Ignore stale requests
            # If terms are equal, or message term is higher (already handled by becoming follower)
            # or it's a response to an RPC, process it.
            new_incoming_messages.append((sender_id, message))

        self.incoming_messages = new_incoming_messages

        # Now process messages based on type and current state
        for sender_id, message in self.incoming_messages:
            if message["type"] == "RequestVote":
                self._handle_request_vote(sender_id, message)
            elif message["type"] == "RequestVoteResponse":
                self._handle_request_vote_response(sender_id, message)
            elif message["type"] == "AppendEntries":
                self._handle_append_entries(sender_id, message)
            elif message["type"] == "AppendEntriesResponse":
                self._handle_append_entries_response(sender_id, message)

        self.incoming_messages.clear()
        # After processing, if leader, try to commit new entries
        if self.state == LEADER:
            self._try_commit_entries()

    def _handle_request_vote(self, candidate_id: int, message: Dict):
        term = message["term"]
        last_log_index = message["lastLogIndex"]
        last_log_term = message["lastLogTerm"]
        vote_granted = False

        if term < self.current_term:
            vote_granted = False
        else:
            if self.voted_for is None or self.voted_for == candidate_id:
                # Raft rule: Candidate's log must be at least as up-to-date as receiver's log
                my_last_log_index = len(self.log)
                my_last_log_term = self.log[-1][0] if self.log else 0

                log_is_ok = (last_log_term > my_last_log_term) or (
                    last_log_term == my_last_log_term
                    and last_log_index >= my_last_log_index
                )

                if log_is_ok:
                    self.voted_for = candidate_id
                    self.last_heartbeat_time = time.time()  # Granting vote resets timer
                    vote_granted = True
                    print(
                        f"Node {self.node_id} grants vote to {candidate_id} for term {term}."
                    )
                else:
                    print(
                        f"Node {self.node_id} denies vote to {candidate_id} (log not up-to-date)."
                    )
            else:
                print(
                    f"Node {self.node_id} denies vote to {candidate_id} (already voted for {self.voted_for})."
                )

        response = {
            "type": "RequestVoteResponse",
            "term": self.current_term,
            "voteGranted": vote_granted,
            "senderId": self.node_id,  # So candidate knows who sent the response
        }
        self.outgoing_messages.append((candidate_id, response))

    def _handle_request_vote_response(self, sender_id: int, message: Dict):
        if self.state == CANDIDATE:
            if message["term"] == self.current_term and message["voteGranted"]:
                self.votes_received += 1
                print(
                    f"Node {self.node_id} (Candidate) received vote from {sender_id}. Total votes: {self.votes_received}"
                )
                if (
                    self.votes_received > (len(self.neighbors) + 1) // 2
                ):  # Majority of total nodes (self + neighbors)
                    print(
                        f"Node {self.node_id} (Candidate) received majority votes. Becoming Leader for term {self.current_term}."
                    )
                    self.become_leader()
            elif message["term"] > self.current_term:
                # Should already be handled by the initial term check in process_messages
                pass
        # Ignore if not candidate or term mismatch (already handled)

    def _handle_append_entries(self, leader_id: int, message: Dict):
        term = message["term"]
        leader_commit = message["leaderCommit"]
        prev_log_index = message["prevLogIndex"]
        prev_log_term = message["prevLogTerm"]
        entries = message["entries"]

        success = False
        if term < self.current_term:
            success = False
        else:
            self.last_heartbeat_time = time.time()  # Reset election timer
            if self.state == CANDIDATE:
                print(
                    f"Node {self.node_id} (Candidate) received AppendEntries from Leader {leader_id}. Stepping down."
                )
                self.become_follower(term)
            elif self.state == LEADER:
                # This should not happen in a correctly functioning Raft (only one leader per term)
                # If it happens, means there's a split-brain or new leader elected. Current leader steps down.
                print(
                    f"Node {self.node_id} (Leader) received AppendEntries from another Leader {leader_id}. Stepping down."
                )
                self.become_follower(term)
            elif self.current_term < term:
                self.become_follower(term)  # Update term and become follower

            # Raft log consistency check
            if prev_log_index > 0 and (
                len(self.log) < prev_log_index
                or self.log[prev_log_index - 1][0] != prev_log_term
            ):
                print(
                    f"Node {self.node_id} AppendEntries rejected: log inconsistency at index {prev_log_index}."
                )
                success = False
            else:
                # Append new entries not already in log
                new_entries_added = False
                for i, entry in enumerate(entries):
                    log_idx = prev_log_index + i
                    if log_idx < len(self.log):
                        if (
                            self.log[log_idx][0] != entry[0]
                        ):  # If existing entry conflicts
                            self.log = self.log[:log_idx]  # Truncate log
                            self.log.append(entry)
                            new_entries_added = True
                    else:  # New entry
                        self.log.append(entry)
                        new_entries_added = True
                if new_entries_added:
                    print(f"Node {self.node_id} appended new entries. Log: {self.log}")

                # Update commit index
                if leader_commit > self.commit_index:
                    self.commit_index = min(leader_commit, len(self.log))
                    self._apply_log_entries()  # Apply newly committed entries
                    print(
                        f"Node {self.node_id} commit_index updated to {self.commit_index}."
                    )
                success = True

        response = {
            "type": "AppendEntriesResponse",
            "term": self.current_term,
            "success": success,
            "senderId": self.node_id,
            "matchIndex": len(self.log)
            if success
            else 0,  # Inform leader of highest index matched
        }
        self.outgoing_messages.append((leader_id, response))

    def _handle_append_entries_response(self, follower_id: int, message: Dict):
        if self.state == LEADER:
            if message["term"] == self.current_term:
                if message["success"]:
                    new_match_index = message["matchIndex"]
                    self.match_index[follower_id] = new_match_index
                    self.next_index[follower_id] = new_match_index + 1
                    # print(f"Leader {self.node_id} received successful AppendEntriesResponse from {follower_id}. matchIndex: {new_match_index}")
                    self._try_commit_entries()
                else:
                    # Log inconsistency: decrement nextIndex and retry AppendEntries
                    self.next_index[follower_id] = max(
                        1, self.next_index[follower_id] - 1
                    )
                    print(
                        f"Leader {self.node_id} received failed AppendEntriesResponse from {follower_id}. Decrementing nextIndex to {self.next_index[follower_id]}."
                    )
                    # In a real Raft, the leader would immediately re-send AppendEntries with the decremented nextIndex
                    # For this simulation, it will happen in the next prepare_messages phase.
            elif message["term"] > self.current_term:
                # Should already be handled by the initial term check in process_messages
                pass

    def _try_commit_entries(self):
        """
        Leader attempts to advance commit_index.
        """
        if self.state != LEADER:
            return

        # Find N such that a majority of matchIndex[i] >= N and log[N].term == current_term
        # and N > commit_index
        for N in range(len(self.log), self.commit_index, -1):
            if (
                self.log[N - 1][0] == self.current_term
            ):  # Only commit entries from current term
                count = 1  # Self
                for follower_id in self.match_index:
                    if self.match_index[follower_id] >= N:
                        count += 1
                if count > (len(self.neighbors) + 1) // 2:  # Majority
                    self.commit_index = N
                    self._apply_log_entries()
                    print(
                        f"Leader {self.node_id} committed entries up to index {self.commit_index}."
                    )
                    break  # Found highest N to commit

    def _apply_log_entries(self):
        """
        Apply committed log entries to the state machine.
        """
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            # In a real system, apply self.log[self.last_applied - 1] to the state machine
            # For this simulation, we just print
            print(
                f"Node {self.node_id} applied log entry: {self.log[self.last_applied - 1]} at index {self.last_applied}"
            )

    @staticmethod
    def run_algorithm(network: List[Node], verbose: bool = False):
        """
        Simulates one run of the Raft election algorithm.
        This simplified simulation focuses on election and leader stability,
        not full log replication.
        """
        pulse = 0
        raft_nodes = [node for node in network if isinstance(node, RaftNode)]

        for node in raft_nodes:
            node.messages_sent = 0
            node.set_election_timeout()  # Reset all timeouts at start

        start_time = time.time()

        # Initial state: all nodes are followers
        for node in raft_nodes:
            node.become_follower(0)

        election_in_progress = True
        leader_found_pulses = 0
        max_election_pulses = 100  # Prevent infinite loops in case of no leader
        stable_leader_threshold = (
            10  # Number of pulses a leader must be stable to declare completion
        )

        while election_in_progress and pulse < max_election_pulses:
            pulse += 1
            if verbose:
                print(f"\n--- Pulse {pulse} ---")

            # Phase 1: Prepare messages
            for node in raft_nodes:
                node.prepare_messages()

            # Phase 2: Collect and deliver messages
            messages_to_deliver: Dict[int, List[Tuple[int, Dict]]] = {
                node.node_id: [] for node in raft_nodes
            }
            for node in raft_nodes:
                node.messages_sent += len(node.outgoing_messages) * len(
                    node.neighbors
                )  # Rough estimate for messages
                for target_id, message in node.outgoing_messages:
                    messages_to_deliver[target_id].append((node.node_id, message))
                node.reset_outgoing_messages()

            # Phase 3: Deliver messages
            for node in raft_nodes:
                node.receive_messages(messages_to_deliver[node.node_id])

            # Phase 4: Process messages
            for node in raft_nodes:
                node.process_messages()

            # Check for leader and stability
            current_leader = None
            leader_count = 0
            for node in raft_nodes:
                if node.state == LEADER:
                    current_leader = node
                    leader_count += 1

            if leader_count == 1:
                leader_found_pulses += 1
                if leader_found_pulses >= stable_leader_threshold:
                    election_in_progress = False
                    if verbose:
                        print(
                            f"Leader {current_leader.node_id} stable for {stable_leader_threshold} pulses. Election completed."
                        )
            else:
                leader_found_pulses = 0  # Reset if no clear leader or multiple leaders

            # Add a simple log entry at a fixed interval for the leader
            if current_leader and current_leader.state == LEADER and pulse % 5 == 0:
                current_leader.log.append(
                    (current_leader.current_term, f"Command-{pulse}")
                )
                if verbose:
                    print(
                        f"Leader {current_leader.node_id} added a log entry: Command-{pulse}"
                    )

        end_time = time.time()
        total_runtime = end_time - start_time

        # Final check if a leader was elected
        final_leader = None
        for node in raft_nodes:
            if node.state == LEADER:
                final_leader = node
                break

        if verbose:
            print("\n--- Final States (Raft) ---")
            for node in raft_nodes:
                print(
                    f"Node {node.node_id}: State={node.state}, Term={node.current_term}, Voted For={node.voted_for}, Commit Index={node.commit_index}"
                )
            if final_leader:
                print(
                    f"\nElection successful. Leader is Node {final_leader.node_id} in term {final_leader.current_term}."
                )
            else:
                print(
                    "\nElection inconclusive or no stable leader found within simulation pulses."
                )

        total_messages = sum(node.messages_sent for node in raft_nodes)
        return total_runtime, total_messages
