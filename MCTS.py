import math
import random


# ---------------------------------------------------------
# Grid environment
# ---------------------------------------------------------

GRID = [
    ["S", "A", "B"],
    ["C", "D", "E"],
    ["F", "G", "T"]
]

ROWS = 3
COLS = 3

START = (0, 0)
GOAL = (2, 2)

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}


def next_state(state, action):
    """Return next state after taking an action."""

    r, c = state
    dr, dc = ACTIONS[action]

    nr = r + dr
    nc = c + dc

    # Invalid movement: remain in the same state
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        return state

    return (nr, nc)


def reward(state):
    """Reward associated with arriving at a state."""

    if state == GOAL:
        return 10

    # Example dangerous cell
    if state == (1, 1):
        return -5

    return -1


def is_terminal(state):
    return state == GOAL


# ---------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------

class Node:

    def __init__(self, state, parent=None, action=None):

        self.state = state
        self.parent = parent
        self.action = action

        self.children = {}

        self.visits = 0
        self.total_reward = 0.0

        # Actions available from this state
        self.untried_actions = []

        for action_name in ACTIONS:
            next_s = next_state(state, action_name)

            # Don't consider actions that do not move
            if next_s != state:
                self.untried_actions.append(action_name)

    def fully_expanded(self):
        return len(self.untried_actions) == 0


# ---------------------------------------------------------
# UCT
# ---------------------------------------------------------

def uct_value(child, parent, C=1.414):

    if child.visits == 0:
        return float("inf")

    exploitation = child.total_reward / child.visits

    exploration = C * math.sqrt(
        math.log(parent.visits) / child.visits
    )

    return exploitation + exploration


# ---------------------------------------------------------
# Selection
# ---------------------------------------------------------

def select_child(node):

    return max(
        node.children.values(),
        key=lambda child: uct_value(child, node)
    )


# ---------------------------------------------------------
# Expansion
# ---------------------------------------------------------

def expand(node):

    action = node.untried_actions.pop()

    new_state = next_state(node.state, action)

    child = Node(
        state=new_state,
        parent=node,
        action=action
    )

    node.children[action] = child

    return child


# ---------------------------------------------------------
# Simulation / Rollout
# ---------------------------------------------------------

def simulate(state, max_steps=20):

    total_reward = 0

    current_state = state

    for _ in range(max_steps):

        if is_terminal(current_state):
            break

        possible_actions = []

        for action in ACTIONS:
            ns = next_state(current_state, action)

            if ns != current_state:
                possible_actions.append(action)

        action = random.choice(possible_actions)

        current_state = next_state(
            current_state,
            action
        )

        total_reward += reward(current_state)

        if is_terminal(current_state):
            break

    return total_reward


# ---------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------

def backpropagate(node, result):

    current = node

    while current is not None:

        current.visits += 1
        current.total_reward += result

        current = current.parent


# ---------------------------------------------------------
# One complete MCTS simulation
# ---------------------------------------------------------

def mcts_iteration(root):

    # -------------------------------
    # 1. Selection
    # -------------------------------

    node = root

    while (
        node.fully_expanded()
        and not is_terminal(node.state)
        and node.children
    ):

        node = select_child(node)

    # -------------------------------
    # 2. Expansion
    # -------------------------------

    if (
        not is_terminal(node.state)
        and node.untried_actions
    ):

        node = expand(node)

    # -------------------------------
    # 3. Simulation
    # -------------------------------

    result = simulate(node.state)

    # -------------------------------
    # 4. Backpropagation
    # -------------------------------

    backpropagate(node, result)


# ---------------------------------------------------------
# Display root statistics
# ---------------------------------------------------------

def print_tree_statistics(root):

    print("\nRoot state:", GRID[root.state[0]][root.state[1]])

    print(
        f"{'Action':<10}"
        f"{'Visits':<10}"
        f"{'Average Reward':<20}"
    )

    print("-" * 40)

    for action, child in root.children.items():

        average = (
            child.total_reward / child.visits
            if child.visits > 0
            else 0
        )

        print(
            f"{action:<10}"
            f"{child.visits:<10}"
            f"{average:<20.3f}"
        )


# ---------------------------------------------------------
# Run MCTS
# ---------------------------------------------------------

def run_mcts(number_of_simulations=100):

    root = Node(START)

    for simulation in range(1, number_of_simulations + 1):

        mcts_iteration(root)

        # Display progress
        if simulation in [1, 5, 10, 25, 50, 100]:

            print("\n===================================")
            print("After", simulation, "simulations")
            print("===================================")

            print_tree_statistics(root)

    # Choose action with highest visit count

    best_action = max(
        root.children.values(),
        key=lambda child: child.visits
    )

    print("\n===================================")
    print("FINAL DECISION")
    print("===================================")

    print(
        "Selected action:",
        best_action.action
    )

    print(
        "Visits:",
        best_action.visits
    )

    print(
        "Average reward:",
        best_action.total_reward /
        best_action.visits
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    run_mcts(100)