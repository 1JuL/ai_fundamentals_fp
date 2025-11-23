import numpy as np
import math

class MCTSNode:
    def __init__(self, state: np.ndarray, player: int, parent=None, action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children:list["MCTSNode"] = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self, legal_moves: list[int]) -> bool:
        return len(self.children) == len(legal_moves)

    def best_child(self, c_param: float = math.sqrt(2)) -> "MCTSNode":
        choices_weights = [
            (child.wins / child.visits)
            + c_param * math.sqrt((math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, action: int, new_state: np.ndarray, next_player: int) -> "MCTSNode":
        child = MCTSNode(new_state, next_player, self, action)
        self.children.append(child)
        return child