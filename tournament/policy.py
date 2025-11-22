import numpy as np
import math
import random
from connect4.policy import Policy


class MCTSNode:
    def __init__(self, state: np.ndarray, player: int, parent=None, action=None):
        self.state = state
        self.player = player  # Player who just moved to reach this state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = []
        self.visits = 0
        self.wins = 0  # Wins from the perspective of the player to move at this node

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


class SmartMCTS(Policy):
    def __init__(self, iterations: int = 200, c_param: float = 1.414, max_rollout_depth: int = 20):
        """
        iterations: número de iteraciones de MCTS por jugada.
        max_rollout_depth: profundidad máxima de las simulaciones aleatorias.
        """
        self.iterations = iterations
        self.c_param = c_param
        self.max_rollout_depth = max_rollout_depth

    def mount(self) -> None:
        pass


    #Gradescope
    def act(self, s: np.ndarray) -> int:
        root_player = self._get_current_player(s)
        legal_moves = self._get_legal_moves(s)
        if not legal_moves:
            raise ValueError("No legal moves available")

      

        # 0.1 Jugada ganadora inmediata propia
        for col in legal_moves:
            tmp_board = self._simulate_move(s, col, root_player)
            if self._get_winner(tmp_board) == root_player:
                return col

        # 0.2 Bloquear jugada ganadora inmediata del rival
        opp = -root_player
        for col in legal_moves:
            tmp_board = self._simulate_move(s, col, opp)
            if self._get_winner(tmp_board) == opp:
        
                
                return col

        # 1) MCTS
        root = MCTSNode(s, -root_player)

        for _ in range(self.iterations):
            node = root
            state = s.copy()

            # --------- Selection ---------
            while node.children and not self._is_terminal(state):
                node = node.best_child(self.c_param)
                state = self._simulate_move(state, node.action, -node.parent.player)

            # --------- Expansion ---------
            current_legal = self._get_legal_moves(state)
            if current_legal and not self._is_terminal(state):
                action = random.choice(current_legal)
                new_state = self._simulate_move(state, action, -node.player)
                node = node.expand(action, new_state, -node.player)

            # --------- Simulation (con límite de profundidad) ---------
            sim_state = node.state.copy()
            sim_player = -node.player
            depth = 0
            while not self._is_terminal(sim_state) and depth < self.max_rollout_depth:
                sim_actions = self._get_legal_moves(sim_state)
                if not sim_actions:
                    break
                sim_action = random.choice(sim_actions)
                sim_state = self._simulate_move(sim_state, sim_action, sim_player)
                sim_player = -sim_player
                depth += 1

            # --------- Backpropagation ---------
            winner = self._get_winner(sim_state)
            if winner == root_player:
                result = 1
            elif winner == -root_player:
                result = -1
            else:
                result = 0

            current_node = node
            while current_node is not None:
                current_node.visits += 1
                # Desde la perspectiva de la "side to move" en el nodo:
                if -current_node.player == root_player:
                    current_node.wins += result
                else:
                    current_node.wins += -result
                current_node = current_node.parent

        # --------- Elegir la mejor acción ---------
        if not root.children:
            return int(random.choice(legal_moves))

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    # ========================= Helpers ========================= #

    def _get_winner(self, board: np.ndarray) -> int:
        # Horizontal
        for r in range(6):
            for c in range(4):
                p = board[r, c]
                if p != 0 and np.all(board[r, c : c + 4] == p):
                    return p
        # Vertical
        for c in range(7):
            for r in range(3):
                p = board[r, c]
                if p != 0 and np.all(board[r : r + 4, c] == p):
                    return p
        # Diagonal positiva
        for r in range(3):
            for c in range(4):
                p = board[r, c]
                if p != 0 and all(board[r + i, c + i] == p for i in range(4)):
                    return p
        # Diagonal negativa
        for r in range(3):
            for c in range(3, 7):
                p = board[r, c]
                if p != 0 and all(board[r + i, c - i] == p for i in range(4)):
                    return p
        return 0

    def _is_terminal(self, board: np.ndarray) -> bool:
        return self._get_winner(board) != 0 or np.all(board[0] != 0)

    def _get_legal_moves(self, board: np.ndarray) -> list[int]:
        return [c for c in range(7) if board[0, c] == 0]

    def _simulate_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        new_board = board.copy()
        for r in range(5, -1, -1):
            if new_board[r, col] == 0:
                new_board[r, col] = player
                return new_board
        raise ValueError(f"Invalid column {col}")

    def _get_current_player(self, board: np.ndarray) -> int:
        num_red = np.count_nonzero(board == -1)
        num_yellow = np.count_nonzero(board == 1)
        # -1 empieza; si están iguales, le toca a -1
        return -1 if num_red == num_yellow else 1
