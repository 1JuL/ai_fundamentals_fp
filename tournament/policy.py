import numpy as np
import math
import random
import json
import os
from connect4.policy import Policy


class MCTSNode:
    def __init__(self, state: np.ndarray, player: int, parent=None, action=None):
        self.state = state
        self.player = player  # Player who just moved to reach this state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children:list["MCTSNode"] = []
        self.visits = 0
        self.wins = 0  # Wins from the perspective of the player to move at this node

    def is_fully_expanded(self, legal_moves: list[int]) -> bool:
        return len(self.children) == len(legal_moves)

    def best_child(self, c_param: float = math.sqrt(2)) -> "MCTSNode":
        # Uso UCB1
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
    def __init__(
        self,
        iterations: int = 50, 
        c_param: float = math.sqrt(2), 
        max_rollout_depth: int = 20,
        internal_iterations: int = 30, 
        max_trial_depth:int = 8,       # longitud máxima del trial externo
        q_path: str = "q_values.json"  # ruta de q-table
        
    ):
        """
        iterations: número de iteraciones de MCTS por jugada.
        max_rollout_depth: profundidad máxima de las simulaciones aleatorias.
        """
        self.iterations = iterations
        self.c_param = c_param
        self.max_rollout_depth = max_rollout_depth
        self.internal_iterations = internal_iterations
        self.max_trial_depth = max_trial_depth
        self.q_path = q_path
        
        self.q_table: dict[str, float] = {}
        self.sa_counts: dict[str, int] = {}
        

    def mount(self) -> None:
        """
        
        Cargar q-table desde fichero un json
        
        """
        pass
        if os.path.exists(self.q_path):
            try:
                with open(self.q_path, "r") as f:
                    data = json.load(f)

                self.q_table = {str(k): float(v) for k, v in data.get("q_table", {}).items()}
                self.sa_counts = {str(k): int(v) for k, v in data.get("sa_counts", {}).items()}
            except Exception:
                # si algo sale mal leyendo, empezamos de cero
                self.q_table = {}
                self.sa_counts = {}
        else:
            self.q_table = {}
            self.sa_counts = {}
    def _save(self) ->None:
        """
        Guarda q-table en disco (persistencia)
        """
        data = {
            "q_table": self.q_table,
            "sa_counts": self.sa_counts
        }
        with open(self.q_path, "w") as f:
            json.dump(data, f)

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




        # ==== GPI con Trial-Based Online Policy Improvement ====
        
        
        for _ in range(self.iterations):
            trial, final_state = self._generate_trial(s, root_player)
            reward = self._evaluate_trial(final_state, root_player)
            self._update_q_global(trial, reward, root_player)

        # Elegir acción real con la política derivada de q̂ (π(s) = argmax_a q̂(s,a))
        action = self._select_final_action(s)
        
        self._save()
        
        return action


        # ==================== Trial-Based Online PI ==================== #
    def _generate_trial(
    self,
    state: np.ndarray,
    root_player: int
    ) -> tuple[list[tuple[np.ndarray, int, int]], np.ndarray]:
        
        
        # tuplas (estado,accion,jugador en el estado)
        trial: list[tuple[np.ndarray, int, int]] = []
        current_state = state.copy()
        current_player = root_player

        for _ in range(self.max_trial_depth):
            
            if self._is_terminal(current_state):
                break

            legal_moves = self._get_legal_moves(current_state)
            if not legal_moves:
                break

            # mini-GPI con MCTS
            q_prime = self._mini_mcts(current_state, current_player)
            action = self._select_from_qprime(q_prime, legal_moves)

            
            trial.append((current_state.copy(), action, current_player))

            current_state = self._simulate_move(current_state, action, current_player)
            current_player = -current_player

        return trial, current_state
    
    def _evaluate_trial(self, final_state: np.ndarray, root_player: int) -> int:
        """
        Política de evaluación (PE): asigna un valor al trial usando
        sólo el estado final (Monte Carlo simple).
        """
        winner = self._get_winner(final_state)
        if winner == root_player:
            return 1
        elif winner == -root_player:
            return -1
        else:
            return 0
        
    def _update_q_global(
        self,
        trial: list[tuple[np.ndarray, int, int]],
        reward: int,
        root_player: int
    ) -> None:
        """
        Actualiza q̂(s,a) usando el retorno del trial (Monte Carlo).
        Usamos promedio incremental con un contador de visitas por (s,a).
        Guardamos el valor desde la perspectiva del jugador que mueve en s.
        """
        for state, action, player_at_state in trial:
            # Ajustar signo del reward según el jugador en ese estado
            target = reward if player_at_state == root_player else -reward
            key = self._sa_key(state, action)

            n = self.sa_counts.get(key, 0) + 1
            self.sa_counts[key] = n

            old_q = self.q_table.get(key, 0.0)
            new_q = old_q + (target - old_q) / n
            self.q_table[key] = new_q
    
    # ==================== mini-GPI: MCTS interno ==================== #
    
    def _mini_mcts(self, state: np.ndarray, current_player: int) -> dict[int, float]:
        """
        MCTS interno para estimar q'

        Devuelve un diccionario {action: valor_est_mini}.
        """
        # Esto es lo mismo que hizo Esteban al principio
        
        root = MCTSNode(state.copy(), -current_player)  
        # -current_player acaba de mover,
        # le toca a current_player

        for _ in range(self.internal_iterations):
            node = root
            sim_state = state.copy()

            # --------- Selection ---------
            while node.children and not self._is_terminal(sim_state):
                node = node.best_child(self.c_param)
                # jugador que mueve en el nodo hijo es -node.player
                sim_state = self._simulate_move(sim_state, node.action, -node.parent.player)

            # --------- Expansion ---------
            if not self._is_terminal(sim_state):
                current_legal = self._get_legal_moves(sim_state)
                if current_legal:
                    action = random.choice(current_legal)
                    new_state = self._simulate_move(sim_state, action, -node.player)
                    node = node.expand(action, new_state, -node.player)
                    sim_state = new_state

            # --------- Simulation (rollout limitado) ---------
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
            if winner == current_player:
                result = 1
            elif winner == -current_player:
                result = -1
            else:
                result = 0

            current_node = node
            while current_node is not None:
                current_node.visits += 1
                # perspectiva del "side to move" en el nodo: -current_node.player
                if -current_node.player == current_player:
                    current_node.wins += result
                else:
                    current_node.wins += -result
                current_node = current_node.parent

        # Estimaciones q'(state, a) desde la raíz, siendo state el estado dado
        # es decir que la clave del diccionario es la accion tomada desde ese estado
        q_prime: dict[int, float] = {}
        for child in root.children:
            if child.visits > 0:
                q_prime[child.action] = child.wins / child.visits
        return q_prime
    
        # ==================== Selección de acción ==================== #

    def _select_from_qprime(self, q_prime: dict[int, float], legal_moves: list[int]) -> int:
        """
        Selecciona una acción a partir de q'(s,a).
        - Si no hay info en q_prime, elige aleatoria
        - Si hay info, usa greedy
        Esta función es utilizada para la generación del trial y la acción escogida no es la final
        """
        if not q_prime:
            return int(random.choice(legal_moves))

        # Aseguramos que todas las jugadas legales tienen algún valor
        vals = []
        for a in legal_moves:
            vals.append(q_prime.get(a, 0.0))

        # Greedy puro por simplicidad
        best_idx = int(np.argmax(vals))
        return legal_moves[best_idx]
    
    def _select_final_action(self, s: np.ndarray) -> int:
        """
        Política final π(s): elige la acción con mayor q̂(s,a).
        Si no hay info en q̂, elige aleatoriamente.
        """
        legal_moves = self._get_legal_moves(s)
        if not legal_moves:
            raise ValueError("No legal moves available")

        q_values = []  # q values dados por la politica global
        for a in legal_moves:
            key = self._sa_key(s, a)
            q_values.append(self.q_table.get(key, 0.0))

        best_idx = int(np.argmax(q_values))
        return legal_moves[best_idx]
    
     

    # ========================= Helpers ========================= #
    
    def _state_key(self, board: np.ndarray) -> str:
        # Representación compacta del estado para usar como clave de diccionario/JSON
        return board.tobytes().hex()

    def _sa_key(self, board: np.ndarray, action: int) -> str:
        """
        Codifica el par (s,a) de una forma simplificada para que la persistencia con un json funcione, lo único llamativo es que convierte el estado en un número hexadecimal
        """
        return f"{self._state_key(board)}:{action}"

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
