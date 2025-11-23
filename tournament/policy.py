import numpy as np
import math
import random
import json
import os
import time  # NUEVO: para medir deadline global dentro de act y evitar timeouts
from connect4.policy import Policy

from helpers import (
    ROWS, COLS,
    ordered_moves, get_heights,
    creates_odd_threat, claims_even, is_followup_threat,
    opponent_can_respond, is_useful_threat,
    heuristic, find_winning_move, is_block_move, creates_useful_threat
)


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


class SmartMCTS(Policy):
    def __init__(
        self,
        iterations: int = 50, 
        c_param: float = math.sqrt(2), 
        max_rollout_depth: int = 10,
        internal_iterations: int = 30, 
        max_trial_depth:int = 8,
        q_path: str = "q_values.json",

   
        # Agregamos pequeñas recompensas extra para guiar el aprendizaje

        block_bonus: float = 0.25, # por bloquear
        threat_bonus: float = 0.15, # por crear amenaza útil
        center_bonus: float = 0.05 # por jugar en columna 3 (centro teniendo en cuenta que de aquí salen más 4 en linea)
    ):
        self.iterations = iterations
        self.c_param = c_param
        self.max_rollout_depth = max_rollout_depth
        self.internal_iterations = internal_iterations
        self.max_trial_depth = max_trial_depth
        self.q_path = q_path
        
        # - Define orden al centro para expansión/rollouts y define parámetros de heurística tipo minimax.

        self.block_bonus = block_bonus
        self.threat_bonus = threat_bonus
        self.center_bonus = center_bonus

        self.move_order = [3, 2, 4, 1, 5, 0, 6]  # centro primero
        self.win_value = 1_000_000
        self.window_scores = {4: self.win_value, 3: 5000, 2: 200, 1: 10}

        self.q_table: dict[str, float] = {}
        self.sa_counts: dict[str, int] = {}

        # NUEVO: si Gradescope llama act() sin mount(), usamos un timeout seguro por defecto
        self.action_timeout = 9.5
        self._deadline = None
        
    def mount(self, time_out: int) -> None:
        """
        Gradescope manda time_out (segundos).
        Guardamos el deadline y ajustamos parámetros para no pasarnos.
        """
        self.action_timeout = float(time_out)  # segundos disponibles
        self._deadline = None  # se setea en act()

        # Como estás en modo MCTS puro (sin json),
        # bajamos costos para asegurar respuesta rápida.
        # Ojo: reglas Allis + rollout guiado ya ganan random.
        self.iterations = 0  # <- desactiva GPI/trials en este escenario
        self.internal_iterations = min(self.internal_iterations, 12)
        self.max_rollout_depth = min(self.max_rollout_depth, 8)
        self.max_trial_depth = min(self.max_trial_depth, 0)

        # No cargamos json porque no existe
        self.q_table = {}
        self.sa_counts = {}


        # tu load del json igual
        if os.path.exists(self.q_path):
            try:
                with open(self.q_path, "r") as f:
                    data = json.load(f)
                self.q_table = {str(k): float(v) for k, v in data.get("q_table", {}).items()}
                self.sa_counts = {str(k): int(v) for k, v in data.get("sa_counts", {}).items()}
            except Exception:
                self.q_table = {}
                self.sa_counts = {}
        else:
            self.q_table = {}
            self.sa_counts = {}


    def _save(self) ->None:
        data = {
            "q_table": self.q_table,
            "sa_counts": self.sa_counts
        }
        with open(self.q_path, "w") as f:
            json.dump(data, f)

    def act(self, s: np.ndarray) -> int:
        # NUEVO: deadline global (para nunca pasar 10s aunque no llamen mount)
        start = time.perf_counter()
        deadline = start + 0.9 * float(getattr(self, "action_timeout", 9.5))
        self._deadline = deadline

        root_player = self._get_current_player(s)
        legal_moves = self._get_legal_moves(s)
        if not legal_moves:
            raise ValueError("No legal moves available")

        for col in legal_moves:
            tmp_board = self._simulate_move(s, col, root_player)
            if self._get_winner(tmp_board) == root_player:
                return col

        opp = -root_player
        for col in legal_moves:
            tmp_board = self._simulate_move(s, col, opp)
            if self._get_winner(tmp_board) == opp:
                return col

       
        #  Con esto Calculamos alturas de columnas y su paridad (odd/even), además de filtrar jugadas útiles
        #  "useful_cols": jugadas que tienen sentido estratégico según la policy Allis 
        #  Haciendo que no nos centremos en nodos trash como diría el profe

        heights = get_heights(s)
        parities = [h % 2 for h in heights]

        candidate_moves = ordered_moves(legal_moves, self.move_order)

        # NUEVO: filtro de seguridad 1-ply (clave para mejorar Amarillo)
        # Evita jugadas que le regalan al rival una victoria inmediata al siguiente turno.
        safe_moves = []
        for c in candidate_moves:
            b2 = self._simulate_move(s, c, root_player)
            opp_can_win = False
            for oc in self._get_legal_moves(b2):
                b3 = self._simulate_move(b2, oc, opp)
                if self._get_winner(b3) == opp:
                    opp_can_win = True
                    break
            if not opp_can_win:
                safe_moves.append(c)
        candidate_moves = safe_moves if safe_moves else candidate_moves

        # NUEVO: si ya vamos tarde, devolvemos la mejor legal simple
        if time.perf_counter() > deadline:
            return candidate_moves[0] if candidate_moves else legal_moves[0]

        # Reglas Allis por color:
        #  Si soy Rojo busco odd threats desde alturas pares y priorizo las que el rival
        #  Si soy Amarillo aplico claim-even (ya que esta policy es mejor en amarillo y no en general)

        # Rojo:
        if root_player == -1:  
            odd_threat_cols = [  
                c for c in candidate_moves  # Miramos cada columna candidata 
                if parities[c] == 0  # Nos quedamos solo con columnas cuya altura actual es par
                and creates_odd_threat(s, c, root_player, heights)  # Amenazamos en una columna impar
            ]
            if odd_threat_cols: 
                unanswerable = [  # Filtramos amenazas que el rival no puede contestar bien (zugzwang)
                    c for c in odd_threat_cols 
                    if not opponent_can_respond(
                        s, c, root_player,
                        self._get_legal_moves, self._simulate_move, self._get_winner
                    )  # Guardamos las que dejan al rival sin respuesta segura
                ]
                if unanswerable:  
                    best_un = self._quick_minimax(s, unanswerable, root_player, deadline)
                    if best_un is not None:
                        return best_un
                    return unanswerable[0]  # Jugamos la mejor-primera 
                best_odd = self._quick_minimax(s, odd_threat_cols, root_player, deadline)
                if best_odd is not None:
                    return best_odd
                return odd_threat_cols[0]  # Si no hay incontestables, jugamos igual la mejor odd threat normal

            # No odd threat
            best = self._quick_minimax(s, candidate_moves, root_player, deadline)
            if best is not None:
                return best
        #Ammarillo:
        else:  
            claim_even_cols = [  # Creamos lista de columnas donde aplicar "claim-even"
                c for c in candidate_moves  # Recorremos columnas candidatas legales
                if parities[c] == 1  # columnas con altura impar
                and claims_even(s, c, root_player, heights)  # Y además jugar ahí reclama una casilla par útil
            ]
            if claim_even_cols: 
                followup_cols = [  # Filtramos las claim-even
                    c for c in claim_even_cols 
                    if is_followup_threat(s, c, opp)  # Nos quedamos con las que responden amenaza rival en esa columna
                ]
                if followup_cols:  # Si hay alguna follow-up disponible
                    best_follow = self._quick_minimax(s, followup_cols, root_player, deadline)
                    if best_follow is not None:
                        return best_follow
                    return followup_cols[0]  # Jugamos la mejor follow-up 
                
                # ========================= NUEVO (MEJOR CLAIM-EVEN PARA AMARILLO) =========================
                # Qué hace:
                # - Antes Amarillo hacía: return claim_even_cols[0]
                # - Eso a veces escoge una claim-even "segura" pero mala a largo plazo.
                # - Ahora evaluamos SOLO esas claim-even con tu minimax táctico 2-ply barato,
                #   y elegimos la que maximiza el peor caso contra Rojo.
                best_claim = self._quick_minimax(
                    s, claim_even_cols, root_player, deadline
                )
                if best_claim is not None:
                    return best_claim
                # ========================================================================================

                return claim_even_cols[0]  # Si no jugamos la mejor claim-even normal

            # ========================= NUEVO (AMARILLO TÁCTICO 2-PLY) =========================
            # Qué hace:
            # - Si no hay claim-even/followup, Amarillo evalúa cada jugada mirando:
            #   1) Heurística propia después de jugar.
            #   2) La PEOR respuesta posible de Rojo (min).
            # - Devuelve la jugada con mejor "peor caso".
            # Por qué:
            # - Evita las 1-2 jugadas trampa que random a veces encuentra por suerte.
            # - Es baratísimo en tiempo (máx 7*7 simulaciones).
            best_yellow = self._quick_minimax(s, candidate_moves, root_player, deadline)
            if best_yellow is not None:
                return best_yellow
            # ================================================================================

        # GPI 

        for _ in range(self.iterations):
            trial, final_state = self._generate_trial(s, root_player)
            reward = self._evaluate_trial(final_state, root_player)
            self._update_q_global(trial, reward, root_player)

        action = self._select_final_action(s)
        return action

    # ========================= NUEVO (HELPER AMARILLO 2-PLY) =========================
    def _quick_minimax(self, board: np.ndarray, moves: list[int], player: int, deadline: float):
        # Si estamos cortos de tiempo, no hacemos nada
        if time.perf_counter() > deadline:
            return None

        legal_len = len(self._get_legal_moves(board))
        max_depth = 4 + (8 - legal_len) * 2

        opp = -player
        best_score = -float("inf")
        best_move = None

        moves = ordered_moves(moves, self.move_order)

        for a in moves:
            if time.perf_counter() > deadline:
                break

            b_after = self._simulate_move(board, a, player)

            # Si mi jugada gana, listo
            if self._get_winner(b_after) == player:
                return a

            # Ahora llamamos al nivel min (opp)
            score = self._min_value(b_after, 1, -float("inf"), float("inf"), deadline, player, max_depth)

            if score > best_score:
                best_score = score
                best_move = a

        return best_move

    def _min_value(self, board: np.ndarray, depth: int, alpha: float, beta: float, deadline: float, maximizing_player: int, max_depth: int) -> float:
        if time.perf_counter() > deadline:
            return heuristic(board, maximizing_player, self.window_scores)

        winner = self._get_winner(board)
        if winner == maximizing_player:
            return self.win_value - depth
        elif winner == -maximizing_player:
            return -self.win_value + depth
        if np.all(board[0] != 0):  # full board, draw
            return 0

        if depth >= max_depth:
            return heuristic(board, maximizing_player, self.window_scores)

        min_score = float("inf")
        legal = ordered_moves(self._get_legal_moves(board), self.move_order)
        current_player = -maximizing_player
        for a in legal:
            if time.perf_counter() > deadline:
                return heuristic(board, maximizing_player, self.window_scores)

            b_after = self._simulate_move(board, a, current_player)
            score = self._max_value(b_after, depth + 1, alpha, beta, deadline, maximizing_player, max_depth)
            min_score = min(min_score, score)
            beta = min(beta, min_score)
            if beta <= alpha:
                break
        return min_score

    def _max_value(self, board: np.ndarray, depth: int, alpha: float, beta: float, deadline: float, maximizing_player: int, max_depth: int) -> float:
        if time.perf_counter() > deadline:
            return heuristic(board, maximizing_player, self.window_scores)

        winner = self._get_winner(board)
        if winner == maximizing_player:
            return self.win_value - depth
        elif winner == -maximizing_player:
            return -self.win_value + depth
        if np.all(board[0] != 0):
            return 0

        if depth >= max_depth:
            return heuristic(board, maximizing_player, self.window_scores)

        max_score = -float("inf")
        legal = ordered_moves(self._get_legal_moves(board), self.move_order)
        current_player = maximizing_player
        for a in legal:
            if time.perf_counter() > deadline:
                return heuristic(board, maximizing_player, self.window_scores)

            b_after = self._simulate_move(board, a, current_player)
            score = self._min_value(b_after, depth + 1, alpha, beta, deadline, maximizing_player, max_depth)
            max_score = max(max_score, score)
            alpha = max(alpha, max_score)
            if beta <= alpha:
                break
        return max_score
    # ================================================================================

    def _generate_trial(
        self,
        state: np.ndarray,
        root_player: int
    ) -> tuple[list[tuple[np.ndarray, int, int]], np.ndarray]:
        trial: list[tuple[np.ndarray, int, int]] = []
        current_state = state.copy()
        current_player = root_player

        for _ in range(self.max_trial_depth):
            if self._is_terminal(current_state):
                break

            legal_moves = self._get_legal_moves(current_state)
            if not legal_moves:
                break

            # PODA EN TRIAL - Repite la lógica useful_cols dentro del trial.

            heights = get_heights(current_state)
            parities = [h % 2 for h in heights]
            useful_cols = [
                c for c in legal_moves
                if is_useful_threat(current_state, c, current_player, heights, parities)
            ]
            pruned_moves = useful_cols if useful_cols else legal_moves
            pruned_moves = ordered_moves(pruned_moves, self.move_order)

            q_prime = self._mini_mcts(current_state, current_player)
            action = self._select_from_qprime(q_prime, pruned_moves)

            trial.append((current_state.copy(), action, current_player))

            current_state = self._simulate_move(current_state, action, current_player)
            current_player = -current_player

        return trial, current_state
    
    def _evaluate_trial(self, final_state: np.ndarray, root_player: int) -> float:
        winner = self._get_winner(final_state)
        if winner == root_player:
            return 1.0
        elif winner == -root_player:
            return -1.0

        # Heurística de Minimax
        # Si el trial termina sin ganador, no recompensamos con 0, usamos una heurística tipo minimax (ventanas propias - rival).

        h = heuristic(final_state, root_player, self.window_scores)
        return float(np.tanh(h / 5000.0))

    def _update_q_global(
        self,
        trial: list[tuple[np.ndarray, int, int]],
        reward: float,
        root_player: int
    ) -> None:
        for state, action, player_at_state in trial:
            target = reward if player_at_state == root_player else -reward

            # Rewards adicionales
            #  block_bonus si la acción bloquea victoria rival inmediata
            #  threat_bonus si crea amenaza útil
            #  center_bonus por jugar en columna 3

            shap = 0.0

            if is_block_move(
                state, action, player_at_state,
                self._get_legal_moves, self._simulate_move, self._get_winner
            ):
                shap += self.block_bonus

            if creates_useful_threat(
                state, action, player_at_state,
                self._simulate_move, self.window_scores
            ):
                shap += self.threat_bonus

            if action == 3:
                shap += self.center_bonus

            target = target + shap

            key = self._sa_key(state, action)
            n = self.sa_counts.get(key, 0) + 1
            self.sa_counts[key] = n

            old_q = self.q_table.get(key, 0.0)
            new_q = old_q + (target - old_q) / n
            self.q_table[key] = new_q
    
    def _mini_mcts(self, state: np.ndarray, current_player: int) -> dict[int, float]:
        root = MCTSNode(state.copy(), -current_player)

        for _ in range(self.internal_iterations):
            # NUEVO: corte por tiempo dentro de MCTS interno
            if self._deadline is not None and time.perf_counter() > self._deadline:
                break

            node = root
            sim_state = state.copy()

            while node.children and not self._is_terminal(sim_state):
                node = node.best_child(self.c_param)
                sim_state = self._simulate_move(sim_state, node.action, -node.parent.player)

            if not self._is_terminal(sim_state):
                current_legal = self._get_legal_moves(sim_state)
                if current_legal:

                    # Centro de expansión: Ordena jugadas al centro y elige aleatorio entre top3.

                    current_legal = ordered_moves(current_legal, self.move_order)
                    action = random.choice(current_legal[:3])

                    new_state = self._simulate_move(sim_state, action, -node.player)
                    node = node.expand(action, new_state, -node.player)
                    sim_state = new_state

            sim_player = -node.player
            depth = 0
            while not self._is_terminal(sim_state) and depth < self.max_rollout_depth:
                # NUEVO: corte por tiempo dentro de rollout
                if self._deadline is not None and time.perf_counter() > self._deadline:
                    break

                sim_actions = self._get_legal_moves(sim_state)
                if not sim_actions:
                    break

                # Rollout guiado (que no sea random puro)
                # Hace cosas obvias como si puedo ganar, gano; si debo bloquear, bloqueo
                # Cpsas no tan obvias como: si soy rojo, intento odd threats - si soy amarillo, intento claim-even

                win_now = find_winning_move(
                    sim_state, sim_player, sim_actions,
                    self._simulate_move, self._get_winner
                )
                if win_now is not None:
                    sim_action = win_now
                else:
                    block_now = find_winning_move(
                        sim_state, -sim_player, sim_actions,
                        self._simulate_move, self._get_winner
                    )
                    if block_now is not None:
                        sim_action = block_now
                    else:
                        heights = get_heights(sim_state)
                        parities = [h % 2 for h in heights]

                        if sim_player == -1:
                            odd_cols = [
                                c for c in sim_actions
                                if parities[c] == 0 and creates_odd_threat(sim_state, c, sim_player, heights)
                            ]
                            if odd_cols:
                                sim_action = ordered_moves(odd_cols, self.move_order)[0]
                            else:
                                sim_action = random.choice(ordered_moves(sim_actions, self.move_order)[:3])
                        else:
                            claim_cols = [
                                c for c in sim_actions
                                if parities[c] == 1 and claims_even(sim_state, c, sim_player, heights)
                            ]
                            if claim_cols:
                                sim_action = ordered_moves(claim_cols, self.move_order)[0]
                            else:
                                sim_action = random.choice(ordered_moves(sim_actions, self.move_order)[:3])

                sim_state = self._simulate_move(sim_state, sim_action, sim_player)
                sim_player = -sim_player
                depth += 1

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
                if -current_node.player == current_player:
                    current_node.wins += result
                else:
                    current_node.wins += -result
                current_node = current_node.parent

        q_prime: dict[int, float] = {}
        for child in root.children:
            if child.visits > 0:
                q_prime[child.action] = child.wins / child.visits
        return q_prime

    def _select_from_qprime(self, q_prime: dict[int, float], legal_moves: list[int]) -> int:
        if not q_prime:
            return int(random.choice(legal_moves))

        vals = []
        for a in legal_moves:
            vals.append(q_prime.get(a, 0.0))

        best_idx = int(np.argmax(vals))
        return legal_moves[best_idx]
    
    def _select_final_action(self, s: np.ndarray) -> int:
        legal_moves = self._get_legal_moves(s)
        if not legal_moves:
            raise ValueError("No legal moves available")

        q_values = []
        for a in legal_moves:
            key = self._sa_key(s, a)
            q_values.append(self.q_table.get(key, 0.0))

        # fallback a MCTS normal si json está vacio
        if max(q_values) == min(q_values):
            current_player = self._get_current_player(s)
            best_action = self._quick_minimax(s, ordered_moves(legal_moves, self.move_order), current_player, self._deadline)
            if best_action is not None:
                return best_action
            return ordered_moves(legal_moves, self.move_order)[0]

        best_idx = int(np.argmax(q_values))
        return legal_moves[best_idx]

    def _state_key(self, board: np.ndarray) -> str:
        return board.tobytes().hex()

    def _sa_key(self, board: np.ndarray, action: int) -> str:
        return f"{self._state_key(board)}:{action}"

    def _get_winner(self, board: np.ndarray) -> int:
        for r in range(6):
            for c in range(4):
                p = board[r, c]
                if p != 0 and np.all(board[r, c : c + 4] == p):
                    return p
        for c in range(7):
            for r in range(3):
                p = board[r, c]
                if p != 0 and np.all(board[r : r + 4, c] == p):
                    return p
        for r in range(3):
            for c in range(4):
                p = board[r, c]
                if p != 0 and all(board[r + i, c + i] == p for i in range(4)):
                    return p
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
        return -1 if num_red == num_yellow else 1