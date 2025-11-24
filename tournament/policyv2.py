import numpy as np
import math
import random
import os
import time  # NUEVO: para medir deadline global dentro de act y evitar timeouts
import pandas as pd
from connect4.policy import Policy
from mctsnode import MCTSNode

from helpers import (
    ROWS, COLS,
    ordered_moves, get_heights,
    creates_odd_threat, claims_even, is_followup_threat,
    opponent_can_respond, is_useful_threat,
    heuristic, find_winning_move, is_block_move, creates_useful_threat
)

from board_utils import (
    get_winner, is_terminal, 
    get_legal_moves, simulate_move, get_current_player
)

from tactical_search import (
    quick_minimax, min_value, max_value
)

from gpi_core import (
    generate_trial, evaluate_trial, update_q_global,
    mini_mcts, select_from_qprime, select_final_action 
)

from gpi_core import sa_key as make_sa_key

class SmartMCTS(Policy):
    def __init__(
        self,
        iterations: int = 50, 
        c_param: float = math.sqrt(2), 
        max_rollout_depth: int = 10,
        internal_iterations: int = 30, 
        max_trial_depth:int = 8,
        q_path: str = "q_values.parquet",

   
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
        
        self.last_action_source = None
        
        self._autosave_parquet = True
        
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
        self.iterations = 50  # <- desactiva GPI/trials en este escenario
        self.internal_iterations = min(self.internal_iterations, 12)
        self.max_rollout_depth = min(self.max_rollout_depth, 8)
        self.max_trial_depth = min(self.max_trial_depth, 8)

        if os.path.exists(self.q_path):
            try:
                df = pd.read_parquet(self.q_path)
                # Expecting columns: state (str), action (int), q (float), sa_count (int) optionally.
                for _, row in df.iterrows():
                    state_key = str(row["state"])   # here state is the hex string produced by gpi_core.state_key
                    a = int(row["action"])
                    sa_key = f"{state_key}:{a}"     # <-- use colon, not pipe
                    self.q_table[sa_key] = float(row["q"])
                    self.sa_counts[sa_key] = int(row.get("sa_count", 1))
            except Exception as e:
                print("Warning: failed to load parquet q table:", e)
                self.q_table = {}
                self.sa_counts = {}
        else:
            self.q_table = {}
            self.sa_counts = {}

    def _save(self) -> None:
        try:
            rows = []
            for k, qv in self.q_table.items():
                # expected key format: "<state_hex>:<action>"
                if ":" in k:
                    state_part, a_part = k.rsplit(":", 1)
                else:
                    # fallback: store entire key as state and action=-1
                    state_part, a_part = k, -1
                rows.append({
                    "state": state_part,
                    "action": int(a_part),
                    "q": float(qv),
                    "sa_count": int(self.sa_counts.get(k, 0))
                })
            if rows:
                df = pd.DataFrame(rows, columns=["state", "action", "q", "sa_count"])
            else:
                # empty DataFrame with right columns
                df = pd.DataFrame([], columns=["state", "action", "q", "sa_count"])
            tmp = f"{self.q_path}.tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, self.q_path)
        except Exception as e:
            print("Warning: parquet save failed:", e)

    def act(self, s: np.ndarray) -> int:
        # NUEVO: deadline global (para nunca pasar 10s aunque no llamen mount)
        start = time.perf_counter()
        deadline = start + 0.9 * float(getattr(self, "action_timeout", 9.5))
        self._deadline = deadline

        root_player = get_current_player(board=s)
        legal_moves = get_legal_moves(s)
        if not legal_moves:
            raise ValueError("No legal moves available")

        for col in legal_moves:
            tmp_board = simulate_move(s, col, root_player)
            if get_winner(tmp_board) == root_player:
                self.last_action_source = "shortcircuit"
                return col

        opp = -root_player
        for col in legal_moves:
            tmp_board = simulate_move(s, col, opp)
            if get_winner(tmp_board) == opp:
                self.last_action_source = "shortcircuit"
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
            b2 = simulate_move(s, c, root_player)
            opp_can_win = False
            for oc in get_legal_moves(b2):
                b3 = simulate_move(b2, oc, opp)
                if get_winner(b3) == opp:
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
                        get_legal_moves, simulate_move, get_winner
                    )  # Guardamos las que dejan al rival sin respuesta segura
                ]
                if unanswerable:  
                    best_un = quick_minimax(s, unanswerable, root_player, deadline)
                    if best_un is not None:
                        return best_un
                    return unanswerable[0]  # Jugamos la mejor-primera 
                best_odd = quick_minimax(s, odd_threat_cols, root_player, deadline)
                if best_odd is not None:
                    return best_odd
                return odd_threat_cols[0]  # Si no hay incontestables, jugamos igual la mejor odd threat normal

            # No odd threat
            best = quick_minimax(s, candidate_moves, root_player, deadline)
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
                    best_follow = quick_minimax(s, followup_cols, root_player, deadline)
                    if best_follow is not None:
                        return best_follow
                    return followup_cols[0]  # Jugamos la mejor follow-up 
                
                # ========================= NUEVO (MEJOR CLAIM-EVEN PARA AMARILLO) =========================
                # Qué hace:
                # - Antes Amarillo hacía: return claim_even_cols[0]
                # - Eso a veces escoge una claim-even "segura" pero mala a largo plazo.
                # - Ahora evaluamos SOLO esas claim-even con tu minimax táctico 2-ply barato,
                #   y elegimos la que maximiza el peor caso contra Rojo.
                best_claim = quick_minimax(
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
            best_yellow = quick_minimax(s, candidate_moves, root_player, deadline)
            if best_yellow is not None:
                return best_yellow
            # ================================================================================
        
        # ========================
        # 1. Try to use stored Q-values
        # ========================

        # Collect all Q(s,a) for available actions
        q_available = []
        for a in legal_moves:
            sa = make_sa_key(s, a)
            if sa in self.q_table:
                q_available.append((self.q_table[sa], a))

        # If we have full Q info for all legal moves → choose best action directly
        if len(q_available) == len(legal_moves):
            # pick max reward for current player
            if root_player == 1:
                self.last_action_source = "q"
                return max(q_available)[1]
            else:
                self.last_action_source = "q"
                return min(q_available)[1]

        # Otherwise, continue to compute missing values

        # GPI 

        for _ in range(self.iterations):
            trial, final_state = generate_trial(s, root_player, self.max_trial_depth, self.internal_iterations, self._deadline, self.c_param, self.max_rollout_depth)
            reward = evaluate_trial(final_state, root_player)
            update_q_global(trial, reward, root_player, self.block_bonus, self.threat_bonus, self.center_bonus, self.sa_counts, self.q_table)
        
        if self._autosave_parquet:
            try:
                self._save()
            except Exception as e:
                # don't crash the game if file I/O fails
                print("Warning: _save() failed in act:", e)

        action = select_final_action(s, self._deadline, self.q_table)
        self.last_action_source = "mcts"
        return action