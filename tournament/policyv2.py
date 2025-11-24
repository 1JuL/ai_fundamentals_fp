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
        """
        Write this process' q_table to a per-pid parquet file:
        {self.q_path}.pid{pid}.parquet
        This avoids multiple processes trying to replace the same file.
        """
        try:
            rows = []
            for k, qv in self.q_table.items():
                if ":" in k:
                    state_part, a_part = k.rsplit(":", 1)
                else:
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
                df = pd.DataFrame([], columns=["state", "action", "q", "sa_count"])

            pid = os.getpid()
            tmp = f"{self.q_path}.pid{pid}.parquet.tmp"
            out = f"{self.q_path}.pid{pid}.parquet"

            df.to_parquet(tmp, index=False)
            os.replace(tmp, out)  # atomic rename on POSIX
        except Exception as e:
            print("Warning: parquet per-pid save failed:", e)

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

        # ------------------------------------------------------------------
        # PATCH: no early-return on Allis/minimax rules.
        # We compute them as priors, learn with GPI, then blend Q + priors.
        # ------------------------------------------------------------------

        # Reglas Allis por color:
        #  Si soy Rojo busco odd threats desde alturas pares y priorizo las que el rival
        #  Si soy Amarillo aplico claim-even (ya que esta policy es mejor en amarillo y no en general)

        rule_preferred: set[int] = set()

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
                    rule_preferred.update(unanswerable)
                else:
                    rule_preferred.update(odd_threat_cols)

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
                if followup_cols:
                    rule_preferred.update(followup_cols)
                else:
                    rule_preferred.update(claim_even_cols)

        # minimax suggestion as an additional (non-forced) prior
        mm_suggestion = quick_minimax(s, candidate_moves, root_player, deadline)
        if mm_suggestion is not None:
            rule_preferred.add(mm_suggestion)

        # GPI (online learning for this move)

        for _ in range(self.iterations):
            if time.perf_counter() > deadline:
                break
            trial, final_state = generate_trial(
                s, root_player,
                self.max_trial_depth,
                self.internal_iterations,
                self._deadline,
                self.c_param,
                self.max_rollout_depth
            )
            reward = evaluate_trial(final_state, root_player)
            update_q_global(
                trial, reward, root_player,
                self.block_bonus, self.threat_bonus, self.center_bonus,
                self.sa_counts, self.q_table
            )
        
        if self._autosave_parquet:
            try:
                self._save()
            except Exception as e:
                # don't crash the game if file I/O fails
                print("Warning: _save() failed in act:", e)

        # Blend Q(s,a) with rule priors
        RULE_BONUS = 0.30
        CENTER_BONUS_PRIOR = 0.05

        best_a = None
        best_score = -float("inf")

        for a in candidate_moves:
            sa = make_sa_key(s, a)
            q = self.q_table.get(sa, 0.0)

            prior = 0.0
            if a in rule_preferred:
                prior += RULE_BONUS
            if a == 3:
                prior += CENTER_BONUS_PRIOR

            score = q + prior
            if score > best_score:
                best_score = score
                best_a = a

        self.last_action_source = "blend"
        return best_a if best_a is not None else candidate_moves[0]
    
    
    def reload_global_file(self):
        if os.path.exists(self.q_path):
            try:
                df = pd.read_parquet(self.q_path)
                for _, row in df.iterrows():
                    state_key = str(row["state"])
                    a = int(row["action"])
                    sa_key = f"{state_key}:{a}"
                    self.q_table[sa_key] = float(row["q"])
                    self.sa_counts[sa_key] = int(row.get("sa_count", 1))
            except Exception as e:
                print("Warning: reload_global_file failed:", e)
