import numpy as np
import hashlib
import json
import os
from connect4.policy import Policy

class ALjuriRuiz(Policy):

    def __init__(self, gamma=0.99, q_filename="Q_table_NuevoCleaned.json"):
        # Memoria entre partidas (ya entrenada)
        self.Q = {}
        self.N = {}
        self.gamma = gamma

        # Construir ruta absoluta al archivo dentro del paquete
        base_path = os.path.dirname(__file__)      
        q_path = os.path.join(base_path, q_filename)

        # Cargar la memoria aprendida
        try:
            with open(q_path, "r") as f:
                data = json.load(f)
            for key, val in data["Q"].items():
                s, a = key.split("|")
                self.Q[(s, int(a))] = val
            for key, val in data["N"].items():
                s, a = key.split("|")
                self.N[(s, int(a))] = val
            print("[INFO] Q-table cargada con", len(self.Q), "entradas")
        except:
            print("[WARNING] No se encontró Q_table.json. Se jugará sin memoria.")

    # UTILIDADES

    def encode(self, board):
        return hashlib.sha1(board.tobytes()).hexdigest()

    def mount(self, time_out=None):
        pass

    # POLICY FINAL = Q + tácticas + MCTS fuerte

    def act(self, board: np.ndarray) -> int:

        reds = np.sum(board == -1)
        yellows = np.sum(board == 1)
        player = -1 if reds == yellows else 1

        legal = [c for c in range(7) if board[0, c] == 0]
        state = self.encode(board)

        # 1. Win/Block
        tact = self.immediate_tactics(board, player, legal)
        if tact is not None:
            return tact

        # 2. Si el estado está en la Q-table → greedy
        q_candidates = [(self.Q[(state, a)], a) for a in legal if (state, a) in self.Q]
        if q_candidates:
            return max(q_candidates)[1]

        # 3. Si no está en memoria → usar MCTS fuerte
        return self.mcts(board, player, legal)

    # ---------------------------------------
    # TÁCTICAS INMEDIATAS
    # ---------------------------------------

    def immediate_tactics(self, b, player, legal):

        def apply_move(board, col, p):
            newb = board.copy()
            for r in range(5, -1, -1):
                if newb[r, col] == 0:
                    newb[r, col] = p
                    break
            return newb

        def winner(bb):
            # Horizontal
            for r in range(6):
                for c in range(4):
                    line = bb[r, c:c+4]
                    if np.all(line == 1): return 1
                    if np.all(line == -1): return -1
            # Vertical
            for c in range(7):
                col = bb[:, c]
                for r in range(3):
                    line = col[r:r+4]
                    if np.all(line == 1): return 1
                    if np.all(line == -1): return -1
            # Diag derecha
            for r in range(3):
                for c in range(4):
                    diag = np.array([bb[r+i,c+i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1
            # Diag izquierda
            for r in range(3):
                for c in range(3,7):
                    diag = np.array([bb[r+i,c-i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1
            return 0

        # Win
        for m in legal:
            if winner(apply_move(b, m, player)) == player:
                return m

        # Block
        opp = -player
        for m in legal:
            if winner(apply_move(b, m, opp)) == opp:
                return m

        return None

    # MCTS FUERTE (ORIGINAL)

    def mcts(self, board, player, legal):
        ITER = 50
        C = 1.41

        def apply_move(b, col, p):
            newb = b.copy()
            for r in range(5, -1, -1):
                if newb[r, col] == 0:
                    newb[r, col] = p
                    break
            return newb

        def winner(b):
            # Horizontal
            for r in range(6):
                for c in range(4):
                    line = b[r, c:c+4]
                    if np.all(line == 1): return 1
                    if np.all(line == -1): return -1
            # Vertical
            for c in range(7):
                col = b[:, c]
                for r in range(3):
                    line = col[r:r+4]
                    if np.all(line == 1): return 1
                    if np.all(line == -1): return -1
            # Diag derecha
            for r in range(3):
                for c in range(4):
                    diag = np.array([b[r+i, c+i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1
            # Diag izquierda
            for r in range(3):
                for c in range(3,7):
                    diag = np.array([b[r+i, c-i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1

            if np.all(b[0] != 0):
                return 2
            return 0

        def rollout(b, p):
            current = p
            bb = b.copy()
            while True:
                w = winner(bb)
                if w != 0:
                    return w
                legal2 = [c for c in range(7) if bb[0, c] == 0]
                mv = np.random.choice(legal2)
                bb = apply_move(bb, mv, current)
                current = -current

        # ----------- UCB --------------
        def select_ucb(moves, wins, plays):
            total = sum(plays[m] for m in moves) + 1e-9
            best_score = -1
            best_move = moves[0]
            for m in moves:
                if plays[m] == 0:
                    return m
                exploit = wins[m] / plays[m]
                explore = C * np.sqrt(np.log(total) / plays[m])
                if exploit + explore > best_score:
                    best_score = exploit + explore
                    best_move = m
            return best_move

        # ----------- MCTS MAIN LOOP --------------

        wins = {m: 0 for m in legal}
        plays = {m: 0 for m in legal}

        for _ in range(ITER):
            move = select_ucb(legal, wins, plays)
            b2 = apply_move(board, move, player)
            reward = rollout(b2, -player)

            plays[move] += 1
            if reward == player:
                wins[move] += 1
            elif reward == 2:
                wins[move] += 0.5

        # robust child
        return max(legal, key=lambda m: plays[m])
