import numpy as np
import random
from connect4.policy import Policy


class ClaimEvenPolicy(Policy):
    def __init__(self):
        self.move_order = [3, 2, 4, 1, 5, 0, 6]

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        player = self._get_current_player(s)
        opp = -player
        legal_moves = self._get_legal_moves(s)

        # Prioridad 1: Movimiento ganador inmediato
        for col in sorted(legal_moves, key=lambda c: self.move_order.index(c)):
            new_board = self._simulate_move(s, col, player)
            if self._get_winner(new_board) == player:
                return col

        # Prioridad 2: Bloquear amenaza inmediata del rival
        threat_cols = []
        for col in legal_moves:
            opp_board = self._simulate_move(s, col, opp)
            if self._get_winner(opp_board) == opp:
                threat_cols.append(col)
        if threat_cols:
            return sorted(threat_cols, key=lambda c: self.move_order.index(c))[0]

        # Prioridad 3: Claim Even - Jugar en columnas de altura impar (para reclamar altura par)
        heights = self._get_heights(s)
        odd_height_cols = [c for c in legal_moves if heights[c] % 2 == 1]
        if odd_height_cols:
            return sorted(odd_height_cols, key=lambda c: self.move_order.index(c))[0]

        # Prioridad 4: Centro preferido
        return sorted(legal_moves, key=lambda c: self.move_order.index(c))[0]

    def _get_heights(self, board: np.ndarray) -> np.ndarray:
        heights = np.zeros(7, dtype=int)
        for c in range(7):
            for r in range(6):
                if board[r, c] != 0:
                    heights[c] = 6 - r
                    break
        return heights

    def _get_winner(self, board: np.ndarray) -> int:
        # Horizontal
        for r in range(6):
            for c in range(4):
                p = board[r, c]
                if p != 0 and np.all(board[r, c:c+4] == p):
                    return p
        # Vertical
        for c in range(7):
            for r in range(3):
                p = board[r, c]
                if p != 0 and np.all(board[r:r+4, c] == p):
                    return p
        # Diagonal /
        for r in range(3):
            for c in range(4):
                p = board[r, c]
                if p != 0 and all(board[r + i, c + i] == p for i in range(4)):
                    return p
        # Diagonal \
        for r in range(3):
            for c in range(3, 7):
                p = board[r, c]
                if p != 0 and all(board[r + i, c - i] == p for i in range(4)):
                    return p
        return 0

    def _get_legal_moves(self, board: np.ndarray) -> list[int]:
        return [c for c in range(7) if board[0, c] == 0]

    def _simulate_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        new_board = board.copy()
        for r in range(5, -1, -1):
            if new_board[r, col] == 0:
                new_board[r, col] = player
                return new_board
        return new_board  # No debería llegar aquí con legal moves

    def _get_current_player(self, board: np.ndarray) -> int:
        num_red = np.count_nonzero(board == -1)
        num_yellow = np.count_nonzero(board == 1)
        return -1 if num_red == num_yellow else 1