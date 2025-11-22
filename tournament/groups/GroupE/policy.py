import numpy as np
import random
from connect4.policy import Policy

ROWS = 6
COLS = 7


class AllisPolicy(Policy):
    """
    Política inspirada en Victor Allis:
    - odd threats, claim even, base inverse, control básico de zugzwang
    - evita amenazas inútiles por paridad/posición
    - alpha-beta como fallback cuando reglas no deciden
    """

    def __init__(self, search_depth=6):
        self.search_depth = search_depth
        self.move_order = [3, 2, 4, 1, 5, 0, 6]  # centro primero
        self.win_value = 1000000
        self.threat_value = 1000

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        player = self._get_current_player(s)
        legal_moves = self._get_legal_moves(s)

        # 1) Ganar inmediato si se puede
        for col in self._ordered_moves(legal_moves):
            new_board = self._simulate_move(s, col, player)
            if self._get_winner(new_board) == player:
                return col

        # 2) Bloquear victoria inmediata del rival
        threat_cols = []
        opp = -player
        for col in legal_moves:
            opp_board = self._simulate_move(s, col, opp)
            if self._get_winner(opp_board) == opp:
                threat_cols.append(col)
        if threat_cols:
            return self._ordered_moves(threat_cols)[0]

        # alturas y paridades por columna
        heights = self._get_heights(s)
        parities = [h % 2 for h in heights]  # 0 even, 1 odd

        # 3) Rojo (-1): crear odd threats desde altura even
        if player == -1:
            odd_threat_cols = [
                c for c in legal_moves
                if parities[c] == 0 and self._creates_odd_threat(s, c, player, heights)
            ]
            if odd_threat_cols:
                zugzwang_cols = [
                    c for c in odd_threat_cols
                    if not self._opponent_can_respond(s, c, player, heights)
                ]
                if zugzwang_cols:
                    return self._ordered_moves(zugzwang_cols)[0]
                return self._ordered_moves(odd_threat_cols)[0]

        # 4) Amarillo (1): claim-even / follow-up
        else:
            claim_even_cols = [
                c for c in legal_moves
                if parities[c] == 1 and self._claims_even(s, c, player, heights)
            ]
            if claim_even_cols:
                followup_cols = [
                    c for c in claim_even_cols
                    if self._is_followup_threat(s, c, opp)
                ]
                if followup_cols:
                    return self._ordered_moves(followup_cols)[0]
                return self._ordered_moves(claim_even_cols)[0]

        # 5) Amenazas útiles según paridad/posición
        useful_cols = [
            c for c in legal_moves
            if self._is_useful_threat(s, c, player, heights, parities)
        ]

        # fallback a alpha-beta
        if useful_cols:
            best_col = self._alpha_beta_search(s, player, legal_moves=useful_cols)
        else:
            best_col = self._alpha_beta_search(s, player)

        return best_col

    # ===================== SEARCH ===================== #

    def _alpha_beta_search(self, board: np.ndarray, player: int, legal_moves=None) -> int:
        if legal_moves is None:
            legal_moves = self._get_legal_moves(board)

        alpha = -self.win_value
        beta = self.win_value
        best_score = -self.win_value
        best_col = self._ordered_moves(legal_moves)[0]

        for col in self._ordered_moves(legal_moves):
            new_board = self._simulate_move(board, col, player)
            score = self._minimax(new_board, self.search_depth - 1, alpha, beta, False, player)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, score)

        return best_col

    def _minimax(self, board: np.ndarray, depth: int, alpha: int, beta: int,
                 max_player: bool, root_player: int) -> int:
        winner = self._get_winner(board)
        if winner != 0:
            return (self.win_value if winner == root_player else -self.win_value) * (depth + 1)

        if depth == 0 or not self._get_legal_moves(board):
            return self._heuristic(board, root_player)

        legal_moves = self._ordered_moves(self._get_legal_moves(board))

        if max_player:
            value = -self.win_value
            for col in legal_moves:
                new_board = self._simulate_move(board, col, root_player)
                value = max(value, self._minimax(new_board, depth - 1, alpha, beta, False, root_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        else:
            value = self.win_value
            for col in legal_moves:
                new_board = self._simulate_move(board, col, -root_player)
                value = min(value, self._minimax(new_board, depth - 1, alpha, beta, True, root_player))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # ===================== HEURISTIC ===================== #

    def _heuristic(self, board: np.ndarray, player: int) -> int:
        """
        Heurística diferencial SIN recursión:
        score(player) - score(opponent)
        """
        score_player = self._heuristic_for_player(board, player)
        score_opp = self._heuristic_for_player(board, -player)
        return score_player - score_opp

    def _heuristic_for_player(self, board: np.ndarray, player: int) -> int:
        """
        Score absoluto para un jugador fijo (sin comparar).
        """
        score = 0
        heights = self._get_heights(board)
        parities = [h % 2 for h in heights]

        for c in range(COLS):
            if self._has_odd_threat(board, c, player):
                score += self.threat_value if parities[c] == 1 else -self.threat_value
            if self._has_claim_even(board, c, player):
                score += self.threat_value / 2

        # Zugzwang/paridad favorable simplificada
        zugzwang = sum(1 for p in parities if (p == 1 if player == -1 else p == 0))
        score += zugzwang * 100
        return score

    # ===================== ALLIS RULES ===================== #

    def _creates_odd_threat(self, board: np.ndarray, col: int, player: int, heights: list[int]) -> bool:
        new_height = heights[col] + 1
        if new_height % 2 == 1:
            return self._potential_threat(board, col, new_height, player)
        return False

    def _opponent_can_respond(self, board: np.ndarray, col: int, player: int, heights: list[int]) -> bool:
        opp = -player
        new_board = self._simulate_move(board, col, player)
        opp_legal = self._get_legal_moves(new_board)
        for opp_col in opp_legal:
            opp_board = self._simulate_move(new_board, opp_col, opp)
            if self._get_winner(opp_board) != player:
                return True
        return False

    def _claims_even(self, board: np.ndarray, col: int, player: int, heights: list[int]) -> bool:
        new_height = heights[col] + 1
        if new_height % 2 == 0:
            return self._potential_threat(board, col, new_height, player) or self._base_inverse_check(board, col, heights)
        return False

    def _base_inverse_check(self, board: np.ndarray, col: int, heights: list[int]) -> bool:
        if heights[col] == 0:
            return True
        return heights[col] % 2 == 1

    def _is_followup_threat(self, board: np.ndarray, col: int, opp: int) -> bool:
        return self._has_odd_threat(board, col, opp)

    def _is_useful_threat(self, board: np.ndarray, col: int, player: int,
                          heights: list[int], parities: list[int]) -> bool:
        new_height = heights[col] + 1
        if new_height > ROWS:
            return False

        r = ROWS - new_height
        above_r = r - 1
        if above_r >= 0 and board[above_r, col] == -player:
            return False

        global_parity = sum(ROWS - h for h in heights) % 2
        return (new_height % 2 == 1 if player == -1 else new_height % 2 == 0) \
               or global_parity == (1 if player == -1 else 0)

    def _potential_threat(self, board: np.ndarray, col: int, height: int, player: int) -> bool:
        r = ROWS - height
        count_h = 1 + sum(
            1 for dc in (-1, 1)
            if 0 <= col + dc < COLS and board[r, col + dc] == player
        )
        count_v = 1 + (
            1 if r + 1 < ROWS and board[r + 1, col] == player else 0
        )
        count_d1 = 1 + sum(
            1 for d in (-1, 1)
            if 0 <= col + d < COLS and 0 <= r + d < ROWS and board[r + d, col + d] == player
        )
        count_d2 = 1 + sum(
            1 for d in (-1, 1)
            if 0 <= col + d < COLS and 0 <= r - d < ROWS and board[r - d, col + d] == player
        )
        return max(count_h, count_v, count_d1, count_d2) >= 2

    def _has_odd_threat(self, board: np.ndarray, col: int, player: int) -> bool:
        for r in range(ROWS):
            if board[r, col] == player and (ROWS - r) % 2 == 1:
                if self._potential_threat(board, col, ROWS - r, player):
                    return True
        return False

    def _has_claim_even(self, board: np.ndarray, col: int, player: int) -> bool:
        for r in range(ROWS):
            if board[r, col] == player and (ROWS - r) % 2 == 0:
                if self._potential_threat(board, col, ROWS - r, player):
                    return True
        return False

    # ===================== BASIC HELPERS ===================== #

    def _ordered_moves(self, moves: list[int]) -> list[int]:
        return sorted(moves, key=lambda c: self.move_order.index(c))

    def _get_winner(self, board: np.ndarray) -> int:
        for r in range(ROWS):
            for c in range(COLS - 3):
                if board[r, c] != 0 and np.all(board[r, c:c+4] == board[r, c]):
                    return board[r, c]
        for c in range(COLS):
            for r in range(ROWS - 3):
                if board[r, c] != 0 and np.all(board[r:r+4, c] == board[r, c]):
                    return board[r, c]
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if board[r, c] != 0 and all(board[r + i, c + i] == board[r, c] for i in range(4)):
                    return board[r, c]
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                if board[r, c] != 0 and all(board[r + i, c - i] == board[r, c] for i in range(4)):
                    return board[r, c]
        return 0

    def _get_legal_moves(self, board: np.ndarray) -> list[int]:
        return [c for c in range(COLS) if board[0, c] == 0]

    def _simulate_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        new_board = board.copy()
        for r in range(ROWS - 1, -1, -1):
            if new_board[r, col] == 0:
                new_board[r, col] = player
                return new_board
        raise ValueError(f"Invalid column {col}")

    def _get_current_player(self, board: np.ndarray) -> int:
        num_moves = np.count_nonzero(board)
        return -1 if num_moves % 2 == 0 else 1

    def _get_heights(self, board: np.ndarray) -> list[int]:
        heights = [0] * COLS
        for c in range(COLS):
            for r in range(ROWS):
                if board[r, c] != 0:
                    heights[c] = ROWS - r
                    break
        return heights
