import numpy as np
from connect4.policy import Policy


class SmartMCTS(Policy):
    def _init_(self):
        self.move_order = [3, 2, 4, 1, 5, 0, 6]
        self.win_value = 10000000
        self.window_scores = {4: 1000000, 3: 100, 2: 10, 1: 1}

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        current_player = self._get_current_player(s)
        legal_moves = self._get_legal_moves(s)

        # immediate win check
        for col in legal_moves:
            temp_board = self._simulate_move(s, col, current_player)
            if self._get_winner(temp_board) == current_player:
                return col

        best_score = -self.win_value
        best_col = min(legal_moves, key=lambda c: self.move_order.index(c))
        ordered_moves = self._order_moves(legal_moves)
        for col in ordered_moves:
            new_board = self._simulate_move(s, col, current_player)
            score = self._minimax(new_board, 7, -self.win_value, self.win_value, False, current_player)
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    def _minimax(self, board: np.ndarray, depth: int, alpha: int, beta: int, maximizing: bool, root_player: int) -> int:
        winner = self._get_winner(board)
        if winner != 0:
            if winner == root_player:
                return self.win_value + depth * 100
            return -self.win_value - depth * 100

        legal_moves = self._get_legal_moves(board)
        if depth == 0 or not legal_moves:
            return self._evaluate(board, root_player)

        ordered_moves = self._order_moves(legal_moves)
        turn_player = root_player if maximizing else -root_player

        if maximizing:
            value = -self.win_value
            for col in ordered_moves:
                new_board = self._simulate_move(board, col, turn_player)
                score = self._minimax(new_board, depth - 1, alpha, beta, False, root_player)
                value = max(value, score)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = self.win_value
            for col in ordered_moves:
                new_board = self._simulate_move(board, col, turn_player)
                score = self._minimax(new_board, depth - 1, alpha, beta, True, root_player)
                value = min(value, score)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def _evaluate(self, board: np.ndarray, root_player: int) -> int:
        return self._count_windows(board, root_player) - self._count_windows(board, -root_player)

    def _count_windows(self, board: np.ndarray, player: int) -> int:
        score = 0
        # Horizontal
        for r in range(6):
            for c in range(4):
                window = board[r, c : c + 4]
                score += self._window_score(window, player)
        # Vertical
        for c in range(7):
            for r in range(3):
                window = board[r : r + 4, c]
                score += self._window_score(window, player)
        # Diagonal /
        for r in range(3):
            for c in range(4):
                sub = board[r : r + 4, c : c + 4]
                window = np.diagonal(sub)
                score += self._window_score(window, player)
        # Diagonal \
        for r in range(3):
            for c in range(3, 7):
                window = np.array([board[r + i, c - i] for i in range(4)])
                score += self._window_score(window, player)
        return score

    def _window_score(self, window: np.ndarray, player: int) -> int:
        p_count = np.count_nonzero(window == player)
        o_count = np.count_nonzero(window == -player)
        if o_count > 0:
            return 0
        return self.window_scores.get(p_count, 0)

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
        # Diagonal /
        for r in range(3):
            for c in range(4):
                p = board[r, c]
                if p != 0 and np.all([board[r + i, c + i] == p for i in range(4)]):
                    return p
        # Diagonal \
        for r in range(3):
            for c in range(3, 7):
                p = board[r, c]
                if p != 0 and np.all([board[r + i, c - i] == p for i in range(4)]):
                    return p
        return 0

    def _get_legal_moves(self, board: np.ndarray) -> list[int]:
        return [c for c in range(7) if board[0, c] == 0]

    def _order_moves(self, legal_moves: list[int]) -> list[int]:
        return sorted(legal_moves, key=lambda c: self.move_order.index(c))

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