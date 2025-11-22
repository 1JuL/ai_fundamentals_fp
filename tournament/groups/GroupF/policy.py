import numpy as np
from connect4.policy import Policy


class MinimaxPolicy(Policy):
    def __init__(self, max_depth: int = 6):
        # Comentario: Inicializamos la profundidad máxima de búsqueda y valores para la heurística
        self.max_depth = max_depth
        self.move_order = [3, 2, 4, 1, 5, 0, 6]  # Orden preferido: centro primero para mejor poda
        self.win_value = 1000000  # Valor para victoria inmediata
        self.window_scores = {4: self.win_value, 3: 1000, 2: 100, 1: 10}  # Puntuaciones para líneas propias (2,3,4)

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        # Comentario: Obtenemos el jugador actual y movimientos legales
        current_player = self._get_current_player(s)
        legal_moves = self._get_legal_moves(s)

        # Comentario: Búsqueda Minimax con Alfa-Beta: exploramos el árbol alternando MAX (nosotros) y MIN (oponente)
        # para maximizar nuestro beneficio y minimizar el del rival, podando ramas innecesarias
        best_score = -self.win_value
        best_col = legal_moves[0]  # Fallback
        alpha = -self.win_value
        beta = self.win_value
        ordered_moves = self._ordered_moves(legal_moves)
        for col in ordered_moves:
            new_board = self._simulate_move(s, col, current_player)
            score = self._minimax(new_board, self.max_depth - 1, alpha, beta, False, current_player)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, score)
        return best_col

    def _minimax(self, board: np.ndarray, depth: int, alpha: int, beta: int, maximizing: bool, root_player: int) -> int:
        # Comentario: Chequeo de estado terminal: victoria, derrota o empate
        winner = self._get_winner(board)
        if winner != 0:
            if winner == root_player:
                return self.win_value + depth * 10  # Preferir victorias rápidas
            return -self.win_value - depth * 10  # Penalizar derrotas rápidas
        if depth == 0 or not self._get_legal_moves(board):
            # Comentario: Evaluación heurística en profundidad máxima: premia líneas propias y penaliza las del oponente
            return self._heuristic(board, root_player)

        # Comentario: Ordenamos movimientos para mejor poda Alfa-Beta
        legal_moves = self._ordered_moves(self._get_legal_moves(board))
        turn_player = root_player if maximizing else -root_player

        if maximizing:
            value = -self.win_value
            for col in legal_moves:
                new_board = self._simulate_move(board, col, turn_player)
                value = max(value, self._minimax(new_board, depth - 1, alpha, beta, False, root_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Poda Beta: descartamos rama que no puede mejorar
            return value
        else:
            value = self.win_value
            for col in legal_moves:
                new_board = self._simulate_move(board, col, turn_player)
                value = min(value, self._minimax(new_board, depth - 1, alpha, beta, True, root_player))
                beta = min(beta, value)
                if alpha >= beta:
                    break  # Poda Alfa: descartamos rama que no puede empeorar para oponente
            return value

    def _heuristic(self, board: np.ndarray, player: int) -> int:
        # Comentario: Evaluamos contando ventanas de 4 posiciones en todas direcciones,
        # premiando configuraciones favorables (2,3,4 fichas propias en ventana vacía para oponente)
        own_score = self._count_windows(board, player)
        opp_score = self._count_windows(board, -player)
        return own_score - opp_score

    def _count_windows(self, board: np.ndarray, player: int) -> int:
        score = 0
        # Horizontal
        for r in range(6):
            for c in range(4):
                window = board[r, c:c+4]
                score += self._window_score(window, player)
        # Vertical
        for c in range(7):
            for r in range(3):
                window = board[r:r+4, c]
                score += self._window_score(window, player)
        # Diagonal /
        for r in range(3):
            for c in range(4):
                window = [board[r+i, c+i] for i in range(4)]
                score += self._window_score(np.array(window), player)
        # Diagonal \
        for r in range(3):
            for c in range(3, 7):
                window = [board[r+i, c-i] for i in range(4)]
                score += self._window_score(np.array(window), player)
        return score

    def _window_score(self, window: np.ndarray, player: int) -> int:
        # Comentario: Puntuación por ventana: solo si no hay fichas oponente, premia según conteo propio
        p_count = np.count_nonzero(window == player)
        o_count = np.count_nonzero(window == -player)
        if o_count > 0:
            return 0
        return self.window_scores.get(p_count, 0)

    def _get_winner(self, board: np.ndarray) -> int:
        # Comentario: Chequea victoria en todas direcciones
        for r in range(6):
            for c in range(4):
                p = board[r, c]
                if p != 0 and np.all(board[r, c:c+4] == p):
                    return p
        for c in range(7):
            for r in range(3):
                p = board[r, c]
                if p != 0 and np.all(board[r:r+4, c] == p):
                    return p
        for r in range(3):
            for c in range(4):
                p = board[r, c]
                if p != 0 and all(board[r+i, c+i] == p for i in range(4)):
                    return p
        for r in range(3):
            for c in range(3, 7):
                p = board[r, c]
                if p != 0 and all(board[r+i, c-i] == p for i in range(4)):
                    return p
        return 0

    def _get_legal_moves(self, board: np.ndarray) -> list[int]:
        return [c for c in range(7) if board[0, c] == 0]

    def _ordered_moves(self, legal_moves: list[int]) -> list[int]:
        # Comentario: Ordena por centro para mejor poda
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