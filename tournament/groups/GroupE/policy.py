import numpy as np
import random
from connect4.policy import Policy

# Comentario: Definimos constantes para el tablero estándar de Connect 4 (6 filas x 7 columnas)
ROWS = 6
COLS = 7

# Comentario: Clase principal de la política basada en la estrategia de Victor Allis.
# Integra reglas formales (odd threats, claim even, base inverse, zugzwang control),
# seguimiento de paridad, detección de amenazas inútiles, y búsqueda alpha-beta cuando las reglas no deciden.
# Nota: Como rojo (-1, primer jugador) busca forzar victoria; como amarillo (1) defiende.
class AllisPolicy(Policy):
    def __init__(self, search_depth=6):
        self.search_depth = search_depth  # Profundidad de búsqueda para alpha-beta cuando reglas no deciden
        self.move_order = [3, 2, 4, 1, 5, 0, 6]  # Orden preferido: centro primero
        self.win_value = 1000000  # Valor para victoria
        self.threat_value = 1000  # Valor base para amenazas

    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        # Comentario: Obtenemos jugador actual y movimientos legales
        player = self._get_current_player(s)
        legal_moves = self._get_legal_moves(s)

        # Comentario: Prioridad 1: Buscar movimiento ganador inmediato (regla básica)
        for col in self._ordered_moves(legal_moves):
            new_board = self._simulate_move(s, col, player)
            if self._get_winner(new_board) == player:
                return col

        # Comentario: Prioridad 2: Bloquear victoria inmediata del oponente (defensa)
        threat_cols = []
        opp = -player
        for col in legal_moves:
            opp_board = self._simulate_move(s, col, opp)
            if self._get_winner(opp_board) == opp:
                threat_cols.append(col)
        if threat_cols:
            # Comentario: Elegir el bloqueo preferido por orden central
            return self._ordered_moves(threat_cols)[0]

        # Comentario: Calcular alturas y paridades de columnas (clave para odd/even threats y claim even)
        heights = self._get_heights(s)
        parities = [h % 2 for h in heights]  # 0: even, 1: odd (desde abajo, fila 1 es odd, 2 even, etc.)

        # Comentario: Prioridad 3: Como primer jugador (rojo, -1), crear "odd threat": jugar en columna con altura even (para hacer odd threat encima)
        # Odd threat: amenaza en posición impar (contando desde abajo: 1,3,5 impares)
        if player == -1:  # Primer jugador
            odd_threat_cols = [c for c in legal_moves if parities[c] == 0 and self._creates_odd_threat(s, c, player, heights)]
            if odd_threat_cols:
                # Comentario: Elegir la que fuerza zugzwang (oponente no puede responder sin perder)
                zugzwang_cols = [c for c in odd_threat_cols if not self._opponent_can_respond(s, c, player, heights)]
                if zugzwang_cols:
                    return self._ordered_moves(zugzwang_cols)[0]
                return self._ordered_moves(odd_threat_cols)[0]

        # Comentario: Prioridad 4: Como segundo jugador (amarillo, 1), "claim even": jugar en columna con altura odd (para reclamar even encima)
        # Follow-up: responder en misma columna si conviene (base inverse para asegurar al menos una en par base)
        else:  # Segundo jugador
            claim_even_cols = [c for c in legal_moves if parities[c] == 1 and self._claims_even(s, c, player, heights)]
            if claim_even_cols:
                # Comentario: Priorizar follow-up si refuta amenaza blanca
                followup_cols = [c for c in claim_even_cols if self._is_followup_threat(s, c, opp)]
                if followup_cols:
                    return self._ordered_moves(followup_cols)[0]
                return self._ordered_moves(claim_even_cols)[0]

        # Comentario: Prioridad 5: Detectar y evitar amenazas inútiles (por paridad o posición)
        useful_cols = [c for c in legal_moves if self._is_useful_threat(s, c, player, heights, parities)]

        # Comentario: Si reglas no deciden, fallback a búsqueda alpha-beta con heurística (incluye zugzwang y paridad)
        if useful_cols:
            best_col = self._alpha_beta_search(s, player, legal_moves=useful_cols)
        else:
            best_col = self._alpha_beta_search(s, player)
        return best_col

    # Comentario: Función para búsqueda alpha-beta (inspirada en VICTOR: depth-first con transpositions, aquí simplificada)
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

    def _minimax(self, board: np.ndarray, depth: int, alpha: int, beta: int, max_player: bool, root_player: int) -> int:
        winner = self._get_winner(board)
        if winner != 0:
            return (self.win_value if winner == root_player else -self.win_value) * (depth + 1)  # Preferir wins rápidas
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

    # Comentario: Heurística: valora odd threats, control zugzwang (paridad favorable), claim even
    def _heuristic(self, board: np.ndarray, player: int) -> int:
        score = 0
        heights = self._get_heights(board)
        parities = [h % 2 for h in heights]
        for c in range(COLS):
            if self._has_odd_threat(board, c, player):
                score += self.threat_value if parities[c] == 1 else -self.threat_value  # Odd threat buena si impar
            if self._has_claim_even(board, c, player):
                score += self.threat_value / 2
        # Zugzwang: + si controlas paridad (más columnas con paridad favorable)
        zugzwang = sum(1 for p in parities if (p == 1 if player == -1 else p == 0))
        score += zugzwang * 100
        return score - self._heuristic(board, -player)  # Diferencial

    # Comentario: Detecta si movimiento crea odd threat (amenaza en posición impar)
    def _creates_odd_threat(self, board: np.ndarray, col: int, player: int, heights: list[int]) -> bool:
        new_height = heights[col] + 1
        if new_height % 2 == 1:  # Nueva posición es impar
            # Verificar si crea amenaza (potencial 4-in-row involucrando esta posición)
            return self._potential_threat(board, col, new_height, player)
        return False

    # Comentario: Verifica si oponente puede responder sin perder (zugzwang check)
    def _opponent_can_respond(self, board: np.ndarray, col: int, player: int, heights: list[int]) -> bool:
        opp = -player
        new_board = self._simulate_move(board, col, player)
        opp_legal = self._get_legal_moves(new_board)
        for opp_col in opp_legal:
            opp_board = self._simulate_move(new_board, opp_col, opp)
            if self._get_winner(opp_board) != player:  # No pierde inmediatamente
                return True
        return False

    # Comentario: Claim even: reclamar posición par crítica
    def _claims_even(self, board: np.ndarray, col: int, player: int, heights: list[int]) -> bool:
        new_height = heights[col] + 1
        if new_height % 2 == 0:  # Nueva posición par
            return self._potential_threat(board, col, new_height, player) or self._base_inverse_check(board, col, heights)
        return False

    # Comentario: Base inverse: asegurar al menos una en par base jugable
    def _base_inverse_check(self, board: np.ndarray, col: int, heights: list[int]) -> bool:
        if heights[col] == 0:  # Base vacía
            return True  # Asegurar par (2,4,6) eventualmente
        return heights[col] % 2 == 1  # Si odd actual, jugar hace even base

    # Comentario: Follow-up: si columna tiene amenaza oponente
    def _is_followup_threat(self, board: np.ndarray, col: int, opp: int) -> bool:
        return self._has_odd_threat(board, col, opp)

    # Comentario: Amenaza útil: no inútil por paridad o posición encima
    def _is_useful_threat(self, board: np.ndarray, col: int, player: int, heights: list[int], parities: list[int]) -> bool:
        new_height = heights[col] + 1
        if new_height > ROWS:
            return False
        # Inútil si paridad no permite explotar (e.g., threat encima de rival threat)
        if board[ROWS - new_height + 1, col] == -player:  # Encima de rival
            return False
        # Inútil si paridad global no favorece
        global_parity = sum(ROWS - h for h in heights) % 2  # Huecos restantes par/impar
        return (new_height % 2 == 1 if player == -1 else new_height % 2 == 0) or global_parity == (1 if player == -1 else 0)

    # Comentario: Verifica potencial amenaza (simplificado: chequea si contribuye a 3+ en línea potencial)
    def _potential_threat(self, board: np.ndarray, col: int, height: int, player: int) -> bool:
        r = ROWS - height
        # Chequea horizontal, vertical, diagonales para potencial 4
        # Simplificado: si hay 2+ adyacentes ya, es amenaza
        count_h = 1 + sum(1 for dc in [-1,1] if 0 <= col + dc < COLS and board[r, col + dc] == player)
        count_v = 1 + sum(1 for dr in [1] if r + dr < ROWS and board[r + dr, col] == player)  # Abajo
        count_d1 = 1 + sum(1 for d in [-1,1] if 0 <= col + d < COLS and r + d >= 0 and r + d < ROWS and board[r + d, col + d] == player)
        count_d2 = 1 + sum(1 for d in [-1,1] if 0 <= col + d < COLS and r - d >= 0 and r - d < ROWS and board[r - d, col + d] == player)
        return max(count_h, count_v, count_d1, count_d2) >= 2

    # Comentario: Chequea si hay odd threat en columna para jugador
    def _has_odd_threat(self, board: np.ndarray, col: int, player: int) -> bool:
        for r in range(ROWS):
            if board[r, col] == player and (ROWS - r) % 2 == 1:  # Posición impar
                if self._potential_threat(board, col, ROWS - r, player):
                    return True
        return False

    # Comentario: Chequea claim even en columna
    def _has_claim_even(self, board: np.ndarray, col: int, player: int) -> bool:
        for r in range(ROWS):
            if board[r, col] == player and (ROWS - r) % 2 == 0:  # Posición par
                if self._potential_threat(board, col, ROWS - r, player):
                    return True
        return False

    # Comentario: Ordena movimientos por preferencia central
    def _ordered_moves(self, moves: list[int]) -> list[int]:
        return sorted(moves, key=lambda c: self.move_order.index(c))

    # Funciones auxiliares estándar (get_winner, etc.) como en políticas anteriores
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
        return new_board

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