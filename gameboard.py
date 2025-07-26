from typing import Dict, List, Tuple, Optional
from enum import Enum


class Player(Enum):
    PLAYER = "Player"
    ENEMY = "Enemy"


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @classmethod
    def from_str(cls, direction: str) -> Optional['Direction']:
        """Convert string to Direction enum."""
        direction = direction.lower()
        if direction == "up":
            return cls.UP
        elif direction == "down":
            return cls.DOWN
        elif direction == "left":
            return cls.LEFT
        elif direction == "right":
            return cls.RIGHT
        else:
            return None

class Color(Enum):
    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"
    YELLOW = "Yellow"

    @classmethod
    def from_idx(cls, i: int) -> 'Color':
        if i == 0:
            return cls.RED
        elif i == 1:
            return cls.BLUE
        elif i == 2:
            return cls.GREEN
        elif i == 3:
            return cls.YELLOW

    def to_prompt(self) -> str:
        """All colors represented by single uppercase character on the baord."""
        return self.value[0]


class Piece:
    """Individual game piece with unique ID and position."""

    def __init__(self, piece_id: int, owner: Player, row: int, col: int, color: Color):
        self.id = piece_id
        self.owner = owner
        self.row = row
        self.col = col
        self.position = (row, col)
        self.color = color

    def __repr__(self):
        return f"Piece({self.id}, {self.owner.value}, {self.position})"

    def to_prompt(self, player: Player) -> str:
        """ Friendly units represented as their color, enemy units as 'E'."""
        if player == self.owner:
            return self.color.to_prompt()
        else:
            return "E"


class GameBoard:
    """10x10 game board with piece management and game logic."""

    def __init__(self):
        self.size = 10
        self.board: List[List[Optional[Piece]]] = [
            [None for _ in range(self.size)] for _ in range(self.size)
        ]
        self.pieces: Dict[int, Piece] = {}
        self.next_piece_id = 1

        # Initialize with 4 pieces per side
        self._initialize_pieces()

    def _initialize_pieces(self):
        """Initialize 4 pieces for each player on opposite sides of the board."""
        # Player pieces start on bottom rows
        player_positions = [(8, 2), (8, 4), (8, 6), (8, 8)]
        # Enemy pieces start on top rows
        enemy_positions = [(1, 1), (1, 3), (1, 5), (1, 7)]

        # Create player pieces
        for i, (row, col) in enumerate(player_positions):
            piece = Piece(self.next_piece_id, Player.PLAYER, row, col, Color.from_idx(i))
            self.pieces[piece.id] = piece
            self.board[row][col] = piece
            self.next_piece_id += 1

        # Create enemy pieces
        for i, (row, col) in enumerate(enemy_positions):
            piece = Piece(self.next_piece_id, Player.ENEMY, row, col, Color.from_idx(i))
            self.pieces[piece.id] = piece
            self.board[row][col] = piece
            self.next_piece_id += 1

    def get_piece_at(self, row: int, col: int) -> Optional[Piece]:
        """Get piece at specified position."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.board[row][col]
        return None

    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= row < self.size and 0 <= col < self.size

    def get_adjacent_positions(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all valid adjacent positions (up, down, left, right)."""
        adjacent = []
        for direction in Direction:
            dr, dc = direction.value
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                adjacent.append((new_row, new_col))
        return adjacent

    def get_support_count(self, piece: Piece) -> Tuple[int, int]:
        """Get count of friendly and enemy pieces adjacent to this piece.
        Returns (friendly_count, enemy_count)."""
        friendly_count = 0
        enemy_count = 0

        for adj_row, adj_col in self.get_adjacent_positions(piece.row, piece.col):
            adj_piece = self.get_piece_at(adj_row, adj_col)
            if adj_piece:
                if adj_piece.owner == piece.owner:
                    friendly_count += 1
                else:
                    enemy_count += 1

        return friendly_count, enemy_count

    def check_captures(self) -> List[Piece]:
        """Check for pieces that should be captured based on support rules.
        A piece is captured if enemy support > friendly support."""
        captured_pieces = []

        for piece in self.pieces.values():
            friendly_support, enemy_support = self.get_support_count(piece)

            # If enemy pieces outnumber friendly support, piece is captured
            if enemy_support > friendly_support:
                captured_pieces.append(piece)

        return captured_pieces

    def remove_piece(self, piece: Piece):
        """Remove a piece from the board and pieces dictionary."""
        self.board[piece.row][piece.col] = None
        del self.pieces[piece.id]

    def move_piece(self, piece_id: int, direction: Direction) -> bool:
        """Attempt to move a piece in the specified direction.
        Returns True if move was successful, False otherwise."""
        if piece_id not in self.pieces:
            return False

        piece = self.pieces[piece_id]
        dr, dc = direction.value
        new_row = piece.row + dr
        new_col = piece.col + dc

        # Check if new position is valid
        if not self.is_valid_position(new_row, new_col):
            return False

        # Check if new position is occupied
        if self.get_piece_at(new_row, new_col) is not None:
            return False

        # Move the piece
        self.board[piece.row][piece.col] = None
        piece.row = new_row
        piece.col = new_col
        piece.position = (new_row, new_col)
        self.board[new_row][new_col] = piece

        return True

    def execute_turn(self, moves: Dict[int, Direction]) -> Dict[str, any]:
        """Execute a full turn with multiple piece moves.

        Args:
            moves: Dictionary mapping piece_id to Direction

        Returns:
            Dictionary containing move results and captures
        """
        move_results = {}
        conflicts = {}

        # First pass: validate all moves and detect conflicts
        for piece_id, direction in moves.items():
            if piece_id not in self.pieces:
                move_results[piece_id] = {"success": False, "reason": "Piece not found"}
                continue

            piece = self.pieces[piece_id]
            dr, dc = direction.value
            new_row = piece.row + dr
            new_col = piece.col + dc

            # Check bounds
            if not self.is_valid_position(new_row, new_col):
                move_results[piece_id] = {"success": False, "reason": "Out of bounds"}
                continue

            # Check if position is occupied
            if self.get_piece_at(new_row, new_col) is not None:
                move_results[piece_id] = {
                    "success": False,
                    "reason": "Position occupied",
                }
                continue

            # Check for conflicts with other moves this turn
            target_pos = (new_row, new_col)
            if target_pos in conflicts:
                # Multiple pieces trying to move to same position
                conflicts[target_pos].append(piece_id)
                move_results[piece_id] = {
                    "success": False,
                    "reason": "Movement conflict",
                }
            else:
                conflicts[target_pos] = [piece_id]
                move_results[piece_id] = {"success": True, "new_position": target_pos}

        # Resolve conflicts - pieces moving to same position all fail
        for target_pos, piece_ids in conflicts.items():
            if len(piece_ids) > 1:
                for piece_id in piece_ids:
                    move_results[piece_id] = {
                        "success": False,
                        "reason": "Movement conflict",
                    }

        # Second pass: execute successful moves
        for piece_id, result in move_results.items():
            if result["success"]:
                piece = self.pieces[piece_id]
                new_row, new_col = result["new_position"]

                # Clear old position
                self.board[piece.row][piece.col] = None

                # Update piece position
                piece.row = new_row
                piece.col = new_col
                piece.position = (new_row, new_col)

                # Set new position
                self.board[new_row][new_col] = piece

        # Check for captures after all moves
        captured_pieces = self.check_captures()
        for captured_piece in captured_pieces:
            self.remove_piece(captured_piece)

        return {
            "move_results": move_results,
            "captured_pieces": [p.id for p in captured_pieces],
            "remaining_pieces": len(self.pieces),
        }

    def get_pieces_by_owner(self, owner: Player) -> List[Piece]:
        """Get all pieces belonging to a specific player."""
        return [piece for piece in self.pieces.values() if piece.owner == owner]

    def display_board(self) -> str:
        """Create a string representation of the board."""
        display = "   " + " ".join([str(i) for i in range(self.size)]) + "\n"

        for row in range(self.size):
            display += f"{row:2} "
            for col in range(self.size):
                piece = self.board[row][col]
                if piece is None:
                    display += ". "
                elif piece.owner == Player.PLAYER:
                    display += f"P{piece.id % 10} "  # P1, P2, etc.
                else:
                    display += f"E{piece.id % 10} "  # E1, E2, etc.
            display += "\n"

        return display

    def get_game_state(self) -> Dict[str, any]:
        """Get current game state information."""
        player_pieces = self.get_pieces_by_owner(Player.PLAYER)
        enemy_pieces = self.get_pieces_by_owner(Player.ENEMY)

        return {
            "board_size": self.size,
            "player_pieces": len(player_pieces),
            "enemy_pieces": len(enemy_pieces),
            "total_pieces": len(self.pieces),
            "piece_positions": {
                piece.id: piece.position for piece in self.pieces.values()
            },
        }

    def to_prompt(self, player: Player) -> str:
        """Generate a prompt representation of the game board for the specified player."""
        prompt = ""
        for row in range(self.size):
            for col in range(self.size):
                piece = self.get_piece_at(row, col)
                if piece is None:
                    prompt += "X "
                else:
                    prompt += piece.to_prompt(player) + " "
            prompt += "\n"
        return prompt.strip()
