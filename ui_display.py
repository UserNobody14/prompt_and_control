import tkinter as tk
from tkinter import Canvas
from gameboard import GameBoard, Player, Color, Direction


class GameBoardUI:
    def __init__(self, master, game_board=None):
        self.master = master
        self.master.title("Game Board Display")
        self.master.geometry("600x600")

        # Initialize the game board (use provided one or create new)
        self.game_board = game_board if game_board is not None else GameBoard()

        # Create canvas
        self.canvas = Canvas(master, width=500, height=500, bg="white")
        self.canvas.pack(padx=50, pady=50)

        # Calculate cell size
        self.cell_size = 500 // self.game_board.size

        # Map Color enum to hex colors
        self.color_map = {
            Color.RED: "#FF0000",
            Color.GREEN: "#00FF00",
            Color.BLUE: "#0000FF",
            Color.YELLOW: "#FFD700",
        }

        # Enemy pieces are all black
        self.enemy_color = "#000000"

        self.draw_board()

    def set_gameboard(self, game_board):
        """Set a new gameboard and update the display."""
        self.game_board = game_board
        self.cell_size = (
            500 // self.game_board.size
        )  # Recalculate in case board size changed
        self.draw_board()

    def update_display(self):
        """Refresh the display to show current game board state."""
        self.draw_board()

    def refresh(self):
        """Alias for update_display() for convenience."""
        self.update_display()

    def draw_board(self):
        """Draw the game board and pieces."""
        self.canvas.delete("all")

        # Draw grid lines
        for i in range(self.game_board.size + 1):
            x = i * self.cell_size
            y = i * self.cell_size

            # Vertical lines
            self.canvas.create_line(
                x, 0, x, self.game_board.size * self.cell_size, fill="gray", width=1
            )
            # Horizontal lines
            self.canvas.create_line(
                0, y, self.game_board.size * self.cell_size, y, fill="gray", width=1
            )

        # Draw pieces
        for row in range(self.game_board.size):
            for col in range(self.game_board.size):
                piece = self.game_board.get_piece_at(row, col)
                if piece:
                    self.draw_piece(piece, row, col)

    def draw_piece(self, piece, row, col):
        """Draw a single piece on the board."""
        x1 = col * self.cell_size + 5
        y1 = row * self.cell_size + 5
        x2 = x1 + self.cell_size - 10
        y2 = y1 + self.cell_size - 10

        # Determine color
        if piece.owner == Player.PLAYER:
            # Use the piece's actual color
            color = self.color_map[piece.color]
        else:
            color = self.enemy_color

        # Draw piece as circle
        self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline="black", width=2)

        # Add piece ID as text in the center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        text_color = "white" if piece.owner == Player.ENEMY else "black"

        self.canvas.create_text(
            center_x,
            center_y,
            text=str(piece.id),
            fill=text_color,
            font=("Arial", 10, "bold"),
        )


def get_piece(game_board, player, color):
    """Find and return the red player piece."""
    for piece in game_board.pieces.values():
        if piece.owner == player and (color is None or piece.color == color):
            return piece
    return None


def demo_move_red_piece(ui):
    """Demo function that moves the red piece up every 2 seconds."""
    red_piece = get_piece(ui.game_board, Player.PLAYER, Color.RED)

    if red_piece is None:
        print("Demo ended: Red piece not found")
        return

    # Try to move the red piece up
    move_successful = ui.game_board.execute_turn(
        {
            red_piece.id: Direction.UP,
        }
    )

    if move_successful:
        ui.update_display()
        print(f"Red piece moved to position {red_piece.position}")

        # Schedule the next move in 2 seconds
        ui.master.after(2000, lambda: demo_move_red_piece(ui))
    else:
        print(f"Demo ended: Red piece at {red_piece.position} cannot move up anymore")


def main():
    root = tk.Tk()
    app = GameBoardUI(root)

    # Move the first black piece down and 3 to the right
    black_piece = get_piece(app.game_board, Player.ENEMY, None)
    if black_piece:
        app.game_board.move_piece(black_piece.id, Direction.DOWN)
        app.game_board.move_piece(black_piece.id, Direction.RIGHT)
        app.game_board.move_piece(black_piece.id, Direction.RIGHT)

    # Start the demo animation after 1 second
    root.after(1000, lambda: demo_move_red_piece(app))

    root.mainloop()


if __name__ == "__main__":
    main()
