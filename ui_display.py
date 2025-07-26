import tkinter as tk
from tkinter import Canvas
from gameboard import GameBoard, Player


class GameBoardUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Game Board Display")
        self.master.geometry("600x600")

        # Initialize the game board
        self.game_board = GameBoard()

        # Create canvas
        self.canvas = Canvas(master, width=500, height=500, bg="white")
        self.canvas.pack(padx=50, pady=50)

        # Calculate cell size
        self.cell_size = 500 // self.game_board.size

        # Define colors for user pieces (4 different colors based on piece ID)
        self.user_colors = [
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FFD700",
        ]  # Red, Green, Blue, Gold

        # Enemy pieces are all black
        self.enemy_color = "#000000"

        self.draw_board()

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
            # Use different colors for player pieces based on their ID
            color_index = (piece.id - 1) % len(self.user_colors)
            color = self.user_colors[color_index]
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


def main():
    root = tk.Tk()
    app = GameBoardUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
