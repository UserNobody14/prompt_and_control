from gameboard import GameBoard, Direction


def main():
    """Demonstrate the game board functionality."""
    print("Creating 10x10 Game Board with 4 pieces per side...")

    # Create game board
    game = GameBoard()

    print("\nInitial board state:")
    print(game.display_board())

    print("\nGame state:")
    state = game.get_game_state()
    for key, value in state.items():
        print(f"{key}: {value}")

    # Demonstrate some moves
    print("\n--- Turn 1: Moving some pieces ---")
    moves = {
        1: Direction.UP,  # Player piece moves up
        2: Direction.LEFT,  # Player piece moves left
        5: Direction.DOWN,  # Enemy piece moves down
        6: Direction.RIGHT,  # Enemy piece moves right
    }

    result = game.execute_turn(moves)
    print("Move results:")
    for piece_id, move_result in result["move_results"].items():
        print(f"  Piece {piece_id}: {move_result}")

    if result["captured_pieces"]:
        print(f"Captured pieces: {result['captured_pieces']}")

    print("\nBoard after turn 1:")
    print(game.display_board())


if __name__ == "__main__":
    main()
