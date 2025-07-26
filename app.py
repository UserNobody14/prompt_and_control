import asyncio
import concurrent.futures
import threading
from gameboard import Player
from llm import get_llm_proposed_moves
from ui_display import GameBoardUI
from async_voice_controller import SimpleAsyncVoiceController
import tkinter as tk

def execute_game_loop(ui, voice_controller):
    """Demo function that moves the red piece up every 2 seconds using voice transcripts."""

    # Get transcripts from voice controller
    full_transcript = voice_controller.get_full_transcript()
    prompt = ui.game_board.to_prompt(Player.PLAYER)
    new_text = ui.transcript.add_message(prompt, full_transcript)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        ai_future = executor.submit(get_llm_proposed_moves, ui.game_board, Player.ENEMY, None)
        user_future = executor.submit(get_llm_proposed_moves, ui.game_board, Player.PLAYER, ui.transcript.conversation)
        
        # Get results from both calls
        ai_selected_moves = ai_future.result()
        user_selected_moves = user_future.result()

    # Execute the turn and get results including win condition
    turn_result = ui.game_board.execute_turn(
        {
            **user_selected_moves,
            **ai_selected_moves,
        }
    )

    # Check if game is over from the turn result
    if turn_result.get("game_over", False):
        winner = turn_result.get("winner")
        if winner:
            print(f"🎉 Game Over! {winner.value} wins by first capture!")
        else:
            print("🎉 Game Over!")

        # Show the game over screen immediately
        ui.show_game_over(victor=winner)

        # Schedule async cleanup after showing the screen
        def cleanup():
            # Create a new event loop for cleanup since we can't use asyncio.run()
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(voice_controller.stop_listening())
            finally:
                loop.close()

        # Run cleanup in a separate thread to avoid blocking
        threading.Thread(target=cleanup, daemon=True).start()
        return

    ui.update_display()

    # Schedule the next move in 2 seconds
    ui.master.after(1000, lambda: execute_game_loop(ui, voice_controller))


async def async_main():
    """Async main function that integrates voice controller with the game."""
    # Initialize voice controller
    voice_controller = SimpleAsyncVoiceController()

    try:
        # Start voice listening
        await voice_controller.start_listening()
        print("🎤 Voice controller started")

        # Create tkinter UI in main thread
        root = tk.Tk()
        app = GameBoardUI(root)

        # Start the demo animation after 1 second
        root.after(200, lambda: execute_game_loop(app, voice_controller))

        root.mainloop()

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        await voice_controller.stop_listening()


def main():
    """Entry point that runs the async main function."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
