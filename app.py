import asyncio
import concurrent.futures
import threading
from gameboard import Player
from llm import get_llm_proposed_moves
from ui_display import GameBoardUI
from async_voice_controller import SimpleAsyncVoiceController
import tkinter as tk

class GameManager:
    """Manages the game state and provides restart functionality."""
    
    def __init__(self):
        self.ui = None
        self.voice_controller = None
        self.game_running = False
        self.current_after_id = None
        
    def set_components(self, ui, voice_controller):
        """Set the UI and voice controller components."""
        self.ui = ui
        self.voice_controller = voice_controller
        
    def start_game_loop(self):
        """Start or restart the game loop."""
        self.game_running = True
        self.execute_game_loop()
        
    def stop_game_loop(self):
        """Stop the current game loop."""
        self.game_running = False
        # Cancel any pending after callbacks
        if self.current_after_id:
            self.ui.master.after_cancel(self.current_after_id)
            self.current_after_id = None
        
    def restart_game(self):
        """Restart the game - called by the UI restart callback."""
        print("🔄 Restarting game...")
        
        # Reset the transcript manager for a fresh conversation
        from transcript_manager import TranscriptManager
        self.ui.transcript = TranscriptManager()
        
        self.start_game_loop()

    def execute_game_loop(self):
        """Game loop that processes moves and checks for victory conditions."""
        if not self.game_running:
            return

def execute_game_loop(ui, voice_controller):
    """Demo function that moves the red piece up every 2 seconds using voice transcripts."""

    # Get transcripts from voice controller
    full_transcript = voice_controller.get_full_transcript()
    prompt = ui.game_board.to_prompt(Player.PLAYER)
    new_text = ui.transcript.add_message(prompt, full_transcript)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        ai_future = executor.submit(
            get_llm_proposed_moves, ui.game_board, Player.ENEMY, None
        )
        user_future = executor.submit(
            get_llm_proposed_moves,
            ui.game_board,
            Player.PLAYER,
            ui.transcript.conversation,
        )

        # Get results from both calls
        ai_selected_moves = ai_future.result()
        user_selected_moves = user_future.result()

    print(f"Last heard: {new_text}")
    if new_text:
        ui.set_last_heard(new_text)

    # Execute the turn and get results including win condition
    turn_result = ui.game_board.execute_turn(
        {
            **user_selected_moves,
            **ai_selected_moves,
        }
    )

        # Get transcripts from voice controller
        full_transcript = self.voice_controller.get_full_transcript()
        prompt = self.ui.game_board.to_prompt(Player.PLAYER)
        new_text = self.ui.transcript.add_message(prompt, full_transcript)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            ai_future = executor.submit(get_llm_proposed_moves, self.ui.game_board, Player.ENEMY, None)
            user_future = executor.submit(get_llm_proposed_moves, self.ui.game_board, Player.PLAYER, self.ui.transcript.conversation)
            
            # Get results from both calls
            ai_selected_moves = ai_future.result()
            user_selected_moves = user_future.result()

        # Execute the turn and get results including win condition
        turn_result = self.ui.game_board.execute_turn(
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
            self.ui.show_game_over(victor=winner)
            
            # Stop the game loop
            self.stop_game_loop()
            return

        self.ui.update_display()

        # Schedule the next move in 1 second if game is still running
        if self.game_running:
            self.current_after_id = self.ui.master.after(1000, lambda: self.execute_game_loop())


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
        
        # Create game manager
        game_manager = GameManager()
        game_manager.set_components(app, voice_controller)
        
        # Set up the restart callback in the UI
        app.set_restart_callback(game_manager.restart_game)

<<<<<<< HEAD
        # Start the game loop after a short delay
        root.after(200, lambda: game_manager.start_game_loop())
=======
        # Start the demo animation after 1 second
        root.after(1000, lambda: execute_game_loop(app, voice_controller))
>>>>>>> main

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
