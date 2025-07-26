import asyncio
import threading
from voice import VoiceController  # Assuming you have a VoiceController class defined elsewhere

class SimpleAsyncVoiceController:
    """Simpler version that runs everything in one process but still async."""
    
    def __init__(self):
        self.transcripts = []
        self.voice_controller = None
        self.listening_task = None
        self._stop_event = threading.Event()

    def _transcript_callback(self, transcript: str):
        """Handle transcripts from VoiceController."""
        self.transcripts.append(transcript)
        print(f"📝 Added transcript #{len(self.transcripts)}: '{transcript}'")

    def _run_voice_controller(self):
        """Run VoiceController in a separate thread."""
        self.voice_controller = VoiceController(callback=self._transcript_callback)
        
        try:
            self.voice_controller.start_listening()
        except Exception as e:
            print(f"Voice controller error: {e}")

    async def start_listening(self):
        """Start voice detection in background thread."""
        if self.listening_task and not self.listening_task.done():
            print("Already listening!")
            return

        self._stop_event.clear()
        
        # Start voice controller in separate thread
        voice_thread = threading.Thread(target=self._run_voice_controller, daemon=True)
        voice_thread.start()

        # Monitor the thread
        self.listening_task = asyncio.create_task(self._monitor_listening(voice_thread))
        print("🎤 Simple async voice controller started")

    async def _monitor_listening(self, voice_thread):
        """Monitor the voice controller thread."""
        while voice_thread.is_alive() and not self._stop_event.is_set():
            await asyncio.sleep(0.1)

    async def stop_listening(self):
        """Stop voice detection."""
        if self.voice_controller:
            self.voice_controller.stop_listening()

        self._stop_event.set()

        if self.listening_task:
            try:
                await asyncio.wait_for(self.listening_task, timeout=2.0)
            except asyncio.TimeoutError:
                self.listening_task.cancel()

        print("🔴 Simple async voice controller stopped")

    def get_all_transcripts(self):
        """Get all transcripts collected so far."""
        return self.transcripts.copy()

    def get_full_transcript(self, separator=" "):
        """Get all transcripts joined as a single string."""
        return separator.join(self.transcripts)

async def simple_main():
    """Simple single-process example."""
    voice_controller = SimpleAsyncVoiceController()

    try:
        await voice_controller.start_listening()

        # Your other async code can run here
        for i in range(60):  # Run for 60 seconds
            await asyncio.sleep(1)
            
            # You can access transcripts at any time
            transcripts = voice_controller.get_all_transcripts()
            if transcripts:
                full_text = voice_controller.get_full_transcript()
                print(f"Current full transcript: '{full_text}'")

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        await voice_controller.stop_listening()

if __name__ == "__main__":
    asyncio.run(simple_main())

