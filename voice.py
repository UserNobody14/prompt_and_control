import speech_recognition as sr
import threading
import time
from typing import Callable, Optional
import queue


class VoiceController:
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the voice controller with simple, reliable speech recognition.
        
        Args:
            callback: Function to call when text is transcribed
        """
        self.callback = callback
        self.is_listening = False
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise on initialization
        print("Initializing speech recognition...")
        print("Adjusting for ambient noise... (this may take a moment)")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✅ Speech recognition initialized successfully!")
        
        # Configure recognizer settings
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3

    def record_and_transcribe(self):
        """Record audio and transcribe it using Google Speech Recognition."""
        try:
            print("🎤 Listening... (say something)")
            
            # Listen for audio with timeout
            with self.microphone as source:
                # Listen for audio input
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
            
            print("🔄 Processing audio...")
            
            try:
                # Use Google Speech Recognition (free, no API key needed)
                text = self.recognizer.recognize_google(audio)
                text = text.strip()
                
                if text:
                    print(f"📝 Transcribed: '{text}'")
                    
                    # Send text to callback if provided
                    if self.callback:
                        self.callback(text)
                    
                    return text
                else:
                    print("🔇 No speech detected")
                    return None
                    
            except sr.UnknownValueError:
                print("🔇 Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"❌ Speech recognition error: {e}")
                return None
                
        except sr.WaitTimeoutError:
            # Timeout is normal - just continue listening
            return None
        except Exception as e:
            print(f"❌ Error during recording: {e}")
            return None

    def continuous_listen(self):
        """Continuously listen for speech in a loop."""
        print("🎯 Starting continuous voice recognition...")
        print("💬 Say something like: 'Move players to attack the other team'")
        print("🛑 Press Ctrl+C to stop")
        
        while self.is_listening:
            try:
                self.record_and_transcribe()
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping voice controller...")
                break
            except Exception as e:
                print(f"❌ Error in continuous listening: {e}")
                time.sleep(1)  # Wait a bit before retrying

    def start_listening(self):
        """Start listening to the microphone in a separate thread."""
        if self.is_listening:
            print("Already listening!")
            return
        
        self.is_listening = True
        
        # Start listening in a separate thread
        self.listen_thread = threading.Thread(target=self.continuous_listen, daemon=True)
        self.listen_thread.start()
        
        try:
            # Keep main thread alive
            while self.is_listening:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping voice controller...")
            self.stop_listening()

    def stop_listening(self):
        """Stop listening to the microphone."""
        self.is_listening = False
        print("🔴 Voice controller stopped")

    def set_callback(self, callback: Callable[[str], None]):
        """Set the callback function for transcribed text."""
        self.callback = callback

    def record_once(self):
        """Record a single audio sample and return the transcribed text."""
        print("🎤 Please speak now...")
        
        with self.microphone as source:
            # Brief adjustment for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(source, phrase_time_limit=5)
        
        print("🔄 Transcribing...")
        
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"📝 Transcribed: '{text}'")
            return text.strip()
        except sr.UnknownValueError:
            print("🔇 Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None


def text_received_handler(text: str):
    """
    Default handler for received text.
    Replace this with your own logic to send text to main app.
    """
    print(f"🎮 Game Command: '{text}'")
    
    # Example: Basic command detection
    text_lower = text.lower()
    if "attack" in text_lower:
        print("⚔️ Attack command detected!")
    elif "move" in text_lower:
        print("🏃 Move command detected!")
    elif "defend" in text_lower:
        print("🛡️ Defend command detected!")
    elif "stop" in text_lower or "halt" in text_lower:
        print("✋ Stop command detected!")


def main():
    """Main function to run the voice controller."""
    # Create voice controller with callback
    voice_controller = VoiceController(callback=text_received_handler)
    
    try:
        # Start listening
        voice_controller.start_listening()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        voice_controller.stop_listening()


if __name__ == "__main__":
    main()
