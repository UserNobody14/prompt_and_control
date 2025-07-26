import queue
import threading
import time
import numpy as np
import sounddevice as sd
import whisper
import io
import wave
from typing import Callable, Optional
from collections import deque
import scipy.signal


class VoiceController:
    def __init__(self, callback: Optional[Callable[[str], None]] = None, 
                 model_size: str = "base", 
                 sample_rate: int = 16000,
                 chunk_duration: float = 3.0,
                 min_audio_length: float = 0.3):
        """
        Initialize the voice controller with advanced audio filtering.
        
        Args:
            callback: Function to call when text is transcribed
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            sample_rate: Audio sample rate
            chunk_duration: Duration of audio chunks to process (seconds)
            min_audio_length: Minimum audio length to process (seconds)
        """
        self.callback = callback
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.min_audio_length = min_audio_length
        
        # Advanced voice detection parameters (from Node.js code)
        self.VOICE_START_THRESHOLD = 0.015  # Level to START recording (normalized 0-1)
        self.VOICE_CONTINUE_THRESHOLD = 0.008  # Level to CONTINUE recording
        self.MIN_RECORDING_TIME = 0.3  # Minimum recording duration (300ms)
        self.SILENCE_THRESHOLD = 1.5  # 1.5 seconds of silence before ending
        self.NOISE_GATE_SAMPLES = 5  # Rolling buffer size for consistency check
        self.KEYBOARD_FILTER_TIME = 0.05  # Keyboard clicks < 50ms
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.recording_thread = None
        self.processing_thread = None
        
        # Advanced audio analysis buffers
        self.audio_samples = deque(maxlen=self.NOISE_GATE_SAMPLES)
        self.frequency_samples = deque(maxlen=10)
        
        # Recording session state
        self.recording_session_active = False
        self.recording_session_start_time = 0
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.silence_timer = None
        self.is_recording_audio = False
        
        # Audio buffer for current recording
        self.audio_buffer = []
        
        # Load Whisper model
        print(f"Loading Whisper model '{model_size}'...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded successfully!")

    def analyze_frequency_spectrum(self, audio_data):
        """
        Analyze frequency spectrum to detect voice characteristics vs noise/keyboard clicks.
        Based on the Node.js frequency analysis logic.
        """
        # Apply FFT to get frequency spectrum
        fft = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        # Human voice typically has energy in 85Hz - 3000Hz range
        voice_freq_mask = (freqs >= 85) & (freqs <= 3000)
        high_freq_mask = freqs >= 3000
        
        total_energy = np.sum(magnitude)
        voice_energy = np.sum(magnitude[voice_freq_mask])
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        
        if total_energy > 0:
            voice_ratio = voice_energy / total_energy
            high_freq_ratio = high_freq_energy / total_energy
        else:
            voice_ratio = 0
            high_freq_ratio = 0
        
        return {
            'voice_ratio': voice_ratio,
            'high_freq_ratio': high_freq_ratio,
            'total_energy': total_energy
        }

    def is_voice_activity(self, audio_level, audio_data):
        """
        Enhanced voice activity detection with keyboard filtering.
        Adapted from the Node.js isVoiceActivity function.
        """
        # Get frequency analysis
        freq_analysis = self.analyze_frequency_spectrum(audio_data)
        voice_ratio = freq_analysis['voice_ratio']
        high_freq_ratio = freq_analysis['high_freq_ratio']
        
        # Keyboard clicks have lots of high frequency energy, voice doesn't
        is_likely_keyboard = high_freq_ratio > 0.4 and voice_ratio < 0.3
        passes_voice_ratio = voice_ratio > 0.15
        
        if is_likely_keyboard:
            print("🔇 Filtered out keyboard click")
            return False
        
        return passes_voice_ratio

    def calculate_audio_consistency(self, audio_level):
        """
        Calculate audio level consistency to distinguish voice from transient noise.
        Voice should be relatively stable, keyboard clicks are spiky.
        """
        self.audio_samples.append(audio_level)
        
        if len(self.audio_samples) < 2:
            return True
        
        # Calculate variance of recent samples
        samples_array = np.array(self.audio_samples)
        avg_level = np.mean(samples_array)
        variance = np.var(samples_array)
        
        # Voice is consistent, keyboard clicks are very spiky
        is_consistent = variance < 0.001  # Normalized variance threshold
        sustained_voice = all(abs(sample - audio_level) < 0.01 for sample in self.audio_samples)
        
        return is_consistent or sustained_voice

    def should_start_recording(self, audio_level, audio_data):
        """
        Determine if we should start a new recording session.
        """
        level_check = audio_level >= self.VOICE_START_THRESHOLD
        voice_check = self.is_voice_activity(audio_level, audio_data)
        consistency_check = self.calculate_audio_consistency(audio_level)
        
        # High audio override: if audio is moderately loud, bypass some strict checks
        high_audio_override = audio_level > 0.05
        
        if high_audio_override:
            return level_check and (voice_check or consistency_check)
        
        return level_check and voice_check and consistency_check

    def should_continue_recording(self, audio_level):
        """
        Determine if we should continue the current recording session.
        """
        if not self.recording_session_active:
            return False
        
        session_duration = time.time() - self.recording_session_start_time
        
        # Always continue if under minimum time (bridges syllables)
        if session_duration < self.MIN_RECORDING_TIME:
            return True
        
        # After minimum time, use continue threshold
        return audio_level >= self.VOICE_CONTINUE_THRESHOLD

    def end_recording_session(self):
        """End the current recording session and process audio."""
        if not self.recording_session_active:
            return
        
        self.recording_session_active = False
        self.is_recording_audio = False
        self.last_speech_time = time.time()
        self.speech_start_time = 0
        
        total_session_time = time.time() - self.recording_session_start_time
        print(f"🔇 RECORDING SESSION ENDED - Total duration: {total_session_time:.1f}s")
        
        # Process the recorded audio if it's long enough
        if len(self.audio_buffer) > 0:
            audio_length = len(self.audio_buffer) / self.sample_rate
            if audio_length >= self.min_audio_length:
                print(f"🔊 Processing recorded audio ({audio_length:.1f}s)...")
                self.audio_queue.put(np.array(self.audio_buffer, dtype=np.float32))
            else:
                print(f"⏭️ Audio too short ({audio_length:.1f}s), skipping...")
        
        self.audio_buffer = []

    def audio_callback(self, indata, frames, time, status):
        """Enhanced audio callback with advanced voice detection."""
        if status:
            print(f"Audio input status: {status}")
        
        # Convert to mono if stereo
        if len(indata.shape) > 1:
            audio_data = indata[:, 0]
        else:
            audio_data = indata.flatten()
        
        # Calculate RMS level (normalized)
        audio_level = np.sqrt(np.mean(audio_data**2))
        
        # STABLE RECORDING SESSION LOGIC (adapted from Node.js)
        
        # Check if we should start a new recording session
        if not self.recording_session_active and self.should_start_recording(audio_level, audio_data):
            self.recording_session_active = True
            self.recording_session_start_time = time.time()
            self.speech_start_time = time.time()
            self.is_recording_audio = True
            self.audio_buffer = []
            
            print(f"🎤 RECORDING SESSION STARTED - Level: {audio_level:.3f}")
            
            # Clear any existing silence timer
            if self.silence_timer:
                self.silence_timer.cancel()
                self.silence_timer = None
        
        # Check if we should continue the current recording session
        elif self.recording_session_active:
            session_duration = time.time() - self.recording_session_start_time
            should_continue = self.should_continue_recording(audio_level)
            
            if should_continue:
                # Continue recording - clear any silence timer
                if self.silence_timer:
                    self.silence_timer.cancel()
                    self.silence_timer = None
                
                # Add audio to buffer
                if self.is_recording_audio:
                    self.audio_buffer.extend(audio_data)
            else:
                # Start silence timer if not already started
                if not self.silence_timer:
                    print(f"🔇 Voice dipped below continue threshold after {session_duration:.1f}s session")
                    
                    def silence_timeout():
                        self.end_recording_session()
                    
                    self.silence_timer = threading.Timer(self.SILENCE_THRESHOLD, silence_timeout)
                    self.silence_timer.start()
                
                # Still add audio during silence period (might resume)
                if self.is_recording_audio:
                    self.audio_buffer.extend(audio_data)

    def process_audio_queue(self):
        """Process audio from the queue using Whisper."""
        while self.is_listening:
            try:
                # Wait for audio data with timeout
                audio_data = self.audio_queue.get(timeout=1.0)
                
                print("🔄 Transcribing audio...")
                
                # Transcribe using Whisper
                result = self.model.transcribe(audio_data, fp16=False)
                text = result["text"].strip()
                
                if text:
                    print(f"📝 Transcribed: '{text}'")
                    
                    # Send text to callback if provided
                    if self.callback:
                        self.callback(text)
                else:
                    print("🔇 No speech detected in audio")
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing audio: {e}")

    def start_listening(self):
        """Start listening to the microphone."""
        if self.is_listening:
            print("Already listening!")
            return
        
        self.is_listening = True
        print("🎯 Starting enhanced voice controller...")
        print("🧠 Advanced noise filtering enabled")
        print("⌨️ Keyboard click filtering active")
        print("💬 Say something like: 'Move players to attack the other team'")
        print("🛑 Press Ctrl+C to stop")
        
        # Start audio processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
        self.processing_thread.start()
        
        # Start audio input stream with smaller block size for better responsiveness
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.05),  # 50ms blocks for better responsiveness
                dtype=np.float32
            ):
                print("🎤 Enhanced listening active... (microphone with noise filtering)")
                while self.is_listening:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping voice controller...")
        except Exception as e:
            print(f"❌ Error with audio stream: {e}")
        finally:
            self.stop_listening()

    def stop_listening(self):
        """Stop listening to the microphone."""
        self.is_listening = False
        
        # Clean up any active timers
        if self.silence_timer:
            self.silence_timer.cancel()
            self.silence_timer = None
        
        # End any active recording session
        if self.recording_session_active:
            self.end_recording_session()
        
        print("🔴 Enhanced voice controller stopped")

    def set_callback(self, callback: Callable[[str], None]):
        """Set the callback function for transcribed text."""
        self.callback = callback


def text_received_handler(text: str):
    """
    Default handler for received text.
    Replace this with your own logic to send text to main app.
    """
    print(f"🎮 Game Command: {text}")
    
    # Here you would typically:
    # - Send to main app via queue, socket, or direct function call
    # - Parse the command for game actions
    # - Handle specific voice commands
    
    # Example: Basic command detection
    text_lower = text.lower()
    if "attack" in text_lower:
        print("⚔️ Attack command detected!")
    elif "move" in text_lower:
        print("🏃 Move command detected!")
    elif "defend" in text_lower:
        print("🛡️ Defend command detected!")


def main():
    """Main function to run the enhanced voice controller."""
    # Create voice controller with callback
    voice_controller = VoiceController(
        callback=text_received_handler,
        model_size="base",  # Good balance of speed and accuracy
        min_audio_length=0.3  # Process shorter audio for responsiveness
    )
    
    try:
        # Start listening
        voice_controller.start_listening()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        voice_controller.stop_listening()


if __name__ == "__main__":
    main() 