import threading
import time
from typing import Callable, Optional
import queue
import sys


class VoiceController:
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """
        Initialize a raw audio voice controller that captures complete speech segments.
        
        Args:
            callback: Function to call when transcript is ready
        """
        self.callback = callback
        self.is_listening = False
        
        # Audio detection settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.voice_threshold = 0.005  # Volume threshold for voice detection
        self.silence_duration = 1.2   # Seconds of silence before ending capture
        self.min_speech_duration = 0.2  # Minimum speech length to process
        self.max_speech_duration = 12.0  # Maximum 12 seconds of capture
        self.pre_speech_buffer = 0.8  # Buffer before speech starts (to catch beginning)
        
        # Audio buffers
        self.audio_buffer = []
        self.pre_buffer = []
        self.is_recording = False
        self.last_voice_time = 0
        self.recording_start_time = 0
        
        # Try to set up audio and speech recognition
        try:
            import sounddevice as sd
            import numpy as np
            self.has_audio = True
        except ImportError:
            self.has_audio = False
            print("❌ Audio libraries not available")
        
        # Try to set up speech recognition
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.has_speech_recognition = True
        except ImportError:
            self.has_speech_recognition = False
            print("⚠️ Speech recognition not available")

    def transcribe_audio(self, audio_array, sample_rate):
        """Transcribe audio array to text."""
        if not self.has_speech_recognition:
            return "[Speech recognition not available]"
        
        try:
            import speech_recognition as sr
            import numpy as np
            import io
            import wave
            
            # Convert numpy array to AudioData
            # Normalize to 16-bit PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_buffer.seek(0)
            
            # Create AudioData object
            audio_data = sr.AudioData(
                wav_buffer.getvalue(),
                sample_rate,
                2  # 16-bit samples
            )
            
            # Try to recognize speech
            try:
                # Try Google Speech Recognition
                text = self.recognizer.recognize_google(audio_data)
                return text.strip()
            except sr.UnknownValueError:
                return "[Could not understand audio]"
            except sr.RequestError:
                return "[Speech service unavailable]"
            except Exception as e:
                return f"[Recognition error]"
                
        except Exception as e:
            return f"[Transcription error]"

    def audio_callback(self, indata, frames, timestamp, status):
        """Process incoming audio data in real-time."""
        try:
            import numpy as np
            
            # Convert to mono if needed
            if len(indata.shape) > 1:
                audio_data = indata[:, 0]
            else:
                audio_data = indata.flatten()
            
            # Calculate audio level (RMS)
            audio_level = np.sqrt(np.mean(audio_data**2))
            
            # Always add to pre-buffer (circular buffer)
            self.pre_buffer.extend(audio_data)
            max_pre_buffer_size = int(self.sample_rate * self.pre_speech_buffer)
            if len(self.pre_buffer) > max_pre_buffer_size:
                excess = len(self.pre_buffer) - max_pre_buffer_size
                self.pre_buffer = self.pre_buffer[excess:]
            
            current_time = time.time()
            
            # Check if voice is detected
            if audio_level > self.voice_threshold:
                self.last_voice_time = current_time
                
                # Start recording if not already
                if not self.is_recording:
                    self.is_recording = True
                    self.recording_start_time = current_time
                    # Include pre-buffer to catch beginning of speech
                    self.audio_buffer = list(self.pre_buffer)
                    print(f"🎤 Recording...")
                
                # Add current audio to buffer
                self.audio_buffer.extend(audio_data)
                
                # Check if we've hit the maximum recording time
                recording_duration = current_time - self.recording_start_time
                if recording_duration >= self.max_speech_duration:
                    print(f"⏰ Maximum recording time reached ({self.max_speech_duration}s)")
                    self.end_recording()
            
            # Check if recording and silence detected
            elif self.is_recording:
                silence_time = current_time - self.last_voice_time
                
                # Continue recording during short silences
                if silence_time < self.silence_duration:
                    self.audio_buffer.extend(audio_data)
                else:
                    # End recording due to silence
                    self.end_recording()
            
        except Exception as e:
            print(f"❌ Audio error: {e}")

    def end_recording(self):
        """End current recording and process the captured audio."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if len(self.audio_buffer) == 0:
            return
        
        # Check if recording is long enough
        duration = len(self.audio_buffer) / self.sample_rate
        
        if duration < self.min_speech_duration:
            print(f"⏭️ Too short ({duration:.1f}s), skipping")
            self.audio_buffer = []
            return
        
        print(f"🔊 Captured {duration:.1f}s of speech")
        
        # Transcribe the audio
        try:
            import numpy as np
            audio_array = np.array(self.audio_buffer, dtype=np.float32)
            
            print("🔄 Transcribing...")
            transcript = self.transcribe_audio(audio_array, self.sample_rate)
            
            # Check if transcription was successful
            if transcript.startswith('[') and transcript.endswith(']'):
                # Error or could not understand
                print("=" * 50)
                print("📝 Not understood, nothing sent")
                print("=" * 50)
                # Don't send anything to callback/LLM
            else:
                # Successful transcription
                print("=" * 50)
                print(f"📝 TRANSCRIPT: '{transcript}'")
                print("=" * 50)
                
                # Send transcript to callback for LLM processing
                if self.callback:
                    self.callback(transcript)
                
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
        
        # Clear buffer
        self.audio_buffer = []

    def start_listening(self):
        """Start raw audio voice detection."""
        if self.is_listening:
            print("Already listening!")
            return
        
        if not self.has_audio:
            print("❌ Audio system not available")
            return
        
        self.is_listening = True
        
        try:
            import sounddevice as sd
            
            # Optimized settings to prevent buffer overflow
            # Start audio input stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.05),  # 50ms blocks for better responsiveness
                dtype='float32',
                latency='low'  # Use low latency mode
            ):
                print("🎤 Listening...")
                
                while self.is_listening:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
        finally:
            self.stop_listening()

    def stop_listening(self):
        """Stop voice detection."""
        self.is_listening = False
        
        # Process any remaining recording
        if self.is_recording:
            self.end_recording()
        
        print("🔴 Stopped")

    def set_callback(self, callback: Callable[[str], None]):
        """Set the callback function for transcripts."""
        self.callback = callback


def transcript_handler(transcript: str):
    """
    Simple handler for transcribed text.
    """
    # This will only receive valid transcripts now
    # (error messages are filtered out before calling this)
    pass


def main():
    """Main function for voice controller."""
    # Create voice controller
    voice_controller = VoiceController(callback=transcript_handler)
    
    try:
        # Start voice detection
        voice_controller.start_listening()
    except KeyboardInterrupt:
        print("\n👋 Done!")
    finally:
        voice_controller.stop_listening()


if __name__ == "__main__":
    main()
