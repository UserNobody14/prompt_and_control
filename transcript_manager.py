from typing import Optional


class TranscriptManager:
    conversation: list[dict[str, str]] = []
    transcript: list[str] = []

    def __init__(self):
        self.conversation = []

    def add_message(self, prompt: str, new_transcript: list[str]) -> Optional[str]:
        """Adds a new message to the conversation. Returns the concate"""
        if len(new_transcript) == len(self.transcript):
            self.conversation += [
                {
                    "role": "user",
                    "content": prompt
                    + "\n\nNo new instructions from user. Keep executing their plan.",
                }
            ]
        else:
            new = new_transcript[len(self.transcript) :]
            self.conversation += [
                {"role": "user", "content": prompt + "\n\n#New instructions\n" + new}
            ]
            self.transcript = new_transcript
            return new
