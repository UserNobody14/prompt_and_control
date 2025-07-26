
from typing import Optional
import json
from openai import OpenAI
from gameboard import GameBoard, Player, Direction
from prompts.system import friendly_prompt, enemy_prompt

client = OpenAI()
model = "gpt-4.1-nano"

def get_llm_proposed_moves(gameboard: GameBoard, player: Player, user_transcript: Optional[str] = None) -> dict[int, Direction]:
    """ Takes in the gameboard and player and queries the llm for a move. Parsed out the response and returns it as a map of id to direction"""
    if player == Player.PLAYER:
        prompt = friendly_prompt
    else:
        prompt = enemy_prompt
    user_prompt = ""
    game_board = f"# Game State\n\n{gameboard.to_prompt(player)}"
    user_prompt += game_board
    if user_transcript:
        user_prompt += f"\n\n# Transcript\n\n{user_transcript}"
    user_prompt += "\n\n## Reminder\n\nPleae reply with just your move"

    
    response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            max_tokens=1000,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    response = response.choices[0].message.content
    try:
        parsed = json.loads(response)
        move = {}
        for color, direction in parsed.items():
            for id, piece in gameboard.pieces.items():
                if piece.owner != player:
                    continue
                if piece.color.value.lower() == color.lower():
                    llm_direction = Direction.from_str(direction)
                    if llm_direction:
                        move[id] = llm_direction
        return move
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from LLM: {e}")
        return {}
