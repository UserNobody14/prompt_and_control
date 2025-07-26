import os
from typing import Optional
import json
from openai import OpenAI
from gameboard import GameBoard, Player, Direction
from prompts.system import friendly_prompt, enemy_prompt
import time

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
player_model = "gpt-4.1"
enemy_model = "gpt-4.1"


def get_llm_proposed_moves(
    gameboard: GameBoard,
    player: Player,
    user_messages: Optional[list[dict[str, str]]] = None,
) -> dict[int, Direction]:
    """Takes in the gameboard and player and queries the llm for a move. Parsed out the response and returns it as a map of id to direction"""
    model = player_model if player == Player.PLAYER else enemy_model
    if player == Player.PLAYER:
        prompt = friendly_prompt
    else:
        prompt = enemy_prompt
    game_board = f"# Game State\n\n{gameboard.to_prompt(player)}"
    if not user_messages:
        user_messages = [
            {"role": "user", "content": f"{game_board}\n\nPlease make a move"}
        ]
    print(user_messages)

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        max_tokens=1000,
        messages=[{"role": "system", "content": prompt}] + user_messages,
    )
    response = response.choices[0].message.content
    print("LLM response:", response, "\nIn: ", time.time() - start, "(s)")
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
