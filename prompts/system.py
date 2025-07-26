game_name = "Prommpt and Control"
board_size = 10
player_count = 2
unit_colors = ["red", "blue", "green", "yellow"]
directions = ["up", "down", "left", "right", "stay"]

friendly_job = """\
Your job is is to listen to the player and output movements to the units.
"""
enemy_job = """\
Your goal is to win the game by defeating the other players.
"""

friendly_information = """\
You will be shown the current state of the game, recent game history and recent voice transcription from the player.
"""

enemy_information = """\
You will be shown the current state of the game and recent game history.
"""

common_rules = f"""\
You are part of a voice controlled real time strategy game called {game_name}. \
The game is played on a {board_size}x{board_size} grid. \
The game is a {player_count} player game. \
{{job}}
Each player gets starts with {len(unit_colors)} units. \
Each unit is represented by a color: {', '.join(unit_colors)}. \
{{information}}

Based on that output json in the following format to determine a move:
{{{{
    color: direction
    ...
}}}}
Where color can be one of the following: {', '.join(unit_colors)} and direction can be one of the following: {', '.join(directions)}. \
For example:
```
{{{{
"red": "up",
"blue": "down",
"green": "left",
"yellow": "right"
}}}}
"""


friendly_prompt = common_rules.format(job=friendly_job, information=friendly_information)

enemy_prompt = common_rules.format(job=enemy_job, information=enemy_information)
