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

capture_rules = """\
# Capture Rules

A unit is captured if it is adjacent to an enemy unit that is not adjacent to a larger group of it's own units.

Your units are represented with R,B,G,Y and enemy units are always represented with E. Empty spaces are X.

## Examples
```
X X X X
X R E X
X X X X
```
Neither R nor E is captured.

```
X X X X
B R E X
X X X X
```
R and B capture E.

```
X X E X
X R E X
X X X X
```
The two E's capture R.

```
X X E X
X G E X
X Y X X
```
No capture.
```
X X E X
X G E X
B Y X X
```
G, Y and B capture the two Es.

There is one additional rule: If your four units are ever part of a single connected component you lose. \
(Because there would be no way to capture a group of 4 with 4 units)
"""

common_rules = f"""\
You are part of a voice controlled real time strategy game called {game_name}. \
The game is played on a {board_size}x{board_size} grid. \
The game is a {player_count} player game. \
{{job}}
Each player gets starts with {len(unit_colors)} units. \
Each unit is represented by a color: {', '.join(unit_colors)}. \
{capture_rules}
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
All units from all players move simultaneously.
If there is a conflict where a unit tries to move through or into anotehr unit that unit will stay in place.
"""


friendly_prompt = common_rules.format(job=friendly_job, information=friendly_information)

enemy_prompt = common_rules.format(job=enemy_job, information=enemy_information)
