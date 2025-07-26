
from llm import get_llm_proposed_moves
from gameboard import GameBoard, Player, Direction
gameboard = GameBoard()

all_up = "Please move all units up"
print(get_llm_proposed_moves(gameboard, Player.PLAYER, all_up))
mix = "Please move red up, blue down, green left, yellow right"
print(get_llm_proposed_moves(gameboard, Player.PLAYER, mix))
