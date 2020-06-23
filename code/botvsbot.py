from algos.agent import randombot
from algos import gohelper
from algos import godomain
from algos.utils import display_board, alphaNumnericMove_from_point
import time
import math

def main(win_rec):
    board_size = 5
    game = godomain.GameState.new_game(board_size)
    bots = {
        gohelper.Player.black: randombot.RandomBot(),
        gohelper.Player.white: randombot.RandomBot(),
    }
    moves = 0
    start = time.time()
    while not game.is_over():
        moves = moves + 1
        #game.board.display_board()
        display_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        #print_move(game.next_player, bot_move)
        if bot_move.is_pass:
            print(game.next_player, "PASS")
        else:
            print(game.next_player, alphaNumnericMove_from_point(bot_move.point))
        game = game.apply_move(bot_move)

    finish = time.time()    
    #game.board.display_board()
    display_board(game.board)
    print("Total moves : ", moves)
    print("Winner is ", game.winner())
    win_rec[game.winner()] = win_rec[game.winner()] + 1
    print("Time taken to play a game is {} secs".format(finish - start))


if __name__ == '__main__':
    total_games = 1
    start = time.time()
    win_rec = [0, 0 ,0] # draw, Black wins, White wins
    # Play total_games games
    for i in range(total_games):
        main(win_rec)

    finish = time.time()
    print("Time taken to play {} games is {} secs".format(total_games, math.floor(finish - start)))
    print("Draws: {} Black wins: {} White wins: {}  ".format(win_rec[0],win_rec[1], win_rec[2]))
    
