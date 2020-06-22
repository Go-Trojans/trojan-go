from algos.agent import randombot
from algos import gohelper
from algos import godomain
from algos.utils import print_board, print_move
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
        #time.sleep(0.3)  # <1>
        #print(chr(27) + "[2J")  # <2>
        #print_board(game.board)
        ##game.board.display_board()
        bot_move = bots[game.next_player].select_move(game)
        #print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)

    finish = time.time()    
    #print(chr(27) + "[2J")  # <2>
    #print_board(game.board)
    ##game.board.display_board()
    #print("Total moves : ", moves)
    #print("Winner is ", game.winner())
    win_rec[game.winner()] = win_rec[game.winner()] + 1
    #print("Time taken to play a game is {} secs".format(finish - start))


if __name__ == '__main__':
    total_games = 10000
    start = time.time()
    win_rec = [0, 0 ,0] # draw, Black wins, White wins
    # Play total_games games
    for i in range(total_games):
        main(win_rec)

    finish = time.time()
    print("Time taken to play {} games is {} secs".format(total_games, math.floor(finish - start)))
    print("Draws: {} Black wins: {} White wins: {}  ".format(win_rec[0],win_rec[1], win_rec[2]))
    
