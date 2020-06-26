from algos.agent import randombot
from algos.agent import humanbot
from algos import gohelper
from algos import godomain
from algos.utils import display_board, alphaNumnericMove_from_point
from algos.encoders.trojangoPlane import TrojanGoPlane
import time
import math

def main(bot1, bot2, win_rec, encoder):
    board_size = 5
    game = godomain.GameState.new_game(board_size)
    bots = {
        gohelper.Player.black: bot1,
        gohelper.Player.white: bot2,
    }
   
    
    moves = 0
    start = time.time()
    while not game.is_over():
        #game.board.display_board()
        display_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        if not game.board.is_on_grid(bot_move.point):
            print("Invalid move, Try Again ...")
            continue
            
        moves = moves + 1
            
        print(game.next_player, alphaNumnericMove_from_point(bot_move.point))
        game = game.apply_move(bot_move)
        """uncomment the print the encoder"""
        board_tensor = encoder.encode(game)
        #print("Board tensor for this game state ...")
        #print(board_tensor)
        

    finish = time.time()    

    #game.board.display_board()
    display_board(game.board)
    print("Total moves : ", moves)
    print("Winner is ", game.winner())
    win_rec[game.winner()] = win_rec[game.winner()] + 1
    print("Time taken to play a game is {} secs".format(finish - start))


if __name__ == '__main__':
    # bot1 is Black and bot2 is white ()see main function for this)

    total_games = 1
    start = time.time()
    win_rec = [0, 0 ,0] # draw, Black wins, White wins

    """encoder code piece"""
    board_size = (5, 5)
    num_planes = 7
    encoder = TrojanGoPlane(board_size, num_planes)
    
    
    # Play total_games between HumanBot and RandomBot
    bot1 = humanbot.HumanBot()
    bot2 = randombot.RandomBot()
    print("Human playing (move eg. C2) as Black and RandomBot as White !!!")    
    for i in range(total_games):
        main(bot1, bot2, win_rec, encoder)




    finish = time.time()
    print("Time taken to play {} games is {} secs".format(total_games, math.floor(finish - start)))
    print("Draws: {} Black wins: {} White wins: {}  ".format(win_rec[0],win_rec[1], win_rec[2]))
    
