import numpy as np
from algos.agent import randombot
from algos.encoders.trojangoPlane import TrojanGoPlane
from algos import gohelper
from algos import godomain
from algos.utils import display_board, alphaNumnericMove_from_point
import time
import math
import h5py


class ExperienceBuffer:
    def __init__(self, model_input, action_target, value_target):
        self.model_input = model_input
        self.action_target = action_target
        self.value_target = value_target
        
    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('model_input', data=self.model_input)
        h5file['experience'].create_dataset('action_target', data=self.action_target)
        h5file['experience'].create_dataset('value_target', data=self.value_target)

    
    def load_experience(self, h5file):
        return ExperienceBuffer(model_input=np.array(h5file['experience']['model_input']),
                               action_target=np.array(h5file['experience']['action_target']),
                               value_target=np.array(h5file['experience']['value_target'])
                               )

    def display_experience_buffer(self):
        print("Model Input : ")
        print(self.model_input)
        

def save_examples(encoder, exp_buff, game, bot_move, val):
    board_tensor = encoder.encode(game)

    exp_buff.model_input.append(board_tensor)

    """ create a flat 26 numpy array"""
    search_prob = np.zeros(26)
    # Convert the bot_move to particular index
    index = 0
    if bot_move.is_pass:
        index = 25
    else:    
        row = bot_move.point.row
        col = bot_move.point.col
        index = int(game.board.board_width * row + col)
        
    search_prob[index] = 1
    exp_buff.action_target.append(search_prob)
    
    
    exp_buff.value_target.append(val)
    return None


    
def main(win_rec, exp_buff):
    board_size = 5
    num_planes = 7
    
    encoder = TrojanGoPlane((board_size, board_size), num_planes)
    
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
        #display_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        #print_move(game.next_player, bot_move)
        """
        if bot_move.is_pass:
            print(game.next_player, "PASS")
        else:
            print(game.next_player, alphaNumnericMove_from_point(bot_move.point))
        """    
        game = game.apply_move(bot_move)

        """store the input_tensor, action, value = 1 as of now"""
        save_examples(encoder, exp_buff, game, bot_move, 1)
        
        
        

    finish = time.time()    
    #game.board.display_board()
    #display_board(game.board)
    #print("Total moves : ", moves)
    #print("Winner is ", game.winner())
    win_rec[game.winner()] = win_rec[game.winner()] + 1
    #print("Time taken to play a game is {} secs".format(finish - start))


if __name__ == '__main__':
    model_input = []
    #model_input = np.array(model_input)
    action_target = []
    #action_target = np.array(action_target)
    value_target = []
    #value_target = np.array(value_target)
    
    #with h5py.File('experience_1.hdf5', 'w') as exp_outf:
    exp_buff = ExperienceBuffer(model_input, action_target, value_target)
    
    total_games = 2500
    start = time.time()
    win_rec = [0, 0 ,0] # draw, Black wins, White wins
    # Play total_games games
    for i in range(total_games):
        main(win_rec, exp_buff)


    #Now the games are over, convert the exp_buff members to a np.array and save to file.
    model_input = np.array(exp_buff.model_input)
    action_target = np.array(exp_buff.action_target)
    value_target = np.array(exp_buff.value_target)

    with h5py.File('experience_2.hdf5', 'w') as exp_out:
        ExperienceBuffer(model_input, action_target, value_target).serialize(exp_out)
    

    with h5py.File('experience_2.hdf5', 'r') as exp_input:
        experience_buffer = ExperienceBuffer(model_input, action_target, value_target).load_experience(exp_input)

    print("Input Model ...")
    print(experience_buffer.model_input.shape)
    print("Action Target ...")
    print(experience_buffer.action_target.shape)


    
    finish = time.time()
    print("Time taken to play {} games is {} secs".format(total_games, math.floor(finish - start)))
    print("Draws: {} Black wins: {} White wins: {}  ".format(win_rec[0],win_rec[1], win_rec[2]))
    
