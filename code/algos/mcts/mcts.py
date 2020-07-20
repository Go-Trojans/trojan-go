from algos.godomain import *
from algos.gohelper import *
from algos.encoders import TrojanGoPlane
from algos.nn import AGZ
import h5py
import numpy as np
from math import sqrt
from operator import attrgetter
import time
#import tensorflow as tf
import inspect
import os
import sys
import copy
import logging
logging = logging.getLogger(__name__)

from algos.utils import display_board, alphaNumnericMove_from_point, LOG_FORMAT, save_model_to_disk





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



def combine_experience(collectors):
    combined_model_input = np.concatenate([np.array(c.model_input) for c in collectors])
    combined_action_target = np.concatenate([np.array(c.action_target) for c in collectors])
    combined_value_target = np.concatenate([np.array(c.value_target) for c in collectors])

    return ExperienceBuffer(
        combined_model_input,
        combined_action_target,
        combined_value_target)

def load_experience(h5file):
    return ExperienceBuffer(model_input=np.array(h5file['experience']['model_input']),
                            action_target=np.array(h5file['experience']['action_target']),
                            value_target=np.array(h5file['experience']['value_target'])
                            )

class MCTSNode:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
    """

    def __init__(self, state, p=0,v=0,q=0,move=None,parent=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.state = state  # GameState object that the node represents
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0 # Win count
        self.visits = 0 # Visit count
        self.q = q  # Q value of the node
        self.p = p # Move probability given by the neural network
        self.v = v # Value given by the neural network


    def SelectChild(self,c=4):
        """
            Use PUCT to select a child.
            Returns the child node with the maximum PUCT score.
        """

        puctNodes = [child.PUCT(c) for child in self.childNodes]
        maxChild = np.argmax(puctNodes)
        return self.childNodes[maxChild]

    def expand(self, probs):
        """
            Use probabilites given by the neural network to set the child nodes.
            Checks that the move is legal before adding the child.
        """
        boardSize = self.state.board.board_width
        numMoves = len(probs)
        moves = [Move(Point(int(idx/boardSize),idx%boardSize)) for idx in range(numMoves-1)]
        moves.append(Move.pass_turn())
        moveProb = zip(moves,probs)
        legal_moves = self.state.legal_moves()
        for move,p in moveProb :
             if move in legal_moves :
                
                # if we don't pass; all the moves are legal
                if move.is_play:
                    next_board = copy.deepcopy(self.state.board)
                    next_board.place_stone(self.state.next_player, move.point)
                else:
                    next_board = self.state.board
                #self.state.moves should not matter as such so keeping it same as parent.
                childState = GameState(next_board, self.state.next_player.opp, self.state, move, self.state.moves)
                
                #childState = self.state.apply_move(move)
                child = MCTSNode(state=childState,move=move,parent=self,p=p)
                self.childNodes.append(child)


    def update(self, v):
        """
            Update this node - one additional visit and v additional wins.
            v is given from the neural network and must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += v
        self.q = self.wins / self.visits

    def PUCT(self,c=4):
        """
            Returns the PUCT score for the node.
        """
        N = 0
        for child in self.childNodes :
            N += child.visits
        puct = self.q + c*self.p*sqrt(N)/(1+self.visits)
        return puct


class MCTSPlayer :

    def __init__(self, player, model):
        self.player = player
        self.model = model

    def select_move(self,
                    rootnode, visited, encoder,
                    simulations,
                    epsilon = 0.25, dcoeff = [0.03], c=4,
                    stoch=True):
        nn = self.model
        #print(nn.summary())
        """
        Conduct a tree search for simulations iterations starting from gameState.
        Assumes 2 alternating players(player 1 starts), with game results in the range[-1, 1].
        Return the child MCTS nodes of the root node/gameState for exploratory play, or the move with the most visits for competitive play.
        """
        # rootnode = MCTSNode(state = gameState)
        # visited = set()
        for i in range(simulations):
            currNode = rootnode
            if stoch and i>0 :
                for child in currNode.childNodes:
                    # Dirichlet noise (do we need during acutal game play?)

                    """
                    Additional exploration is achieved by adding Dirichlet noise
                    to the prior probabilities in the root node s0, specifically 
                    P(s, a) = (1 − ε)pa + εηa, where η ∼ Dir(0.03) and ε = 0.25; 
                    this noise ensures that all moves may be tried, 
                    but the search may still overrule bad moves. 
                    """
                    # stoch will be set during self-play only & False during Actual Game-play !!!
                    if stoch:
                        child.p = (1-epsilon)*child.p + epsilon*np.random.dirichlet(alpha = dcoeff)
                    
            # Select
            while currNode in visited: # node is fully expanded and non-terminal
                currNode = currNode.SelectChild(c)

            # Expand
            if currNode not in visited:# if we can expand (i.e. state/node is non-terminal)
                visited.add(currNode)
                hero = currNode.state.next_player
                tensor = encoder.encode(currNode.state)
                tensor = np.expand_dims(tensor,axis=0)
                #print("tensor : ", tensor)
                try:
                    p,v = nn.predict(tensor)
                except OSError as err:
                    print("OS error: {0}".format(err))
                    logging.debug("OS error: {}".format(err))
                    raise
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    logging.debug("Unexpected error: {}".format(sys.exc_info()[0]))
                    raise
                    
                #print("I am able to use neural network model; nn")
                currNode.expand(p.flatten())# add children
            # Backpropagate
            while currNode:# backpropagate from the expanded node and work back to the root node
                currNode.update(v if hero == currNode.state.next_player else -v)# state is terminal. Update node with result from POV of node.playerJustMoved
                currNode = currNode.parentNode
        if stoch :
           return rootnode.childNodes
        else :
            # We should use "TAU"=0 to sample the possible best moves based on visit count 
            # TODO: Raveena and Puranjay to evaluate this.
            #return max(rootnode.childNodes, key=attrgetter('visits')).move
            return max(rootnode.childNodes, key=attrgetter('visits'))



class MCTSSelfPlay :

    def __init__(self, plane_size, board_size, model=None):
        
        self.board_size = board_size
        self.plane_size = plane_size
        self.model = model # not needed as such.
        self.encoder = TrojanGoPlane((board_size,board_size),plane_size)



    def save_moves(self, moves, winner):
            """
            Save input feature stack, search probabilities,and game winner to disk
            """

            for gameState,tensor, searchProb in moves:
                self.expBuff.model_input.append(tensor)
                self.expBuff.action_target.append(searchProb)
                if winner == 0 :
                    self.expBuff.value_target.append(0)
                elif winner == gameState.next_player.value :
                    self.expBuff.value_target.append(1)
                else :
                    self.expBuff.value_target.append(-1)



    def play(self, agent1, agent2,
             expFile, num_games=2500,
             simulations=400,
             c=4,vResign=0,tempMoves=4) :
        
        """
        :param num_game:
        :return:
         Play num_games of self play against two MCTS players and save move information to disk. """

        #print(inspect.currentframe().f_code.co_name, inspect.currentframe().f_back.f_code.co_name)
        
        model_input = []
        action_target = []
        value_target = []
        self.expBuff = ExperienceBuffer(model_input,action_target,value_target)
        self.expFile = expFile
        players = {
            Player.black: MCTSPlayer(Player.black, agent1),
            Player.white: MCTSPlayer(Player.white, agent2)
        }

        num_games_start = time.time()
        """ Play num_games games using self-play rl and MCTS """
        for i in range(num_games) :

            game_start = time.time()
            game = GameState.new_game(self.board_size)
            moves = []
            moveNum = 0
            visited = set()
            rootnode = None
            """ Code to play single game using self-play rl and MCTS """
            while not game.is_over() :
                move_start = time.time()
                #display_board(game.board)
                moveNum += 1
                """
                For the first 30 moves of each game, the temperature is set to τ = 1;
                this selects moves proportionally to their visit count in MCTS, 
                and ensures a diverse set of positions are encountered. 
                For the remainder of the game, an infinitesimal temperature is used, 
                τ→0 (that is, we deterministically select the move with maximum visit count, 
                to give the strongest possible play). 

                In short, τ near to zero means deterministic (exploitation)
                          τ near to 1 means stochasticity (more exploration)
                """
                if moveNum <= tempMoves :
                    tau = 1
                else :
                    #TODO: Raveena and Puranjay to evaluate it.
                    #tau = float('inf')
                    tau = 0.1
                if not rootnode:
                    rootnode  = MCTSNode(state = game)
                """ logic code to select a move using MCTS simulations """    
                mctsNodes =  players[game.next_player].select_move(rootnode,visited ,self.encoder,simulations ,c=c)
                tensor = self.encoder.encode(game)
                nn = players[game.next_player].model
                _, rootV = nn.predict(np.expand_dims(tensor, axis=0))
                childVals = []
                searchProb = np.zeros((self.board_size**2 + 1,),dtype='float')
                tempNodes = []
                for child in mctsNodes :
                    childTensor = self.encoder.encode(child.state)
                    childTensor = np.expand_dims(childTensor, axis=0)
                    _,childV = nn.predict(childTensor)
                    if child.move.is_play :
                        r,c = child.move.point
                        i = self.board_size*r + c
                    else :
                        i = self.board_size**2
                    """
                    At the end of the search, AlphaGo Zero selects a move "a" to play
                    in the root position s{0}, proportional to its exponentiated visit count,
                    π(a|s0)=N(s0,a) ** (1/τ) / ∑{b} N(s0,b)** (1/τ) ,
                    where τ is a temperature parameter that controls the level of exploration. 
                    """    
                    searchProb[i] = child.visits**(1/tau)
                    tempNodes.append(child.visits**(1/tau))
                    childVals.append(childV.item())
                probSum = sum(tempNodes)
                tempNodes = np.divide(tempNodes,probSum)
                searchProb = np.divide(searchProb,probSum)
                
                # Disabling self-resign feature as it looks like it is not working properly.
                """
                if rootV.item() < vResign and max(childVals) < vResign :
                    move = Move.resign()
                    print("It's a Resign !!!")
                else :
                    rootnode = np.random.choice(a=mctsNodes,p=tempNodes)
                    move = rootnode.move
                """

                rootnode = np.random.choice(a=mctsNodes,p=tempNodes)
                move = rootnode.move
                    
                moves.append((game,tensor,searchProb))
                if move.is_play:
                    #print(game.next_player, alphaNumnericMove_from_point(move.point))
                    pass
                if move.is_pass:
                    #print(game.next_player, "PASS")
                    pass
                if move.is_resign:
                    #print(game.next_player, "Resign")
                    pass
                game = game.apply_move(move)
                move_end = time.time()
                #print("Move time: ", move_end - move_start)


            winner = game.winner()
            #print("*"*60)
            #display_board(game.board)
            #print("Total moves : ", moveNum)
            #print("Winner is ", game.winner(), winner)
            game_end = time.time()
            #print("Time taken to play 1 game : ", game_end - game_start)
            self.save_moves(moves,winner)

        model_input = np.array(self.expBuff.model_input)
        action_target = np.array(self.expBuff.action_target)
        value_target = np.array(self.expBuff.value_target)

        num_games_end = time.time()
        print("Total time taken to play {} game(s) is {}".format(num_games, num_games_end - num_games_start))
        logging.debug("[PID : {}] Total time taken to play {} game(s) is {}".format(os.getpid(), num_games, num_games_end - num_games_start))

        """ Save the examples generated after playing 'num_games' self-play games to file """
        print("Going to save the examples here : ", self.expFile)
        with h5py.File(self.expFile, 'w') as exp_out:
            ExperienceBuffer(model_input, action_target, value_target).serialize(exp_out)



        """ filename is saved experienced file inside data dir
            source : https://www.tensorflow.org/tutorials/keras/save_and_load
            NOTE: tf.distribute.Strategy() is not used as of now.
        """
        
        """ output_file is in (.json, .h5) format """
    def train(self, exp_filename, output_file, learning_rate=0.01, batch_size=128, epochs=100):
        from keras.optimizers import SGD
        import tensorflow as tf
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        
        with h5py.File(exp_filename, 'r') as exp_input:
            experience_buffer = load_experience(exp_input)

        num_examples = experience_buffer.model_input.shape[0]
        print("num_examples : ", num_examples)

        model_input = experience_buffer.model_input
        action_target = experience_buffer.action_target
        value_target = experience_buffer.value_target

        # parallel training code
        strategy = tf.distribute.MirroredStrategy()

        # logging line for number of training devices available
        # print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            self.model.compile(
                        SGD(lr=learning_rate),
                        loss=['categorical_crossentropy', 'mse'])

        """ logic code for checkpointing.
             This is to understand how many epochs is best for training
             as loss may start increasing after a certain eochs.
             TARGET : WE NEED TO FIND THE BEST CHECKPOINTED MODELS. 
        """
        """
        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = "checkpoints/epochs_chkpts/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights every 5 epochs.
        # last checkpoint will be equivalent to the entire model saving after training gets over.
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                             filepath=checkpoint_path, 
                                             verbose=1, 
                                             save_weights_only=True,
                                             period=5)
        
        # Save the weights using the `checkpoint_path` format
        self.model.save_weights(checkpoint_path.format(epoch=0))
        """
        """
        @param monitor - the quantity monitored to determine where to stop
        @param mode - whether we want max or min of monitored quantity
        @param patience - how many extra epochs do we try after finding a new best before stopping
        @param restore_best_weights - Whether to restore the best weights found during training or stick with the current weights
        """
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            mode='min',
            patience=5,
            restore_best_weights=True
            )

        # Train the model with the callback
        self.model.fit(
                    model_input, [action_target, value_target],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping_callback])  # Pass callback to training
        """
        # This may generate warnings related to saving the state of the optimizer.
        # These warnings (and similar warnings throughout this notebook)
        # are in place to discourage outdated usage, and can be ignored.

        #latest = tf.train.latest_checkpoint(checkpoint_dir)
        
        # Loads the weights (syntx)
        # model.load_weights(checkpoint_path)
        """

        
        """
        # After training, save the entire model in 'checkpoints/iteration_Savedmodel/' dir.
        my_model = filename.split("/")[-1].split(".")[0]
        model_name = 'checkpoints/iteration_Savedmodel/' + my_model + '_model'
        model.save(model_name)
        """
        
        # Save the entire model to a HDF5 file.
        # The '.h5' extension indicates that the model should be saved to HDF5.
        #model_name = model_name + ".h5"
        #model.save(model_name)
        #self.model.save(output_file)
        
        #Reload a fresh Keras model from the saved model.
        #Recreate the exact same model, including its weights and the optimizer
        #new_model = tf.keras.models.load_model(model_name)


        """ Save the tained model in (.json, .h5) format """
        save_model_to_disk(self.model, output_file)
        print("trained model saved to disk : ", output_file)
"""
if __name__ == "__main__" :

    mctsSP = MCTSSelfPlay(7,5)
    input_shape = (7,5,5)
    nn = AGZ.init_random_model(input_shape)
    mctsSP.play(nn,'./data/experience_1.hdf5',num_games=2500,simulations=1600)
"""    
