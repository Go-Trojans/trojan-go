from algos.godomain import *
from algos.gohelper import *
from algos.encoders import TrojanGoPlane
from algos.nn import AGZ
import h5py
import numpy as np
from math import sqrt
from operator import attrgetter
import time

from algos.utils import display_board, alphaNumnericMove_from_point, LOG_FORMAT


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
                childState = self.state.apply_move(move)
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

    def __init__(self,player):

        self.player = player

    def select_move(self,rootnode,visited,encoder,simulations,nn,epsilon = 0.25,dcoeff = [0.03],c=4,stoch=True):
        """
        Conduct a tree search for simulations iterations starting from gameState.
        Assumes 2 alternating players(player 1 starts), with game results in the range[-1, 1].
        Return the child MCTS nodes of the root node/gameState for exploratory play, or the move with the most visits for competitive play.
        """
        # rootnode = MCTSNode(state = gameState)
        # visited = set()
        for i in range(simulations):
            currNode = rootnode
            if i>0 :
                for child in currNode.childNodes:
                    # Dirichlet noise
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
                p,v = nn.predict(tensor)
                currNode.expand(p.flatten())# add children
            # Backpropagate
            while currNode:# backpropagate from the expanded node and work back to the root node
                currNode.update(v if hero == currNode.state.next_player else -v)# state is terminal. Update node with result from POV of node.playerJustMoved
                currNode = currNode.parentNode
        if stoch :
           return rootnode.childNodes
        else :
            return max(rootnode.childNodes, key=attrgetter('visits')).move



class MCTSSelfPlay :

    def __init__(self,plane_size,board_size):

        self.board_size = board_size
        self.plane_size = plane_size
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



    def play(self,nn,expFile,num_games=2500,simulations=1600,c=4,vResign=0,tempMoves=10) :
        """
        :param num_game:
        :return:
         Play num_games of self play against two MCTS players and save move information to disk. """

        model_input = []
        action_target = []
        value_target = []
        self.expBuff = ExperienceBuffer(model_input,action_target,value_target)
        self.expFile = expFile
        players = {
            Player.black: MCTSPlayer(Player.black),
            Player.white: MCTSPlayer(Player.white)
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
                display_board(game.board)
                moveNum += 1
                if moveNum <= tempMoves :
                    tau = 1
                else :
                    tau = float('inf')
                if not rootnode:
                    rootnode  = MCTSNode(state = game)
                """ logic code to select a move using MCTS simulations """    
                mctsNodes =  players[game.next_player].select_move(rootnode,visited ,self.encoder,simulations,nn,c=c)
                tensor = self.encoder.encode(game)
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
                    searchProb[i] = child.visits**(1/tau)
                    tempNodes.append(child.visits**(1/tau))
                    if childV.item() < 0 :
                        x = 2
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
                    print(game.next_player, alphaNumnericMove_from_point(move.point))
                if move.is_pass:
                    print(game.next_player, "PASS")
                if move.is_resign:
                    print(game.next_player, "Resign")
                game = game.apply_move(move)
                move_end = time.time()
                print("Move time: ", move_end - move_start)


            winner = game.winner()
            print("*"*60)
            display_board(game.board)
            print("Total moves : ", moveNum)
            print("Winner is ", game.winner(), winner)
            game_end = time.time()
            print("Time taken to play 1 game : ", game_end - game_start)
            self.save_moves(moves,winner)

        model_input = np.array(self.expBuff.model_input)
        action_target = np.array(self.expBuff.action_target)
        value_target = np.array(self.expBuff.value_target)

        num_games_end = time.time()
        print("Total time taken to play {} game(s) is {}".format(num_games, num_games_end - num_games_start))

        """ Save the examples generated after playing 'num_games' self-play games to file """
        print("Going to save the examples here : ", self.expFile)
        with h5py.File(self.expFile, 'w') as exp_out:
            ExperienceBuffer(model_input, action_target, value_target).serialize(exp_out)


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
"""
if __name__ == "__main__" :

    mctsSP = MCTSSelfPlay(7,5)
    input_shape = (7,5,5)
    nn = AGZ.init_random_model(input_shape)
    mctsSP.play(nn,'./data/experience_1.hdf5',num_games=2500,simulations=1600)
"""    
