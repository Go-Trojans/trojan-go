from algos import godomain,gohelper,utils
from algos.encoders import TrojanGoPlane
import io, os, sys, types
from algos.nn import AGZ
import h5py
import numpy as np
from math import sqrt
import copy
import random
from keras.optimizers import SGD

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
        max_score = -float('inf')
        best_child = None
        for child in self.childNodes :
            puct = child.PUCT(c)
            if puct > max_score :
                max_score = puct
                best_child = child
        return best_child

    def expand(self, probs):
        """
            Use probabilites given by the neural network to set the child nodes.
            Checks that the move is legal before adding the child.
        """
        boardSize = self.state.board.board_width
        numMoves = len(probs)
        moves = [godomain.Move(gohelper.Point(int(idx/boardSize),idx%boardSize)) for idx in range(numMoves)]
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

    def PUCT(self,c):
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

    def select_move(self,gameState,encoder,simulations,nn,epsilon = 0.25,dcoeff = [0.03],c=4):
        """
        Conduct a tree search for itermax iterations starting from gameState.
        Return the best move from the gameState. Assumes 2 alternating players(player 1 starts), with game results in the range[-1, 1].
        """
        rootnode = MCTSNode(state = gameState)
        visited = set()
        for i in range(simulations):
            currNode = rootnode
            state = copy.deepcopy(gameState)
            if i>0 :
                for child in currNode.childNodes:
                    # Dirichlet noise
                    child.p = (1-epsilon)*child.p + epsilon*np.random.dirichlet(alpha = dcoeff)
            # Select
            while currNode in visited: # node is fully expanded and non-terminal
                currNode = currNode.SelectChild(c)
                #currNode.state = state.apply_move(currNode.move)

            # Expand
            if currNode not in visited:# if we can expand (i.e. state/node is non-terminal)
                visited.add(currNode)
                hero = currNode.state.next_player
                tensor = encoder.encode(currNode.state)
                tensor = np.expand_dims(tensor,axis=0)
                p,v = nn.predict(tensor)
                if i==0 :
                    probs = np.array([])
                    for prob in p.flatten() :
                        probs.append((1-epsilon)*prob + epsilon*np.random.dirichlet(alpha = dcoeff))
                else :
                    probs = p.flatten()
                currNode.expand(probs)# add children
            # Backpropagate
            while currNode:# backpropagate from the expanded node and work back to the root node
                currNode.update(v if hero == currNode.state.next_player else -v)# state is terminal. Update node with result from POV of node.playerJustMoved
                currNode = currNode.parentNode
        return rootnode.childNodes



class MCTSSelfPlay :

    def __init__(self,plane_size,board_size):

        self.board_size = board_size
        self.plane_size = plane_size
        self.encoder = TrojanGoPlane((board_size,board_size),plane_size)



    def save_moves(self, moves, winner):
            """
            Save input feature stack, search probabilities,and game winner to disk
            """

            for gameState, searchProb in moves:
                encodedState = self.encoder.encode(gameState)
                self.expBuff.model_input.append(encodedState)
                self.expBuff.action_target.append(searchProb)
                if winner == 0 :
                    self.expBuff.value_target.append(0)
                elif winner == gameState.next_player :
                    self.expBuff.value_target.append(1)
                else :
                    self.expBuff.value_target.append(-1)

            model_input = np.array(self.expBuff.model_input)
            action_target = np.array(self.expBuff.action_target)
            value_target = np.array(self.expBuff.value_target)

            with h5py.File(self.expFile, 'w') as exp_out:
                ExperienceBuffer(model_input, action_target, value_target).serialize(exp_out)


    def play(self,network,expFile,num_games=2500,c=4,tempMoves=30) :
        """
        :param num_game:
        :return:
         Play num_games of self play against two MCTS players and save move information to disk. """

        input_shape = (self.plane_size,self.board_size,self.board_size)

        self.expBuff = ExperienceBuffer([],[],[])
        self.expFile = expFile
        players = {
            gohelper.Player.black: MCTSPlayer(gohelper.Player.black),
            gohelper.Player.white: MCTSPlayer(gohelper.Player.white)
        }

        for i in range(num_games) :


            game = godomain.GameState.new_game(self.board_size)
            moves = []
            moveNum = 0
            while not game.is_over() :
                moveNum += 1
                if moveNum <= tempMoves :
                    tau = 1
                else :
                    tau = float('inf')
                mctsNodes =  players[game.next_player].select_move(game,self.encoder,1600,nn,c=c)
                if not mctsNodes :
                    move = godomain.Move(is_pass=True)
                else :
                    tempNodes = [node.visits**(1/tau) for node in mctsNodes]
                    tempNodeSum = sum(tempNodes)
                    searchProb = [n/tempNodeSum for n in tempNodes]
                    move = np.random.choice(a=mctsNodes,p=searchProb).move

                game = game.apply_move(move)

                moves.append((game,searchProb))

            winner = game.winner()
            self.save_moves(moves,winner)



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

if __name__ == "__main__" :

    mctsSP = MCTSSelfPlay(7,5)
    input_shape = (7,5,5)
    nn = AGZ.init_random_model(input_shape)
    mctsSP.play(nn,'mctsExperience.hdf5')