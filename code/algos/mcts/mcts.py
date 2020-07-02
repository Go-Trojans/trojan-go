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
        self.wins = 0
        self.visits = 0
        self.playerJustMoved = state.playerJustMoved  # the only part of the state that the Node needs later
        self.q = q  # q value of the node
        self.p = p
        self.v = v


    def SelectChild(self):
        """ Use Q+U to select action
        """
        # returns s which is a sorted list of child nodes according to win formula
        maxChild = np.argmax([child.UCT() for child in self.childNodes])
        return self.childNodes[maxChild]

    def expand(self, pred,dir = False):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        boardSize = self.state.board.board_width
        index = pred.len()
        moves = [godomain.Move(gohelper.Point(int(i/boardSize),i%boardSize)) for i in range(index)]
        moveProb = zip(moves,pred)
        for move,p in moveProb :
            child = MCTSNode(self.state.apply_move(move),move=move,parent=self,p=p)
            self.childNodes.append(child)


    def update(self, result):
        """Update this
        node - one
        additional
        visit and result
        additional
        wins.result
        must
        be
        from the viewpoint
        of
        playerJustmoved.
        """
        self.visits += 1
        self.wins += self.result
        self.q = self.wins / self.visits

    def UCT(self,c_puct):
        """
        Returns
        UCT
        Score
        for node
        """
        N = 0
        for child in self.childNodes() :
            N += child.visits
        uct = self.q + c_puct*self.p*sqrt(N)/(1+self.visits)
        return uct


class MCTSPlayer :

    def __init__(self,player):

        self.player = player

    def select_move(gameState,encoder,simulations,nn,verbose = False,epsilon = 0.25,dcoeff = 0.03):
        """
        Conduct a tree search for itermax iterations starting from gameState.
        Return the best move from the gameState. Assumes 2 alternating players(player 1 starts), with game results in the range[-1, 1].
        """
        rootnode = MCTSNode(state = gameState)
        visited = set()
        for i in range(simulations):
            node = rootnode
            state = copy.deepcopy(gameState)
            for child in node.childNodes:
                # Dirichlet noise
                child.p = (1-epsilon)*child.p + epsilon*np.random.dirichlet(alpha = dcoeff)
            # Select
            while node in visited: # node is fully expanded and non-terminal
                node = node.SelectChild()
                node.state = state.apply_move(node.move)
            # Expand

            if node not in visited:# if we can expand (i.e. state/node is non-terminal)
                visited.add(node)
                hero = node.playerJustmoved
                tensor = encoder.encode(node.state)
                p,v = nn.predict(tensor)
                node.expand(p)# add child and descend tree
            # # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
            # while state.legal_moves():# while state is non-terminal
            #     state.apply_move(random.choice(state.legal_moves()))
            # Backpropagate

            while node:# backpropagate from the expanded node and work back to the root node
                node.update(v if hero == node.playerJustMoved else -v)# state is terminal. Update node with result from POV of node.playerJustMoved
                node = node.parentNode
        return sorted(rootnode.childNodes,key=lambda c: c.visits)# return the move that was most visited



class MCTSSelfPlay :

    def __init__(self,board_size,plane_size,expFile):

        self.board_size = board_size
        self.plane_size = plane_size
        self.encoder = TrojanGoPlane(board_size,plane_size)
        self.expFile = expFile
        self.expBuff = ExperienceBuffer([], [], [])


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


    def play(self,num_games=2500) :
        """
        :param num_game:
        :return:
         Play num_games of self play against two MCTS players and save move information to disk. """

        input_shape = (self.plane_size,self.board_size,self.board_size)
        nn = AGZ.init_random_model()
        self.expBuff = ExperienceBuffer([],[],[])
        players = {
            gohelper.Player.black: MCTSPlayer(gohelper.Player.black),
            gohelper.Player.white: MCTSPlayer(gohelper.Player.white)
        }

        winner = 0
        for i in num_games :


            game = godomain.GameState.new_game(self.board_size)
            moves = []
            while not game.is_over() :

                move,searchProb = players[game.next_player].select_move(game,self.encoder,nn,1600)

                game = game.apply_move(move)

                moves.append((game,searchProb))

            winner = game.winner()
            self.saveMoves(moves,winner)



class ExperienceBuffer:
    def __init__(self, model_input, action_target, value_target):
        self.model_input = model_input
        self.action_target = action_target
        self.value_target = self.value_target

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