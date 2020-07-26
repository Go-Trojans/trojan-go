from algos.godomain import *
from algos.gohelper import *
from algos.encoders import TrojanGoPlane
from algos.nn import AGZ
import h5py
import numpy as np
from math import sqrt
from operator import attrgetter
import concurrent
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn',force=True)
import copy
import time
from functools import partial
import os


""" return True if tensorflow version is equal or higher than 2.0, else False """
def tf_version_comp(tf_v):
    tf_v = tf_v.split(".")
    if int(tf_v[0]) == 2 and int(tf_v[1]) >=0:
        return True
    return False
    

def set_gpu_memory_target(frac):
    """Configure Tensorflow to use a fraction of available GPU memory.

    Use this for evaluating models in parallel. By default, Tensorflow
    will try to map all available GPU memory in advance. You can
    configure to use just a fraction so that multiple processes can run
    in parallel. For example, if you want to use 2 works, set the
    memory fraction to 0.5.

    If you are using Python multiprocessing, you must call this function
    from the *worker* process (not from the parent).

    This function does nothing if Keras is using a backend other than
    Tensorflow.
    """
    import keras
    if keras.backend.backend() != 'tensorflow':
        print("Return without doing anything")
        return
    # Do the import here, not at the top, in case Tensorflow is not
    # installed at all.
    import tensorflow as tf
    #from keras.backend.tensorflow_backend import set_session
    if tf_version_comp(tf.__version__):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = frac
        config.gpu_options.allow_growth = True
        #set_session(tf.compat.v1.Session(config=config))
        session = tf.compat.v1.Session(config=config)
        #tf.compat.v1.keras.backend.set_session(session)

    else:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = frac
        config.gpu_options.allow_growth = True
        #set_session(tf.Session(config=config))
        session = tf.Session(config=config)




def run_simulation(simulations,encoder,rootnode,visited):
    epsilon = 0.25
    dcoeff = [0.03]
    c=4
    vLoss=1
    stoch=True

    import tensorflow as tf
    import keras
    set_gpu_memory_target(0.5)
    print("PID: ", os.getpid())
    
    
    #visited = set()
    input_shape = (7, 5, 5)
    nn = AGZ.init_random_model(input_shape)

    for i in range(simulations) :
        currNode = rootnode
        # if i>0 :
        for child in currNode.childNodes:
            # Dirichlet noise
            child.p = (1 - epsilon) * child.p + epsilon * np.random.dirichlet(alpha=dcoeff)
        # Select
        while currNode in visited:  # node is fully expanded and non-terminal
            currNode = currNode.SelectChild(c)
            #currNode.visits += vLoss
            #currNode.wins -= vLoss

        # Expand
        if currNode not in visited:  # if we can expand (i.e. state/node is non-terminal)
            #currNode.visits += vLoss
            #currNode.wins -= vLoss
            visited.append(currNode)
            hero = currNode.state.next_player
            tensor = encoder.encode(currNode.state)
            tensor = np.expand_dims(tensor, axis=0)
            p, v = nn.predict(tensor)
            currNode.expand(p.flatten())  # add children
        # Backpropagate
        while currNode:  # backpropagate from the expanded node and work back to the root node
            currNode.update(v if hero == currNode.state.next_player else -v,
                            vLoss)  # state is terminal. Update node with result from POV of node.playerJustMoved
            currNode = currNode.parentNode
    if stoch:
        return rootnode.childNodes,visited
    else:
        return max(rootnode.childNodes, key=attrgetter('visits'))




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

    def __members(self):
        return (self.move,self.state,self.childNodes,self.wins,self.visits,self.q,self.p,self.v)

    def __eq__(self, other) :

        if isinstance(other,MCTSNode) :
            comparison = self.move==other.move and self.state==other.state and \
                            self.wins==other.wins and self.visits==other.visits and \
                            self.q==other.q and self.p==other.p and self.v==other.v
            return comparison
        return False

    def __hash__(self):
        return hash(self.__members)

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
        if len(legal_moves)>1 :
            for move,p in moveProb :
                 if move in legal_moves :
                     # if we don't pass; all the moves are legal
                     if move.is_play:
                         next_board = copy.deepcopy(self.state.board)
                         next_board.place_stone(self.state.next_player, move.point)
                     else:
                         next_board = self.state.board
                     # self.state.moves should not matter as such so keeping it same as parent.
                     childState = GameState(next_board, self.state.next_player.opp, self.state, move, self.state.moves)
                     child = MCTSNode(state=childState, move=move, parent=self, p=p)
                     self.childNodes.append(child)

    def update(self, v,vLoss):
        """
            Update this node - one additional visit and v additional wins.
            v is given from the neural network and must be from the viewpoint of playerJustmoved.
        """
        #self.visits += -vLoss + 1
        #self.wins += vLoss + v
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

    def __init__(self,player,nn):

        self.player = player
        self.nn = nn
    """
    def run_simulation(self,simulations,encoder,rootnode):
        epsilon = 0.25
        dcoeff = [0.03]
        c=4
        vLoss=1
        stoch=True

        import tensorflow as tf
        import keras
        set_gpu_memory_target()
        print("PID: ", os.getpid())
        
        
        visited = set()
        input_shape = (7, 5, 5)
        nn = AGZ.init_random_model(input_shape)

        for i in range(simulations) :
            currNode = rootnode
            # if i>0 :
            for child in currNode.childNodes:
                # Dirichlet noise
                child.p = (1 - epsilon) * child.p + epsilon * np.random.dirichlet(alpha=dcoeff)
            # Select
            while currNode in visited:  # node is fully expanded and non-terminal
                currNode = currNode.SelectChild(c)
                currNode.visits += vLoss
                currNode.wins -= vLoss

            # Expand
            if currNode not in visited:  # if we can expand (i.e. state/node is non-terminal)
                currNode.visits += vLoss
                currNode.wins -= vLoss
                visited.add(currNode)
                hero = currNode.state.next_player
                tensor = encoder.encode(currNode.state)
                tensor = np.expand_dims(tensor, axis=0)
                p, v = nn.predict(tensor)
                currNode.expand(p.flatten())  # add children
            # Backpropagate
            while currNode:  # backpropagate from the expanded node and work back to the root node
                currNode.update(v if hero == currNode.state.next_player else -v,
                                vLoss)  # state is terminal. Update node with result from POV of node.playerJustMoved
                currNode = currNode.parentNode
        if stoch:
            return rootnode.childNodes
        else:
            return max(rootnode.childNodes, key=attrgetter('visits')).move
    """
    
    def select_move(self,rootnode,visited,encoder,simulations,numProc=3,epsilon = 0.25,dcoeff = [0.03],c=4,vLoss=1,stoch=True):
        """
        Conduct a tree search for simulations iterations starting from gameState.
        Assumes 2 alternating players(player 1 starts), with game results in the range[-1, 1].
        Return the child MCTS nodes of the root node/gameState for exploratory play, or the move with the most visits for competitive play.
        """
        """executor = concurrent.futures.ProcessPoolExecutor(2)
        simEach = int(simulations/numProc)
        futures = [executor.submit(self.run_simulation, rootnode, visited, encoder, simEach)
                   for i in range(numProc)]
        done,_ = concurrent.futures.wait(futures, return_when='ALL_COMPLETED')
        results = [future.result() for future in done]"""
        """results = []
        def collect_result(result):
            results.append(result)
        """
        start = time.time()
        numProc = 2
        simEach = int(simulations / numProc)
        visiteds = [copy.deepcopy(visited) for i in range(numProc)]
        pool = multiprocessing.Pool(numProc)
        results = [pool.apply_async(run_simulation,(simEach,encoder,rootnode,vis)) for vis in visiteds]
        results = [res.get() for res in results]
        pool.close()
        """
        cores = 1
        root_agents = [copy.deepcopy(rootnode) for _ in range(cores)]
        f = partial(self.run_simulation, int(simulations / cores), encoder)
        with Pool(processes=cores) as pool:
            results = [i for i in pool.imap_unordered(f, root_agents)]
        
        workers = []
        x = int(simulations / cores)
        
        for i in range(cores):
            worker = multiprocessing.Process(
                target=run_simulation,
                args=(
                    x,
                    encoder,
                    copy.deepcopy(rootnode),
                )
            )
            worker.start()
            workers.append(worker)

        # Wait for all workers to finish.
        print('Waiting for workers...')
        for worker in workers:
            worker.join()

            
        """    

        numNodes = len(results[0][0])
        aggNodes = []

        for i in range(numNodes) :
            aggNode = copy.deepcopy(results[0][0][i])
            for j,result in enumerate(results) :
                if j>0 :
                    aggNode.visits += result[0][i].visits
                    aggNode.wins += result[0][i].wins
                    aggNode.q += result[0][i].q
            aggNodes.append(aggNode)

        for result in results :
            for node in result[1] :
                if node not in visited :
                    visited.append(node)
        end = time.time()
        print('Time for move: ',str(end-start))
        return aggNodes,visited





class MCTSSelfPlay :

    def __init__(self,plane_size,board_size,model):

        self.board_size = board_size
        self.plane_size = plane_size
        self.model = model
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



    def play(self,expFile,num_games=2500,simulations=1600,c=4,vResign=-.7,tempMoves=10) :
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
            Player.black: MCTSPlayer(Player.black,self.model),
            Player.white: MCTSPlayer(Player.white,self.model)
        }

        for i in range(num_games) :


            game = GameState.new_game(self.board_size)
            moves = []
            moveNum = 0
            visited = []
            rootnode = None
            while not game.is_over() :
                moveNum += 1
                print('Move: '+str(moveNum))
                if moveNum <= tempMoves :
                    tau = 1
                else :
                    tau = 0.1
                if not rootnode:
                    rootnode  = MCTSNode(state = game)

                mctsNodes,visited =  players[game.next_player].select_move(rootnode,visited ,self.encoder,simulations,c=c)
                print('Visited size: '+str(len(visited)))
                tensor = self.encoder.encode(game)
                _, rootV = self.model.predict(np.expand_dims(tensor, axis=0))
                childVals = []
                searchProbAllM = np.zeros((self.board_size ** 2 + 1,), dtype='float')
                searchProbLegM = []
                for child in mctsNodes:
                    childTensor = self.encoder.encode(child.state)
                    childTensor = np.expand_dims(childTensor, axis=0)
                    _, childV = self.model.predict(childTensor)
                    if child.move.is_play:
                        r, c = child.move.point
                        i = self.board_size * r + c
                    else:
                        i = self.board_size ** 2
                    tVis = child.visits ** (1 / tau)
                    searchProbAllM[i] = tVis
                    searchProbLegM.append(tVis)
                    childVals.append(childV.item())
                probSum = sum(searchProbLegM)
                searchProbLegM = np.divide(searchProbLegM, probSum)
                searchProbAllM = np.divide(searchProbAllM, probSum)
                if rootV.item() < vResign and max(childVals) < vResign:
                    move = Move.resign()
                    print(str(game.next_player) + ' resigning')
                else:
                    rootnode = np.random.choice(a=mctsNodes, p=searchProbLegM)
                    move = rootnode.move

                moves.append((game, tensor, searchProbAllM))
                if move.is_play:
                    # print(game.next_player, alphaNumnericMove_from_point(move.point))
                    pass
                if move.is_pass:
                    # print(game.next_player, "PASS")
                    pass
                if move.is_resign:
                    # print(game.next_player, "Resign")
                    pass
                moves.append((game,tensor,searchProbAllM))
                game = game.apply_move(move)


            winner = game.winner()
            self.save_moves(moves,winner)

        model_input = np.array(self.expBuff.model_input)
        action_target = np.array(self.expBuff.action_target)
        value_target = np.array(self.expBuff.value_target)

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

if __name__ == "__main__" :
    input_shape = (7, 5, 5)
    nn = AGZ.init_random_model(input_shape)
    mctsSP = MCTSSelfPlay(7, 5, nn)

    mctsSP.play('./data/experience_1.hdf5', num_games=2, simulations=1600)
