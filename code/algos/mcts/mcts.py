class MCTSNode:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
    """

    def __init__(self, state, move=None, parent=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.state = state  # GameState object that the node represents

        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()  # future child nodes
        self.playerJustMoved = state.playerJustMoved  # the only part of the state that the Node needs later
        self.q = q  # q value of the node



    def SelectChild(self):
        """ Use Q+U to select action
        """
        # returns s which is a sorted list of child nodes according to win formula


    def expand(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """

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

    def UCT(self):
        """
        Returns
        UCT
        Score
        for node
        """

    def uct_select_move(simulations, verbose = False):
        """
            Conduct
        a
        tree
        search
        for itermax iterations starting from rootstate.
    Return
    the
    best
    move
    from the rootstate.
    Assumes
    2
    alternating
    players(player
    1
    starts), with game results in the range[-1, 1]."""
        # Select
# node is fully expanded and non-terminal

        # Expand
        # if we can expand (i.e. state/node is non-terminal)
        # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        # while state is non-terminal

        # Backpropagate
# backpropagate from the expanded node and work back to the root node
# state is terminal. Update node with result from POV of node.playerJustMoved


    # Output some information about the tree - can be omitted

# return the move that was most visited


class MCTSPlayer :

	def __init__(self,player,encoder):

		self.player = player
		self.encoder = encoder

	def save_move(self, move,search_prob,winner=None) :
	    """
        Save input feature stack, search probabilities,and game winner to disk
        """

	def select_move(self,state,saveToDisk=False) :
		 """ Takes in a GameState object representing the current game state and selects the best move using MCTS.Returns a Move object.Provides optional parameter to save the move to disk ( for use in self-play for training neural network)
        """



class MCTSSelfPlay :

    def __init__(self,encoder):

        self.encoder = encoder

    def play(self,num_game=25000) :
        """
        :param num_game:
        :return:
         Play num_games of self play against two MCTS players and save move information to disk. """
