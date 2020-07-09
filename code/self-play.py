from algos.mcts.mcts import MCTSSelfPlay
from algos.nn import AGZ

"""
Simulate "num_games" games using self-play rl and MCTS
"""
def main():
    filename = "./data/experience_" + str(i) + ".hdf5"
    mctsSP = MCTSSelfPlay(7,5)
    input_shape = (7,5,5)
    nn = AGZ.init_random_model(input_shape)
    mctsSP.play(nn,filename,num_games=10,simulations=400)
    

if __name__ == '__main__':
    """ iteration = n: collect examples/gameStates from 2500 self-play games, n times """
    iteration = 1
    for i in range(iteration):
        main(i)
