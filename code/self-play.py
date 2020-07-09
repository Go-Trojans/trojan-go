import argparse
import datetime
import time

from algos.mcts.mcts import MCTSSelfPlay
from algos.nn import AGZ

"""
Simulate "num_games" games using self-play rl and MCTS
"""
def main(i):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games', '-n', type=int, default=2500)

    args = parser.parse_args()
    
    filename = "./data/experience_" + str(i) + ".hdf5"
    mctsSP = MCTSSelfPlay(7,5)
    input_shape = (7,5,5)
    nn = AGZ.init_random_model(input_shape)
    mctsSP.play(nn,filename,num_games=args.num_games,simulations=400)
    #mctsSP.play(nn,filename,num_games=10,simulations=400)
    

if __name__ == '__main__':
    """ iteration = n: collect examples/gameStates from 2500 self-play games, n times """
    iteration = 1
    iteration_start = time.time()
    for i in range(iteration):
        main(i)

    iteration_end = time.time()
    print("Total time taken to generate {} set of examples file(s) is {}".format(iteration, iteration_end - iteration_start))
