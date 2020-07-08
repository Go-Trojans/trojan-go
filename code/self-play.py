from algos.mcts.mcts import MCTSSelfPlay
from algos.nn import AGZ


if __name__ == "__main__" :

    mctsSP = MCTSSelfPlay(7,5)
    input_shape = (7,5,5)
    nn = AGZ.init_random_model(input_shape)
    mctsSP.play(nn,'./data/experience_2.hdf5',num_games=2500,simulations=400)
