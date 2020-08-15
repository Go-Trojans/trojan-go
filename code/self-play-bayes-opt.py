import argparse
import datetime
import time
import inspect


import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import shutil 
import os
import random
import tempfile
from collections import namedtuple

import h5py
import numpy as np

from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.plots import plot_objective
from skopt.plots import plot_convergence
from skopt import gp_minimize

# from bayes_opt import BayesianOptimization

from algos.mcts.mcts import MCTSSelfPlay, ExperienceBuffer, load_experience, combine_experience
from algos.nn.AGZ import smallNN, init_random_model
from algos.godomain import *
from algos.gohelper import *
from algos.utils import set_gpu_memory_target, save_model_to_disk, load_model_from_disk, display_board, print_loop_info, system_info, bcolors, LOG_FORMAT
from algos.encoders.trojangoPlane import TrojanGoPlane
from algos.mcts.mcts import MCTSPlayer, MCTSNode
from algos.bayes_agent import bayesAgent

#import keras
#import tensorflow as tf


#global graph
#graph = tf.compat.v1.get_default_graph()


import logging
LOG_FILENAME = './trojango.log'
logging.basicConfig(filename=LOG_FILENAME,filemode='a', format=LOG_FORMAT, level=logging.DEBUG)
logging = logging.getLogger(__name__)

ref_hps = (0.03, 10, 4)
num_times_obj_func_called = 0

def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='algo-train')
    os.close(fd)
    print("temp file : ", fname)
    return fname



""" Here both the agents are nn model 
    which will help during move selection during MCTS simulation 
"""
def simulate_game(black_player, black_player_bayes, white_player, white_player_bayes, board_size, simulations, dcoeff=[0.03], c=4):
    plane_size = 7
    encoder = TrojanGoPlane((board_size,board_size),plane_size)
    moves = []
    game = GameState.new_game(board_size)
    #agents are smallNN() NN
    """
    agents = {
        Player.black: smallNN(black_player),
        Player.white: smallNN(white_player),
    }
    while not game.is_over():
        display_board(game.board)
        next_move = agents[game.next_player].select_move(encoder, game)
        print(game.next_player, next_move.point)
        moves.append(next_move)
        game = game.apply_move(next_move)

    #display_board(game.board)
    return game
    """
 
    agents = {
        Player.black: MCTSPlayer(Player.black, black_player, black_player_bayes),
        Player.white: MCTSPlayer(Player.white, white_player, white_player_bayes),
    }

    visited = set()
    rootnode = None
    while not game.is_over() :
        if not rootnode:
            rootnode  = MCTSNode(state = game)
        # Assuming selected_actionNode is a valid move 
        # Also, selected move is using Dirichlet Noise but not "TAU"
        print("Player {} select move with simulations= {} dcoeff={} c={} ".format(game.next_player, agents[game.next_player].bayes_agent.simulations, agents[game.next_player].bayes_agent.dirichlet, agents[game.next_player].bayes_agent.c))
        selected_actionNode = agents[game.next_player].select_move(
                                          rootnode, visited,
                                          encoder,
                                          simulations=int(agents[game.next_player].bayes_agent.simulations),
                                          dcoeff=agents[game.next_player].bayes_agent.dirichlet,
                                          c=agents[game.next_player].bayes_agent.c, 
                                          stoch=False)

        # update new rootnode as selected_actionNode
        rootnode = selected_actionNode
        move =  selected_actionNode.move
        #print(game.next_player, move.point) 
        game = game.apply_move(move)
        
    #display_board(game.board)
    return game                                                       
     

""" agent1_fname (learning_agent) & agent2_fname (reference_agent) are nn models/agents in (.json, .h5) format """
def play_games(args):
    
    print("I am in play_games")
    
    if len(args) == 6:
        agent1_fname, agent2_fname, num_games, board_size, gpu_frac, simulations = args
        dcoeff, simulations, c = ref_hps
        ref_hp = (0.03, 10, 4)
    else:
        agent1_fname, agent2_fname, num_games, board_size, gpu_frac, simulations, dcoeff, c, ref_hp = args
        print("I am in play_games else part , num_games: ", num_games)

    set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_model_from_disk(agent1_fname) # learning agent; should be variable and is using x hyperParams
    agent2 = load_model_from_disk(agent2_fname) #refernece agent; used x{0} or previous fixed hyperParams value.
    
    bayes_agent1 = bayesAgent(agent1, dcoeff, int(simulations), c) # this will keep on changing as per bayesian optimization

    ref_dcoeff, ref_sims, ref_c = ref_hp
    print("=" * 100)
    print("reference hyperparams: %s" % (ref_hp,))
    print("=" * 100)
    bayes_agent2 = bayesAgent(agent2, [ref_dcoeff], ref_sims, ref_c) # fixed hyperParams 
    
    wins, losses = 0, 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
            black_player_bayes, white_player_bayes = bayes_agent1, bayes_agent2
            print("Agent 1 playing as black and Agent 2 as white")
        else:
            white_player, black_player = agent1, agent2
            white_player_bayes, black_player_bayes = bayes_agent1, bayes_agent2
            print("Agent 1 playing as white and Agent 2 as black")
        
        # black_player_bayes and white_player_bayes are bayesian Agents  (Model model, dir, simulations, c). Last 3 arguments are dummy now.   
        game = simulate_game(black_player, black_player_bayes,
                             white_player, white_player_bayes,
                             board_size, int(simulations), dcoeff, c)
        if game.winner() == color1.value:
            print('Agent 1 wins')
            wins += 1
        elif game.winner() == color1.opp.value:
            print('Agent 2 wins')
            losses += 1
        else:
            print('Game is a draw', game.winner())

        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.opp
    return wins, losses


""" learning_agent & reference_agent are nn models/agents in (.json, .h5) format """
def evaluate(learning_agent, reference_agent,
             num_games, num_workers, board_size, simulations):
    games_per_worker = num_games // num_workers
    gpu_frac = 0.95 / float(num_workers)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            learning_agent, reference_agent,
            games_per_worker, board_size, gpu_frac, simulations,
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(play_games, worker_args)

    total_wins, total_losses = 0, 0
    for wins, losses in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Refrnce: %d' % total_losses)
    pool.close()
    pool.join()
    return total_wins



""" agent1_filename (learning agent) and  agent2_filename (reference agent)
    are model in  (.json, .h5) format 
""" 
def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games, simulations, temperature,
                 experience_filename,
                 gpu_frac):
    
    import tensorflow as tf
    import keras
    #import keras.backend as K
    #K.set_session(tf.compat.v1.Session())
    #tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())
    
    #print(inspect.currentframe().f_code.co_name, inspect.currentframe().f_back.f_code.co_name)
    set_gpu_memory_target(gpu_frac)

    #import tensorflow as tf
    #import keras
    
    #global graph
    #graph = tf.compat.v1.get_default_graph()
    
    print("PID: ", os.getpid())
    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    print("learning agent : {} \nreference_agent : {}".format(agent1_filename, agent2_filename))
        
    #print("Loading model from disk ...")
    agent1 = load_model_from_disk(agent1_filename)
    agent2 = load_model_from_disk(agent2_filename)
    
    #print(agent1.summary())
    #print(agent2.summary())
    
    
    """
    # _filename is a model saved in .hdf5 format.
    agent1 = tf.keras.models.load_model(agent1_filename)
    print(agent1.summary())
    agent2 = tf.keras.models.load_model(agent2_filename)
    """
    
    mctsSP = MCTSSelfPlay(7,5)
    input_shape = (7,5,5)
    #nn = AGZ.init_random_model(input_shape)
    print(f"{bcolors.OKBLUE} [PID : {os.getpid()}] self-play game is triggered, Get A Cup Of Coffee And Relax !!!{bcolors.ENDC}")
    logging.debug("[PID : {}] self-play game is triggered, Get A Cup Of Coffee And Relax !!!".format(os.getpid()))
    # agent1 and agent2 are nn model
    mctsSP.play(agent1, agent2,
                experience_filename,
                num_games=num_games, simulations=simulations)


""" learning_agent = (learning_agent_json, learning_agent_h5) and same is reference_agent """
def generate_experience(learning_agent, reference_agent, exp_file,
                        num_games, simulations, board_size, num_workers):
    temperature=0
    experience_files = []
    workers = []
    gpu_frac = 0.95 / float(num_workers)
    games_per_worker = num_games // num_workers
    
    for i in range(num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                board_size,
                learning_agent,
                reference_agent,
                games_per_worker,
                simulations,
                temperature,
                filename,
                gpu_frac,
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print('Waiting for workers...')
    for worker in workers:
        worker.join()
    """
    filename = get_temp_file()
    experience_files.append(filename)
    do_self_play(board_size, learning_agent,
                 reference_agent, games_per_worker,
                 simulations,temperature,filename, gpu_frac)
    """
    # Merge experience buffers.
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filenames = experience_files[1:]
    ###########################################
    with h5py.File(first_filename, 'r') as expf:
        combined_buffer = load_experience(expf)
        print("Examples in file {} is {}".format(first_filename, combined_buffer.model_input.shape[0]))
    for filename in other_filenames:
        with h5py.File(filename, 'r') as expf:
            next_buffer = load_experience(expf)
            print("Examples in next file {} is {}".format(filename, next_buffer.model_input.shape[0]))
        combined_buffer = combine_experience([combined_buffer, next_buffer])
        print("Examples in combined buffer after file {} is {}".format(filename, combined_buffer.model_input.shape[0]))
        
    print(f'{bcolors.OKBLUE}Finally Saved experiences into {exp_file}. Please check the size to verify.{bcolors.ENDC}')
    logging.debug("Finally Saved experiences into {}. Please check the size to verify.".format(exp_file))
    with h5py.File(exp_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)
    

"""  learning_agent (.json, .h5) and output_file is for storing (.json, .h5) """ 
def train_worker(learning_agent, output_file, experience_file,
                 lr, batch_size):
    MCTSSelfPlay(7,5,learning_agent).train(experience_file, output_file, lr, batch_size)


"""  learning_agent (.json, .h5) and output_file is for storing (.json, .h5) """ 
def train_on_experience(learning_agent, output_file, experience_file,
                        lr=0.04, batch_size=128):
    # Do the training in the background process. Otherwise some Keras
    # stuff gets initialized in the parent, and later that forks, and
    # that messes with the workers.
    worker = multiprocessing.Process(
        target=train_worker,
        args=[
            learning_agent,
            output_file,
            experience_file,
            lr,
            batch_size
        ]
    )
    worker.start()
    worker.join()


def main():
    # code here
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-per-batch', '-g', type=int, default=2500)
    parser.add_argument('--simulations', type=int, default=400) # for 5*5 board
    parser.add_argument('--board-size', '-b', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-per-eval', type=int, default=400)
    parser.add_argument('--production', dest='production', action='store_true')
    parser.add_argument('--no-production', dest='production', action='store_false')
    parser.add_argument('--tuning', '-opt', action='store_true')
    parser.set_defaults(production=True)
    
    args = parser.parse_args()        

    print(f"{bcolors.OKBLUE}Welcome to TROJAN-GO !!!{bcolors.ENDC}")
    logging.debug("Welcome to TROJAN-GO !!!")
    system_info()

    agents_path = './checkpoints/iteration_Savedmodel/'
    data_dir = './data/'
    """ These .json & .hdf5 files are nothing but neural network architecture and weights """
    learning_agent_json = agents_path + 'initial.json'
    learning_agent_h5 = agents_path + 'initial.h5'
    learning_agent = (learning_agent_json, learning_agent_h5)
    
    reference_agent_json = agents_path + 'initial.json'
    reference_agent_h5 = agents_path + 'initial.h5'
    reference_agent = (reference_agent_json, reference_agent_h5)

    """
    # Another way of referring the inital model but not good for transfer learning.
    input_shape = (7,5,5)
    model = init_random_model(input_shape)
    save_model_to_disk(model, learning_agent)
    save_model_to_disk(model, reference_agent)
    """
    
    experience_file = os.path.join(data_dir, 'exp_temp.hdf5') # examples data to be stored.

    tmp_agent_json = os.path.join(agents_path, 'agent_temp.json')
    tmp_agent_h5 = os.path.join(agents_path, 'agent_temp.h5')
    tmp_agent = (tmp_agent_json, tmp_agent_h5)

    
    working_agent_json = os.path.join(agents_path, 'agent_cur.json')
    working_agent_h5 = os.path.join(agents_path, 'agent_cur.h5')
    working_agent = (working_agent_json, working_agent_h5)
    
    num_cpu = os.cpu_count()
    if not args.production:
        args.games_per_batch = num_cpu * 10
        args.num_workers = num_cpu
        args.simulations = 10
        args.num_per_eval = num_cpu
        
    total_games = 0
    iter_count = 1
    prod = True

    failed_once = 0 

    """ learning_agent & reference_agent are nn models/agents in (.json, .h5) format """
    # define dimensions for Bayesian Opt.
    dim_sims = Integer(name='simulations', low=10, high=40)
    dim_dcoeff = Real(name='dcoeff', low=0.01, high=0.03)
    dim_c = Integer(name='c', low=3, high=5)
    dimensions = [dim_sims, dim_dcoeff, dim_c]
    hp_game_count = 28
    #num_times_obj_func_called = 0
    @use_named_args(dimensions)
    def obj_func(simulations, dcoeff, c):
        global num_times_obj_func_called
        num_times_obj_func_called = num_times_obj_func_called + 1
        games_per_worker = hp_game_count // args.num_per_eval
        gpu_frac = 0.95 / float(args.num_workers)
        pool = multiprocessing.Pool(args.num_workers)
        worker_args = [
            (
                learning_agent, reference_agent,
                games_per_worker, args.board_size, gpu_frac, simulations, [dcoeff], c, ref_hps
            )
            for _ in range(args.num_workers)
        ]
        game_results = pool.map(play_games, worker_args)

        total_wins, total_losses = 0, 0
        epsilon = 0.0001
        for wins, losses in game_results:
            total_wins += wins
            total_losses += losses
        print(f"{bcolors.OKBLUE}FINAL RESULTS  num_times_obj_func_called:{num_times_obj_func_called}/10{bcolors.ENDC}")
        print('Parameters: %s, %s, %s' % (simulations, dcoeff, c))
        print('Learner: %d' % total_wins)
        print('Reference: %d' % total_losses)
        print('Win rate: %f' % (total_wins/(total_wins + total_losses + epsilon)))
        pool.close()
        pool.join()
        return total_losses / (total_wins + total_losses + epsilon) # want to minimize loss rate 

#     ref_hps = (0.03, 10, 4)
    while prod:
        loop_start = time.time()
        print(f"{bcolors.OKBLUE}[Data Generation starts] Reference: {reference_agent_json} {bcolors.ENDC}")
        logging.debug("[Data Generation starts] Reference: {}".format(reference_agent_json))
        ge_start = time.time()
        generate_experience(
            learning_agent,
            reference_agent,
            #agent1, agent2,
            experience_file,
            args.games_per_batch,
            args.simulations,
            args.board_size,
            num_workers=args.num_workers)
        ge_end = time.time()
        exp_time = ge_end - ge_start
                      
        print(f"{bcolors.OKBLUE}[Data Generation finish] Time taken to finish generate experience with multiprocessing({num_cpu}) is : {exp_time} {bcolors.ENDC}")
        logging.debug("[Data Generation finish] Time taken to finish generate experience with multiprocessing({}) is : {}".format(num_cpu, exp_time))
        print(f"{bcolors.OKBLUE}[Start Training ...] {bcolors.ENDC}")
        logging.debug("[Start Training ...]")
                      
        train_start = time.time()
        train_on_experience(
            learning_agent, tmp_agent, experience_file)
        total_games +=  args.games_per_batch
        train_end = time.time()
        train_time = train_end - train_start
        print(f"{bcolors.OKBLUE}[Training ends !!!] {bcolors.ENDC}")
        logging.debug("[Training ends !!!]")            
        
        # Eval Params: 400 games , "TAU"=0 , 400 simulations per move
        print(f"{bcolors.OKBLUE}[Evaluation starts] ... \nlearning agent {learning_agent} & \nreference_agent {reference_agent}{bcolors.ENDC}")
        logging.debug("[Evaluation starts] ... \nlearning agent {} & \nreference_agent {}".format(learning_agent, reference_agent))            
        #num_games_eval = args.num_per_eval
        num_games_eval = hp_game_count


        eval_start = time.time()
        print ("=" * 30)
        print(f'{bcolors.OKBLUE}[Start hyperparameter tuning]\n{bcolors.ENDC}')
   

        # pbounds = {'simulations': (10, 13), 'dcoeff': (0.01, 0.04), 'c':(4, 6)}
        # optimizer = BayesianOptimization(f=obj_func, pbounds=pbounds, random_state=1)
        # optimizer.maximize(init_points=2, n_iter=3)
        try:
            results = gp_minimize(obj_func, dimensions=dimensions, n_calls=10, acq_func='EI', x0=None, y0=None, noise=1e-8)
            print("obj_func is f(x) where x are hyperparams and it is returnung loss/total, results.fun = " ,results.fun)
            # TODO : Don't we need to update the hyperparams so that it can be used in self-play of next iteration.
            # Also Objective function should have x{0} to be used by refernce agent and x (EI Point) to be used by learning agent.
            wins = hp_game_count-int(results.fun * hp_game_count)
            print('-' * 30)
            print("wins: %s" % wins)
            print("Finished hyperparameter tuning")
            # print ("Found max: %s" % optimizer.max)
            print('-' * 100)
            print(f"{bcolors.OKBLUE}Best params: {results.x} wins: {wins} {bcolors.ENDC}")
            global ref_hps
            ref_hps = results.x
            print('-' * 100)

            global num_times_obj_func_called
            num_times_obj_func_called = 0 
            print('Won %d / %d games (%.3f)' % (wins, hp_game_count, float(wins) / hp_game_count))
            plot_convergence(results, yscale='log').figure.savefig("convergence.png")
            plot_objective(results).flatten()[0].figure.savefig("objective.png")
            
            print ("=" * 30)
        except ValueError:
            print("Optimization Failed!!")
            if failed_once > 10:
                raise ValueError
            else:
                failed_once += 1
                wins = evaluate(
                    learning_agent, reference_agent,
                    num_games=num_games_eval,
                    num_workers=args.num_workers,
                    board_size=args.board_size,
                    simulations=args.simulations)
                print('Won %d / %d games (%.3f)' % (
                    wins, num_games_eval, float(wins) / num_games_eval))
        eval_end = time.time()
        eval_time = eval_end - eval_start

        shutil.copy(tmp_agent_json, working_agent_json)
        shutil.copy(tmp_agent_h5, working_agent_h5)
        learning_agent = working_agent
        if wins >= int(np.multiply(num_games_eval,0.55)):
            next_filename_json = os.path.join(
                agents_path,
                'agent_%08d.json' % (total_games,))
            next_filename_h5 = os.path.join(
                agents_path,
                'agent_%08d.h5' % (total_games,))                
                
            shutil.move(tmp_agent_json, next_filename_json)
            shutil.move(tmp_agent_h5, next_filename_h5)
            next_filename = (next_filename_json, next_filename_h5)
            reference_agent = next_filename
            print('-' * 30)
            print(f"{bcolors.OKBLUE}[Evaluation ends] New reference is : {next_filename} {bcolors.ENDC}")
            logging.debug("[Evaluation ends] New reference is : {}".format(next_filename))        
        else:
            print(f'{bcolors.OKBLUE}[Evaluation ends] Keep learning\n{bcolors.ENDC}')
            logging.debug("[Evaluation ends] Keep learning\n")
            
        
        loop_end = time.time()
        loop_time = loop_end - loop_start

         
        info = print_loop_info(iter_count, learning_agent, reference_agent,
                               args.games_per_batch, args.simulations,
                               args.num_workers, args.num_per_eval,
                                      exp_time, train_time, eval_time, loop_time)
        print(f"{bcolors.OKBLUE}{info}{bcolors.ENDC}")
        logging.debug("{}".format(info))
        #print(info)
        iter_count = iter_count + 1
        # if not args.production:
        #     prod = False


if __name__ == '__main__':
    main()
