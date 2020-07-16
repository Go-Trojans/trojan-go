import numpy as np
from algos import gohelper
#import keras




def print_loop_info(iteration, learning_agent,
                    refernece_agent, num_games_per_iter,
                    simulations, workers, num_eval_games,
                    exp_time, train_time, eval_time, loop_time):
    s = f"""
    {'-'*40}
    # Operator Micro-benchmarks
    # Iteration : {iteration}
    # Learning Agent : {learning_agent}
    # Reference Agent : {refernece_agent}
    # Games Per Batch : {num_games_per_iter}
    # Simulations per move : {simulations}
    # Number of workers : {workers}
    # Number of Games for Eval : {num_eval_games}
    # Experience Time : {exp_time}
    # Training time : {train_time}
    # Evaluation time : {eval_time}
    # Total Loop time : {loop_time}
    
    {'-'*40}
    """
    return s

def system_info():
    import platform,socket,re,uuid,json,psutil,logging,os
    info={}
    info['platform']=platform.system()
    info['platform-release']=platform.release()
    info['platform-version']=platform.version()
    info['architecture']=platform.machine()
    info['hostname']=socket.gethostname()
    info['ip-address']=socket.gethostbyname(socket.gethostname())
    info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
    info['processor']=platform.processor()
    info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
    info['Cores']=os.cpu_count()


    print("*"*60)
    print("System Info ...")
    print(info)    
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    del device_lib
    print("\n\n")
    print("*"*60) 




COLUMNS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
PLAYER_TO_CHAR = {
    None: ' . ',
    gohelper.Player.black: ' x ',
    gohelper.Player.white: ' o ',
}

LOG_FORMAT = "%(asctime)s -- [%(pathname)s]:%(levelname)s %(message)s"

def display_board(board):
    for row in range(board.board_width):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(board.board_height):
            player_val = board.grid[row][col]
            if player_val == 0:
                player = None
            elif player_val == 1:
                player = gohelper.Player.black
            else:
                player = gohelper.Player.white
            line.append(PLAYER_TO_CHAR[player])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLUMNS[:board.board_height]))



def point_from_alphaNumnericMove(alphaNumnericMove):
    col = COLUMNS.index(alphaNumnericMove[0])
    row = int(alphaNumnericMove[1:])
    return gohelper.Point(row=row, col=col)


def alphaNumnericMove_from_point(point):
    return '%s%d' % (
        COLUMNS[point.col],
        point.row
    )



import tempfile
import os

import h5py
import keras
from keras.models import load_model, save_model

""" file_json_h5 format (.json, .h5) """
def save_model_to_disk(model, file_json_h5):
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_json_h5[0], "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_json_h5[1])
    print("Saved model to disk")

""" agent_filename is in (.json , .h5) format """
def load_model_from_disk(agent_filename):
    from keras.models import model_from_json

    agent_json_filepath = agent_filename[0]
    agent_h5_filepath = agent_filename[1]
    
    # load json and create model
    json_file = open(agent_json_filepath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(agent_h5_filepath)
    return loaded_model



""" return True if tensorflow version is equal or higher than 2.0, else False """
def tf_version_comp(tf_v):
    tf_v = tf_v.split(".")
    if int(tf_v[0]) == 2 and int(tf_v[1]) >=0:
        return True
    return False
    


""" AGZ.py or other nn files may also have some setting on gpu,
     so check that too.
"""
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
        #set_session(tf.compat.v1.Session(config=config))
        session = tf.compat.v1.Session(config=config)
        #tf.compat.v1.keras.backend.set_session(session)

    else:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = frac
        #set_session(tf.Session(config=config))
        session = tf.Session(config=config)


