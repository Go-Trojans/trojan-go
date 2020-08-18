from keras.models import load_model, save_model
#import keras
import h5py
import os
import tempfile
import numpy as np
from algos import gohelper
# import keras
import pprint
import logging
logging = logging.getLogger(__name__)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
    from tensorflow.python.client import device_lib
    from tensorflow import config
    pp = pprint.PrettyPrinter(indent=4)
    import platform
    import socket
    import re
    import uuid
    import json
    import psutil
    import os

    # The below line may create tensorflow memory usage issue;
    # can be resolved using allow_growth (check AGZ.py)
    local_device_protos = device_lib.list_local_devices()
    print("*"*60)
    logging.debug("\n\n")
    logging.debug(
        "************************************************************")
    print(f"{bcolors.OKGREEN}System Info ...{bcolors.ENDC}")
    logging.debug("System Info ...")
    print(f"{bcolors.BOLD}GPU/CPU Info ...{bcolors.ENDC}")
    logging.debug("GPU/CPU Info ...")
    print(f"{bcolors.OKBLUE}{local_device_protos}{bcolors.ENDC}")
    logging.debug("{}".format(local_device_protos))
    del device_lib

    info = {}
    info['platform'] = platform.system()
    info['platform-release'] = platform.release()
    info['platform-version'] = platform.version()
    info['architecture'] = platform.machine()
    info['hostname'] = socket.gethostname()
    info['ip-address'] = socket.gethostbyname(socket.gethostname())
    info['mac-address'] = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
    info['processor'] = platform.processor()
    info['ram'] = str(
        round(psutil.virtual_memory().total / (1024.0 ** 3)))+" GB"
    info['Cores'] = os.cpu_count()
    info['GPUs'] = len(config.experimental.list_physical_devices('GPU'))
    del config
    print("\n")
    logging.debug("\n")
    for x, y in info.items():
        print(f"{bcolors.OKBLUE}{x}:{y}{bcolors.ENDC}")
        logging.debug("{}:{}".format(x, y))
    # print("")
    print("*"*60)
    logging.debug(
        "************************************************************")


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
    """
    [GPU-ERROR] : Unable to load the model when we need during training under tf.dustribute.Strategy scope.
    """
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
    if int(tf_v[0]) == 2 and int(tf_v[1]) >= 0:
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
    #import keras
    import os
    # if keras.backend.backend() != 'tensorflow':
    #    print("Return without doing anything")
    #    return
    # Do the import here, not at the top, in case Tensorflow is not
    # installed at all.
    #import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    # if tf_version_comp(tf.__version__):
    if True:

        """
        To force the process to use a specific GPU, I use the environment variable CUDA_VISIBLE_DEVICES,
        which is independent from the master process which forked the worker process.
        So, with 4 GPUs machine and 32 cores/process, each GPUs will have 8 processes running.
        GPU0 = 1,5,...
        GPU1 = 2,6
        GPU2 = 3,7
        GPU3 = 4,9
        """
        gpu_id = '0'
        #n_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
        n_gpu = 8
        print(
            f"{bcolors.OKBLUE}[set_gpu_memory_target] Number of GPUs: {n_gpu}{bcolors.ENDC}")
        if n_gpu > 0:
            gpu_id = np.remainder(os.getpid(), n_gpu)
            # can be commented out.
            # giving error (though all GPUs were working) (logfile: cuda_error1.log)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # os.environ["CUDA_VISIBLE_DEVICES"] = "1"           # out of memory on GPU:1 but no error
            print(
                f"{bcolors.OKBLUE}[set_gpu_memory_target] PID={os.getpid()} gpu_id={str(gpu_id)}{bcolors.ENDC}")

        # ---------------------------------------------------------------------------------
        # this block enables GPU enabled multiprocessing
        print(
            f"{bcolors.OKBLUE}[set_gpu_memory_target config START] PID={os.getpid()} gpu_id={str(gpu_id)}{bcolors.ENDC}")

        """
        config = tf.compat.v1.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = frac #not needed I guess
        config.gpu_options.visible_device_list = str(gpu_id)
        config.gpu_options.allow_growth = True
        # set_session(tf.compat.v1.Session(config=config))
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)
        """

        import keras
        import tensorflow as tf

        # On CPU/GPU placement
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        tf.compat.v1.Session(config=config)

        print(
            f"{bcolors.OKBLUE}[set_gpu_memory_target config END] PID={os.getpid()} gpu_id={str(gpu_id)}{bcolors.ENDC}")
        # ----------------------------------------------------------------------------------

    else:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = frac
        config.gpu_options.allow_growth = True
        # set_session(tf.Session(config=config))
        session = tf.Session(config=config)
