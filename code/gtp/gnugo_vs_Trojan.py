from subprocess import Popen, PIPE

from gtp import parse_vertex, gtp_move, gtp_color
from gtp import BLACK, WHITE, PASS


import os,sys,inspect
# reset the working dir
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from algos import godomain, gohelper
from algos.agent import randombot
from algos.utils import display_board

'''
Things to set before game:
1. Board Size
2. Komi

To do GNU vs Trojan, set GNU as host and start two subprocess for both Trojan and GNU

'''

## TODO: Name changes and function combine

class GTPSubProcess(object):

    def __init__(self, label, args):
        self.label = label
        self.subprocess = Popen(args, stdin=PIPE, stdout=PIPE, universal_newlines=True, bufsize = 1)
        print("{} subprocess created".format(label))

    def send(self, data):
        print("sending {}: {}".format(self.label, data))
        self.subprocess.stdin.write(data)
        result = ""
        while True:
            data = self.subprocess.stdout.readline()
            if not data.strip():
                break
            result += data
        print("got: {}".format(result))
        return result

    def close(self):
        print("quitting {} subprocess".format(self.label))
        self.subprocess.communicate("quit\n")


class GTPFacade(object):

    def __init__(self, label, args):
        self.label = label
        self.gtp_subprocess = GTPSubProcess(label, args)

    def name(self):
        self.gtp_subprocess.send("name\n")

    def version(self):
        self.gtp_subprocess.send("version\n")

    def boardsize(self, boardsize):
        self.gtp_subprocess.send("boardsize {}\n".format(boardsize))

    def komi(self, komi):
        self.gtp_subprocess.send("komi {}\n".format(komi))

    def clear_board(self):
        self.gtp_subprocess.send("clear_board\n")

    def genmove(self, color):
        message = self.gtp_subprocess.send(
            "genmove {}\n".format(gtp_color(color)))
        assert message[0] == "="
        return parse_vertex(message[1:].strip())

    def showboard(self):
        self.gtp_subprocess.send("showboard\n")

    def play(self, color, vertex):
        self.gtp_subprocess.send("play {}\n".format(gtp_move(color, vertex)))

    def final_score(self):
        self.gtp_subprocess.send("final_score\n")

    def close(self):
        self.gtp_subprocess.close()


GNUGO = ["gnugo", "--mode", "gtp"]
GNUGO_LEVEL_ONE = ["gnugo", "--mode", "gtp", "--level", "1"]
GNUGO_MONTE_CARLO = ["gnugo", "--mode", "gtp", "--monte-carlo"]


def gnugov_vs_trojan(boardsize, komi, Trojanbots, black='Trojan'):
    '''
    on 5*5 board:

    Calling invention: GNU A1 = (1,1), B3 = (2,3) indexed from 1
                     : Trojan A1 = (1,1), B3 = (3,1) # row 3 column 1 indexed from 0
    '''
    game = godomain.GameState.new_game(boardsize)
    if black is 'Trojan':
        gnu = GTPFacade("white", GNUGO_MONTE_CARLO)
        gnu.boardsize(boardsize)
        gnu.komi(komi)
        gnu.clear_board()

        first_pass = False

        while True:
            # First select move
            vertex = Trojanbots.select_move(game)
            # Then Apply move if its not a PASS
            game = game.apply_move(vertex)

            if not vertex.point:
                if first_pass:
                    break
                else:
                    first_pass = True
            else:
                first_pass = False
            # Change Cord of Trojan move to GNU move
                vertex = (vertex.point[1] + 1, boardsize - vertex.point[0])
                # GNU play that move
                gnu.play(BLACK, vertex)

            gnu.showboard()
            #display_board(game.board)

        # Trojan has finish his first move
            # genmove applied the move it generated, so GNU finished
            vertex = gnu.genmove(WHITE)
            if vertex == PASS:
                if first_pass:
                    break
                else:
                    first_pass = True
            else:
                first_pass = False
                # Now translate the move to Trojan language
                vertex = godomain.Move(gohelper.Point(boardsize - vertex[1], vertex[0] - 1))
                game = game.apply_move(vertex)

            gnu.showboard()
            #display_board(game.board)

        gnu.final_score()
        gnu.close()

    else:
        gnu = GTPFacade("black", GNUGO)
        gnu.boardsize(boardsize)
        gnu.komi(komi)
        gnu.clear_board()

        first_pass = False

        while True:
            vertex = gnu.genmove(BLACK)
            if vertex == PASS:
                if first_pass:
                    break
                else:
                    first_pass = True
            else:
                first_pass = False
                vertex = godomain.Move(gohelper.Point(boardsize - vertex[1], vertex[0] - 1))
                game = game.apply_move(vertex)
            gnu.showboard()

            vertex = Trojanbots.select_move(game)
            if not vertex.point:
                if first_pass:
                    break
                else:
                    first_pass = True
            else:
                first_pass = False
                vertex = (vertex.point[1] + 1, boardsize - vertex.point[0])
                gnu.play(WHITE, vertex)

            gnu.showboard()

        gnu.final_score()
        gnu.close()


gnugov_vs_trojan(5, 2.5, randombot.RandomBot())
