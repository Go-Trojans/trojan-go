/* Regular MCTS File*/

#include <iostream>
using namespace std;
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <list>
#include <algorithm>
#include "../godomain.h"
#include "../gohelper.h"
#include "H5Cpp.h"
#include <math.h>
using namespace H5;

class MCTSNode
{
public:
    Move *move;
    GameState *state;
    MCTSNode *parentNode;
    vector<MCTSNode> childNodes;
    int wins;
    int visits;
    float q;
    float p;
    float v;

    MCTSNode(GameState *state,float p=0.0,float v=0.0,Move *move=NULL,MCTSNode *parentNode=NULL)
    {
        this->move = move;
        this->state = state;
        this->parentNode = parentNode;
        this->p = p;
        this->v = v; 
         
    }

    MCTSNode select(int c=4)
    {   
        vector<float> puctNodes;
        for (vector<float>::const_iterator it = this->childNodes.begin(); it != this->childNodes.end(); ++it) {
            puctNodes.push_back(it.puct(c));
        }
        int maxChild = distance(puctNodes.begin(), max_element(puctNodes.begin(), puctNodes.end()));
        return this->childNodes[maxChild]
    }

    void expand(vector<float> probs)
    {
        int boardSize = this->state->board->board_width;
        int numMoves = probs.size();
        vector<Move> moves;
        for (int idx = 0; idx < numMoves; idx++)
        {
            moves.push_back(Move(Point((int)(idx / boardSize), idx % boardSize)));
        }
        moves.push_back(Move.pass_turn());
        vector<Move>::const_iterator temp_move = moves.begin();
        vector<float>::const_iterator temp_prob = probs.begin();
        vector<Move> legal_moves = this->state->legal_moves();
        while (temp_move!=moves.end())
        {
            if (find(legal_moves.begin(), legal_moves.end(), temp_move) != legal_moves.end()) {
                GoBoard next_board = *(this->state->board);
                if (temp_move.is_play){
                    next_board.place_stone(this->state->next_player, temp_move.point)
                }
                /*else {
                    GoBoard next_board = this->state->board;
                }*/
                GameState childState = GameState(next_board, *(this->state->next_player->opp), *(this->state), temp_move, *(this->state->moves));
                MCTSNode child = MCTSNode(&childState, temp_prob, 0.0, &temp_move, this);
                this->childNodes.push_back(child);
            }
            temp_move++;
            temp_prob++;
        }
    }

    void update(float v)
    {
        this->visits++;
        this->wins++;
        this->q = (this->wins / this->visits);
    }

    float puct(int c=4)
    {
        int N = 0;
        for (vector<MCTSNode>::const_iterator child = this->childNodes.begin(); child != this->childNodes.end(); ++child) {
            N += child.visits;
        }
        puc = this->q + ((c * this->p * sqrt(N)) / (1 + this->visits));
        return puc;
    }

    
};
class MCTSPlayer
{
public:
    Player *player;
    //Model model;

    MCTSPlayer(Player *player/*,Model model*/)
    {
        this->player = player;
        //this->model = model;
    }

    
    vector<MCTSNode> select_move(MCTSNode *rootnode,set<MCTSNode> visited,
        int simulations,float epsilon=0.25,float dcoeff=0.03,int c=4,bool stoch=True)
    {
        //Model nn = this->model;
        for (int i = 0; i < simulations; i++)
        {
            MCTSNode currNode = *rootnode;
            if ((stoch == true) && (i > 0)) {
                for (vector<MCTSNode>::const_iterator child = currNode.childNodes.begin(); child != currNode.childNodes.end(); ++child)
                {
                    if (stoch == true)
                    {
                        //child.p = (1 - epsilon) * child.p + epsilon * np.random.dirichlet(alpha = dcoeff)
                    }
                }
            }
            while (visited.find(currNode)!=visited.end())
            {
                currNode = currNode.select(c);
            }
            Player hero = currNode.state->next_player;
            if (visited.find(currNode) == visited.end())
            {
                visited.insert(currNode);
                Player hero = currNode.state->next_player;
                //TO BE DONE
            }
            while (currNode!=NULL)
            {
                float val;
                if (hero == currNode.state->next_player)
                {
                    val = v;
                }
                else
                {
                    val = -v;
                }
                currNode.update(val);
                currNode = currNode.parentNode;
            }
            if (stoch == true)
            {
                return rootnode->childNodes;
            }
            else
            {
                MCTSNode maxNode = rootnode->childNodes[0];
                for (vector<MCTSNode>::const_iterator i = rootnode->childNodes.begin(); i != rootnode->childNodes.end(); ++i)
                {
                    if (maxNode.visits<*(i.visits))
                    {
                        maxNode = *i;
                    }
                }
                return maxNode;
            }
        }
    }


};

class MCTSSelfPlay
{
public:
    int board_size;
    int plane_size;
    model_file
    network
    encoder

MCTSSelfPlay(int board_size,int plane_size,model_file=NULL,network=None)
{
    this->board_size = board_size;
    this->plane_size = plane_size;
    this->model_file = model_file;
    this->network = network;
    
}

void play(Network agent1,Network agent2,File expFile,int num_games=2500,int simulations=200,int c=4,float vResion=-0.7,tempMoves=4)
{

}

void save_moves(vector<Move> moves,int winner)
{

}


};

class ExperienceBuffer
{
public:
    vector<int[]> model_input = NULL;
    vector<float[]> action_target = NULL;
    vector<int[]> value_target = NULL;

    ExperienceBuffer(vector<int[]> model_input, vector<float[]> action target, vector<int[]> value_target)
    {
        model_input = model_input;
        action_target = action_target;
        value_target = value_target;
    }

    void serialize(File h5file)
    {

    }
};

ExperienceBuffer combine_experience(collectors)
{

}

ExperienceBuffer load_experience(File h5file)
{
    
}