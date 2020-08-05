#ifndef MCTS_H
#define MCTS_H

#include <iostream>
#include <map>
#include <list>
#include <vector>
#include "../godomain.h"
#include "../gohelper.h"

using namespace std;
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

    MCTSNode(GameState *state,float p=0.0,float v=0.0,Move *move=NULL,MCTSNode *parentNode=NULL);
    MCTSNode select(int c=4);
    void expand(float probs[]);
    void update(float v);
    float puct(int c=4);

};

class MCTSPlayer
{
public:
    Player *player = NULL;

    MCTSPlayer(Player *player);
    vector<MCTSNode> select_move(MCTSNode *rootnode,set<MCTSNode> visited,int simulations,float epsilon=0.25,float dcoeff=0.03,int c=4,bool stoch=True);
};

class MCTSSelfPlay
{
public:
    int board_size;
    int plane_size;
    model_file
    network
    encoder

    MCTSSelfPlay(int board_size,int plane_size,model_file=NULL,network=None);
    void play(Network agent1,Network agent2,File expFile,int num_games=2500,int simulations=200,int c=4,float vResion=-0.7,tempMoves=4);
    void save_moves(vector<Move> moves,int winner);
};

class ExperienceBuffer
{
public:
    vector<int[]> model_input;
    vector<float[]> action_target;
    vector<int[]> value_target;

    ExperienceBuffer(vector<int[]> model_input, vector<float[]> action target, vector<int[]> value_target);
    void serialize(File h5file);
};

ExperienceBuffer combine_experience(collectors);
ExperienceBuffer load_experience(File h5file);