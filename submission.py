import random
import threading
import numpy as np

import Gobblet_Gobblers_Env as gge

import time
import math

not_on_board = np.array([-1, -1])
WON = 420
LOSE = -420

class RB_Mini_Max:
    def __init__(self,curr_state,agent_id):
        self.curr_state = curr_state
        self.agent_id = agent_id
        self.bestMove = None
    
    def mini_max(self, event):
        bestMoveHeuristic = -math.inf
        L=1
        while True:
            print("depth is: " + str(L))
            self.bestMove = self.rb_heuristic_min_max_L(self.curr_state,self.agent_id,L)
            L+=1
            if event.is_set():
                break

    def rb_heuristic_min_max_L(self, curr_state, agent_id,L):
        if gge.is_final_state(curr_state) or L == 0:
            return (heuristic_wrapper(curr_state, agent_id), None)
        turnFlag = 1
        if agent_id == curr_state.turn:
            turnFlag = 0
        neighbors= curr_state.get_neighbors()
        if turnFlag == 0:
            curMax = -math.inf 
            curMaxChild = None
            for child in neighbors:
                v = self.rb_heuristic_min_max_L(child[1],agent_id,L-1)[0]
                if v>curMax:
                    curMax = v
                    curMaxChild = child
            return (curMax, curMaxChild[0])
        else:
            curMin = math.inf 
            curMinChild = None
            for child in neighbors:
                v = self.rb_heuristic_min_max_L(child[1],agent_id,L-1)[0]
                if v<curMin:
                    curMin = v
                    curMinChild = child
            return (curMin, curMinChild[0])


class RB_Alpha_Beta:
    def __init__(self,curr_state,agent_id):
        self.curr_state = curr_state
        self.agent_id = agent_id
        self.bestMove = None
    
    def alpha_beta(self, event):
        bestMoveHeuristic = -math.inf
        L=1
        while True:
            print("depth is: " + str(L))
            self.bestMove = self.rb_heuristic_alpha_beta_L(self.curr_state,self.agent_id,L,-math.inf,math.inf)
            L+=1
            if event.is_set():
                break

    def rb_heuristic_alpha_beta_L(self, curr_state, agent_id,L, alpha,beta):
        if gge.is_final_state(curr_state) or L == 0:
            return (heuristic_wrapper(curr_state, agent_id), None)
        turnFlag = 1
        if agent_id == curr_state.turn:
            turnFlag = 0
        neighbors= curr_state.get_neighbors()
        if turnFlag == 0:
            curMax = -math.inf 
            curMaxChild = None
            for child in neighbors:
                v = self.rb_heuristic_alpha_beta_L(child[1],agent_id,L-1,alpha,beta)[0]
                alpha = max(alpha, curMax)
                if curMax > beta:
                    return (math.inf, curMaxChild[0])
                if v>=curMax:
                    curMax = v
                    curMaxChild = child
            return (curMax, curMaxChild[0])
        else:
            curMin = math.inf 
            curMinChild = None
            for child in neighbors:
                v = self.rb_heuristic_alpha_beta_L(child[1],agent_id,L-1,alpha,beta)[0]
                beta = min(beta,curMin )
                if curMin < alpha:
                    return(-math.inf, curMinChild[0])
                if v<=curMin:
                    curMin = v
                    curMinChild = child
            return (curMin, curMinChild[0])
        
        


# agent_id is which player I am, 0 - for the first player , 1 - if second player
def dumb_heuristic1(state, agent_id):
    is_final = gge.is_final_state(state)
    # this means it is not a final state
    if is_final is None:
        return 0
    # this means it's a tie
    if is_final is 0:
        return -1
    # now convert to our numbers the win
    winner = int(is_final) - 1
    # now winner is 0 if first player won and 1 if second player won
    # and remember that agent_id is 0 if we are first player  and 1 if we are second player won
    if winner == agent_id:
        # if we won
        return 1
    else:
        # if other player won
        return -1


# checks if a pawn is under another pawn
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False


# count the numbers of pawns that i have that aren't hidden
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1

    return sum_pawns


def smart_heuristic(state, agent_id):
    VALUE_PER_PIECE = {
        "B":4,
        "M":2,
        "S":1
    }
    utility = 0
    player1 = state.player1_pawns
    for key, value in player1.items():
        if not np.array_equal(value[0], not_on_board):
            if  np.array_equal(value[0],np.array([1,1])):
                utility += (4 * VALUE_PER_PIECE[value[1]])
            elif np.array_equal(value[0],np.array([0,0])) or np.array_equal(value[0],np.array([0,2])) or np.array_equal(value[0],np.array([2,0])) or np.array_equal(value[0],np.array([2,2])):
                utility += (3 * VALUE_PER_PIECE[value[1]])
            else:
                utility += (2 * VALUE_PER_PIECE[value[1]])

    player2 = state.player2_pawns
    for key,value in player2.items():
        if not np.array_equal(value[0], not_on_board):
            if  np.array_equal(value[0],np.array([1,1])):
                utility -= (4 * VALUE_PER_PIECE[value[1]])
            elif np.array_equal(value[0],np.array([0,0])) or np.array_equal(value[0],np.array([0,2])) or np.array_equal(value[0],np.array([2,0])) or np.array_equal(value[0],np.array([2,2])):
                utility -= (3 * VALUE_PER_PIECE[value[1]])
            else:

                utility -= (2 * VALUE_PER_PIECE[value[1]])
    if agent_id == 1:
        return -1*utility
    else:
        return utility
                


# IMPLEMENTED FOR YOU - NO NEED TO CHANGE
def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


# agent_id is which agent you are - first player or second player
def random_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    rnd = random.randint(0, neighbor_list.__len__() - 1)
    return neighbor_list[rnd][0]


# TODO - instead of action to return check how to raise not_implemented
def greedy(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = dumb_heuristic2(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


# TODO - add your code here
def greedy_improved(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = -1000
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        #print(curr_heuristic)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]

def heuristic_wrapper(curr_state, agent_id):
    is_final = gge.is_final_state(curr_state)
    #print(is_final)
    if is_final is not None:
        if (int(is_final)-1) == agent_id:
            return WON
        else:
            return LOSE
    return smart_heuristic(curr_state,agent_id)
        
# def rb_heuristic_min_max_L(curr_state, agent_id, L):
#     if gge.is_final_state(curr_state) or L == 0:
#         return (heuristic_wrapper(curr_state, agent_id), None)
#     turnFlag = 1
#     if agent_id == curr_state.turn:
#         turnFlag = 0
#     neighbors= curr_state.get_neighbors()
#     if turnFlag == 0:
#         curMax = -math.inf 
#         curMaxChild = None
#         for child in neighbors:
#             v = rb_heuristic_min_max_L(child[1],agent_id,L-1)[0]
#             if v>curMax:
#                 curMax = v
#                 curMaxChild = child
#         return (curMax, curMaxChild[0])
#     else:
#         curMin = math.inf 
#         curMinChild = None
#         for child in neighbors:
#             v = rb_heuristic_min_max_L(child[1],agent_id,L-1)[0]
#             if v<curMin:
#                 curMin = v
#                 curMinChild = child
#         return (curMin, curMinChild[0])

# bestMove = None

# def mini_max(curr_state,agent_id):
#     bestMoveHeuristic = -math.inf
#     L=1
#     while True:
#         bestMove = rb_heuristic_min_max_L(curr_state,agent_id,L)
#         L+=1

def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    rb_minimax = RB_Mini_Max(curr_state=curr_state, agent_id=agent_id)
    event = threading.Event()

    rb_minimax_thread = threading.Thread (target=rb_minimax.mini_max, args= (event,))
    rb_minimax_thread.start()
    rb_minimax_thread.join(timeout=time_limit-0.2)
    event.set()
    #rb_minimax_thread.stop()
    return rb_minimax.bestMove[1]

def alpha_beta(curr_state, agent_id, time_limit):
    rb_alpha_beta = RB_Alpha_Beta(curr_state=curr_state, agent_id=agent_id)
    event = threading.Event()

    rb_alpha_beta_thread = threading.Thread (target=rb_alpha_beta.alpha_beta, args= (event,))
    rb_alpha_beta_thread.start()
    rb_alpha_beta_thread.join(timeout=time_limit-0.2)
    event.set()
    #rb_minimax_thread.stop()
    return rb_alpha_beta.bestMove[1]


def expectimax(curr_state, agent_id, time_limit):
    raise NotImplementedError()

# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    raise NotImplementedError()
