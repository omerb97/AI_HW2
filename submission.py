import random
import threading
import numpy as np

import Gobblet_Gobblers_Env as gge

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
            return (heuristic_wrapper(curr_state, agent_id)+L, None)
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
            return (heuristic_wrapper(curr_state, agent_id)+L, None)
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
        

def arrayToInt(arrayForm):
    return arrayForm[0]*3 +arrayForm[1]

class RB_Expectimax:
    def __init__(self,curr_state,agent_id):
        self.curr_state = curr_state
        self.agent_id = agent_id
        self.bestMove = None
    
    def Expectimax(self, event):
        bestMoveHeuristic = -math.inf
        L=1
        while True:
            print("depth is: " + str(L))
            self.bestMove = self.rb_expectimax_L(self.curr_state,self.agent_id,L)
            L+=1
            if event.is_set():
                break

    def rb_expectimax_L(self, curr_state, agent_id,L):
        if gge.is_final_state(curr_state) or L == 0:
            return (heuristic_wrapper(curr_state, agent_id)+L, None)
        turnFlag = 1
        if agent_id == curr_state.turn:
            turnFlag = 0
        neighbors= curr_state.get_neighbors()
        if turnFlag == 0:
            curMax = -math.inf 
            curMaxChild = None
            for child in neighbors:
                v = self.rb_expectimax_L(child[1],agent_id,L-1)[0]
                if v>curMax:
                    curMax = v
                    curMaxChild = child
            return (curMax, curMaxChild[0])
        else:
            curExp = 0
            probArr = self.Probability_Func(curr_state)
            for item in probArr:
                curExp += item[2] * self.rb_expectimax_L(probArr[1], agent_id, L-1)[0]
            return (curExp,neighbors[0])
    


    def Probability_Func(self, state):
        positions = [0]*9
        for pawn, pos in state.player1_pawns.items():
            if pos[0] != not_on_board:
                positions[arrayToInt(pos[0])] = 1 
        for pawn, pos in state.player2_pawns.items():
            if pos[0] != not_on_board:
                positions[arrayToInt(pos[0])] = 1  
        possible_neighbors = state.get_neighbors(state)
        smallCount = 0
        eatCount = 0
        for neighbor in possible_neighbors:

            action = neighbor[0]
            newState = neighbor[1]
            if action[0] == "S1" or action[0] == "S2":
                smallCount += 1
            if positions[arrayToInt(action[1])] == 1:
                eatCount += 1 

        prob = len(possible_neighbors) + smallCount + eatCount
        new_neighbors = []
        for neighbor in possible_neighbors:
            action = neighbor[0]
            newState = neighbor[1]
            if action[0] == "S1" or action[0] == "S2":
                new_neighbors.append((action,newState, 2/prob))
            elif positions[arrayToInt(action[1])] == 1:
                new_neighbors.append((action,newState, 2/prob))
            else:
                new_neighbors.append((action,newState,1/prob))
        return new_neighbors

    


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

def IsCovered(pieceSize, place, state):
    NOT_COVERED = 1
    M_COVERED = 0.5
    S_COVERED_ONCE = 2/3
    S_COVERED_TWICE = 1/3
    player1 = state.player1_pawns
    player2 = state.player2_pawns
    if pieceSize == "B":
        return NOT_COVERED
    if pieceSize == "M":
        for key, value in player1.items():
            if not np.array_equal(value[0], not_on_board):
                if  np.array_equal(value[0],place) and value[1] == "B":
                    return M_COVERED
        for key, value in player2.items():
            if not np.array_equal(value[0], not_on_board):
                if  np.array_equal(value[0],place) and value[1] == "B":
                    return M_COVERED
        return NOT_COVERED
    
    if pieceSize == "S":
        counter = 0
        for key, value in player1.items():
            if not np.array_equal(value[0], not_on_board):
                if  np.array_equal(value[0],place) and value[1] == "B" or value[1] == "M":
                    counter += 1
        for key, value in player2.items():
            if not np.array_equal(value[0], not_on_board):
                if  np.array_equal(value[0],place) and value[1] == "B" or value[1] == "M":
                    counter += 1
        if counter == 0:
            return NOT_COVERED
        if counter == 1:
            return S_COVERED_ONCE
        if counter == 2:
            return S_COVERED_TWICE
    return NOT_COVERED

        
def smart_heuristic(state, agent_id):
    VALUE_PER_PIECE = {
        "B":9,
        "M":3,
        "S":1
    }
    utility = 0
    player1 = state.player1_pawns
    for key, value in player1.items():
        if not np.array_equal(value[0], not_on_board):
            if  np.array_equal(value[0],np.array([1,1])):
                utility += ((4 * VALUE_PER_PIECE[value[1]]) * IsCovered(value[1],value[0], state))
            elif np.array_equal(value[0],np.array([0,0])) or np.array_equal(value[0],np.array([0,2])) or np.array_equal(value[0],np.array([2,0])) or np.array_equal(value[0],np.array([2,2])):
                utility += ((3 * VALUE_PER_PIECE[value[1]]) * IsCovered(value[1],value[0], state))
            else:
                utility += ((2 * VALUE_PER_PIECE[value[1]]) * IsCovered(value[1],value[0], state))

    player2 = state.player2_pawns
    for key,value in player2.items():
        if not np.array_equal(value[0], not_on_board):
            if  np.array_equal(value[0],np.array([1,1])):
                utility -= ((4 * VALUE_PER_PIECE[value[1]]) * IsCovered(value[1],value[0], state))
            elif np.array_equal(value[0],np.array([0,0])) or np.array_equal(value[0],np.array([0,2])) or np.array_equal(value[0],np.array([2,0])) or np.array_equal(value[0],np.array([2,2])):
                utility -= ((3 * VALUE_PER_PIECE[value[1]]) * IsCovered(value[1],value[0], state))
            else:
                utility -= ((2 * VALUE_PER_PIECE[value[1]])* IsCovered(value[1],value[0], state))
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
        
def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    rb_minimax = RB_Mini_Max(curr_state=curr_state, agent_id=agent_id)
    event = threading.Event()

    rb_minimax_thread = threading.Thread (target=rb_minimax.mini_max, args= (event,))
    rb_minimax_thread.start()
    rb_minimax_thread.join(timeout=time_limit-2)
    event.set()
    #rb_minimax_thread.stop()
    return rb_minimax.bestMove[1]

def alpha_beta(curr_state, agent_id, time_limit):
    rb_alpha_beta = RB_Alpha_Beta(curr_state=curr_state, agent_id=agent_id)
    event = threading.Event()

    rb_alpha_beta_thread = threading.Thread (target=rb_alpha_beta.alpha_beta, args= (event,))
    rb_alpha_beta_thread.start()
    rb_alpha_beta_thread.join(timeout=time_limit-2)
    event.set()
    #rb_minimax_thread.stop()
    return rb_alpha_beta.bestMove[1]


def expectimax(curr_state, agent_id, time_limit):
    rb_expectimax = RB_Expectimax(curr_state=curr_state, agent_id=agent_id)
    event = threading.Event()

    rb_expectimax_thread = threading.Thread (target=rb_expectimax.Expectimax, args= (event,))
    rb_expectimax_thread.start()
    rb_expectimax_thread.join(timeout=time_limit-2)
    event.set()
    #rb_minimax_thread.stop()
    return rb_expectimax.bestMove[1] 

# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    raise NotImplementedError()
