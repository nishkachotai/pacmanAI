# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (pos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        pos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        score = successorGameState.getScore()
        food_list_original = currentGameState.getFood().asList()
        posGhostsArr = []
        ghostDistancesArr = []

        for g in newGhostStates:
            g = g.getPosition()
            x, y = g[0], g[1]
            posGhostsArr.append((x,y))

        for g in posGhostsArr:
            MHdistance = manhattanDistance(g, pos)
            ghostDistancesArr.append(MHdistance)

        foodDistancesArr = []
        for food in newFood.asList(): 
            MHFood = manhattanDistance(food, pos)
            foodDistancesArr.append(MHFood)
    
        if pos in posGhostsArr:
            score = -1 #tell pacman to not go there
        elif pos in food_list_original:
            score = 1
        else:
            score = 1/min(foodDistancesArr)
            score -= 1/min(ghostDistancesArr)
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        #from lecture 9
        def max_val(state, d, index, action = None):
            #base cases
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), action)  
            if d == 0:
                return (self.evaluationFunction(state), action) 

            v = -1 * float('inf')
            actions = state.getLegalActions(index)
            best_action = action
            

            for a in actions:
                s = state.generateSuccessor(index, a)
                curr_val, curr_action = min_val(s, d, index+1, a)
                if curr_val > v:
                    v = curr_val
                    best_action = a
            
            return (v, best_action)
        
        #from lecture 9
        def min_val(state, d, index, action = None):
            
            #base cases
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), action)
            if d == 0:
                return (self.evaluationFunction(state), action) 

            v = float('inf')
            actions = state.getLegalActions(index)
            best_action = action

            for a in actions:
                s = state.generateSuccessor(index, a)
        
                if (index+1) % gameState.getNumAgents() == 0:
                    curr_val, curr_action = max_val(s, d-1, 0, a)
                    if curr_val < v:
                        v = curr_val
                        best_action = a
                else:
                    curr_val, curr_action = min_val(s, d, index+1, a)
                    if curr_val < v:
                        v = curr_val
                        best_action = a
            
            return (v, best_action)

        return max_val(gameState, self.depth, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_val(state, d, index, alpha, beta, action = None):
            #base cases
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), action)  
            if d == 0:
                return (self.evaluationFunction(state), action) 

            v = -1 * float('inf')
            actions = state.getLegalActions(index)
            best_action = action
            

            for a in actions:
                s = state.generateSuccessor(index, a)
                curr_val, curr_action = min_val(s, d, index+1, alpha, beta, a)
                if curr_val > v:
                    v = curr_val
                    best_action = a
                    if v > beta:
                        return (v, best_action)
                alpha = max(alpha, v)
            
            return (v, best_action)
        
        #from lecture 9
        def min_val(state, d, index, alpha, beta, action = None):
            
            #base cases
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), action)
            if d == 0:
                return (self.evaluationFunction(state), action) 

            v = float('inf')
            actions = state.getLegalActions(index)
            best_action = action

            for a in actions:
                s = state.generateSuccessor(index, a)
        
                if (index+1) % gameState.getNumAgents() == 0:
                    curr_val, curr_action = max_val(s, d-1, 0, alpha, beta, a)
                    if curr_val < v:
                        v = curr_val
                        best_action = a
                        if v < alpha:
                            return (v, a)
                    beta = min(v, beta)
                else:
                    curr_val, curr_action = min_val(s, d, index+1, alpha, beta, a)
                    if curr_val < v:
                        v = curr_val
                        best_action = a
                        if v < alpha:
                            return (v, a)
                    beta = min(v, beta)
            return (v, best_action)

        return max_val(gameState, self.depth, 0, -1*float('inf'), float('inf'))[1]
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_val(state, d, index, action = None):
            #base cases
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), action)  
            if d == 0:
                return (self.evaluationFunction(state), action) 

            v = -1 * float('inf')
            actions = state.getLegalActions(index)
            best_action = action
            

            for a in actions:
                s = state.generateSuccessor(index, a)
                curr_val, curr_action = expected_agent(s, d, index+1, a)
                if curr_val > v:
                    v = curr_val
                    best_action = a
            
            return (v, best_action)

        
        def expected_agent(state, d, index, action = None):
            #base cases
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), action)  
            if d == 0:
                return (self.evaluationFunction(state), action) 

            v = 0
            actions = state.getLegalActions(index)
            best_action = action

            for a in actions:
                s = state.generateSuccessor(index, a)
                #n = gameState.getNumAgents()
                prob = 1/len(actions)
                
                if (index+1) % gameState.getNumAgents() == 0:
                    curr_val, curr_action = max_val(s, d-1, 0, a)
                    v += prob * curr_val
                    best_action = a
                else:
                    curr_val, curr_action = expected_agent(s, d, index+1, a)
                    v += prob * curr_val
                    best_action = a

            return (v, best_action)

        return max_val(gameState, self.depth, 0)[1]
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
        # newFood_list = newFood.asList()
        # pos_x, pos_y = pos[0], pos[1]
        
        # feat_scared = sum(newScaredTimes)

        # close_ghosts = []

        # for g in newGhostStates:
        #     g = g.getPosition()
        #     x, y = g[0], g[1]

        #     if (abs(pos_x - x) + abs(pos_y - y)) <= 10:
        #         close_ghosts.append((x, y))

        # feat_close_ghosts = 1

        # if len(close_ghosts) != 0:
        #     feat_close_ghosts = 1/len(close_ghosts)

        # feat_food = 1
        # if len(newFood_list) != 0:
        #     feat_food = 1/len(newFood_list)

        # close_food = 0
        # for p in newFood_list:
        #     x, y = p[0], p[1]
        #     if abs(pos_x - x) <= 10:
        #         close_food += 1
        #     if abs(pos_y - y) <= 10:
        #         close_food += 1
        
        # return 2 * feat_food + feat_scared + 6 * feat_close_ghosts
    
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghost_states]
    score = currentGameState.getScore()
    
    foodList = food.asList()
    # tell whether a food gets eaten with each particular state
    score -= len(foodList)

    #append manhattan distance from each ghost to the score
    for ghost_state in ghost_states:
        dist = manhattanDistance(pos, ghost_state.getPosition())
        score += dist

    return score + 2*sum(newScaredTimes)

# Abbreviation
better = betterEvaluationFunction
