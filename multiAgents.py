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
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        #The evaluation function:
        # - It rewards the agent for being close to food and penalizes it for being close to non-scared ghosts.
        # - Additionally, it deducts points based on the amount of remaining food pellets.
        # - By considering these factors, the agent is encouraged to make safer and more strategic decisions, resulting in better gameplay
        
        score = successorGameState.getScore()

        # Distance to the nearest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            score += 1.0 / min(foodDistances)
        # Distance to the ghosts
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if ghostDistance < 2 and ghostState.scaredTimer == 0:
                score -= 100  # Penalize if too close to a ghost

        # Amount of remaining food
        score -= len(newFood.asList())

        return score
        #return successorGameState.getScore()


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
        #util.raiseNotDefined()
        def minimax(agentIndex, depth, gameState):
            # If we reach the maximum depth or a terminal state, return the evaluation of the state
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # Determine if current agent is Pacman (maximizing) or a ghost (minimizing)
            bestValue = float('-inf') if agentIndex == 0 else float('inf')
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth if nextAgent != 0 else depth + 1

            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                value = minimax(nextAgent, nextDepth, successorState)
                bestValue = max(bestValue, value) if agentIndex == 0 else min(bestValue, value)
                
            return bestValue

        # Find the best action for Pacman
        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successorState)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            # If we reach the maximum depth or a terminal state, return the evaluation of the state
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pacman (maximizing player)
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            else:  # Ghosts (minimizing players)
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alphaBeta(1, depth, successorState, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            v = float('inf')
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth if nextAgent != 0 else depth + 1

            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphaBeta(nextAgent, nextDepth, successorState, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        # Find the best action for Pacman
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            score = alphaBeta(1, 0, successorState, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction

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
        #util.raiseNotDefined()
        def expectimax(state, depth, agentIndex):
            if depth == self.depth * state.getNumAgents() or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn
                return max_value(state, depth, agentIndex)
            else:  # Ghosts' turn
                return exp_value(state, depth, agentIndex)

        def max_value(state, depth, agentIndex):
            v = float('-inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, expectimax(successor, depth + 1, (agentIndex + 1) % state.getNumAgents()))
            return v

        def exp_value(state, depth, agentIndex):
            """
            Computes the expected value for ghosts, assuming they move uniformly at random.
            """
            v = 0
            actions = state.getLegalActions(agentIndex)
            p = 1 / len(actions) # Probability of each action
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v += p * expectimax(successor, depth + 1, (agentIndex + 1) % state.getNumAgents())
            return v

        best_action = None
        best_value = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 1, 1)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    This evaluation function considers:
    - Current game score
    - Distance to the closest food pellet
    - Distance to the closest ghost (penalizing being too close to active ghosts)
    - Number of remaining food pellets (penalizing more remaining food)
    - Distance to the closest capsule (encouraging eating capsules to turn ghosts scared)
    - Distance to the closest scared ghost (encouraging eating scared ghosts)
    
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()

    evaluationScore = currentScore

    # Factor 1: Distance to the closest food pellet
    if foodDistances := [manhattanDistance(pacmanPos, foodPos) for foodPos in food.asList()]:
        minFoodDistance = min(foodDistances)
        evaluationScore += 20.0 / minFoodDistance  # Increased weight for closer food pellets

    # Factor 2: Distance to the closest ghost (higher penalty for closer distance to active ghosts)
    ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghosts]
    activeGhostDistances = [dist for ghost, dist in zip(ghosts, ghostDistances) if ghost.scaredTimer == 0]
    scaredGhostDistances = [dist for ghost, dist in zip(ghosts, ghostDistances) if ghost.scaredTimer > 0]

    if activeGhostDistances:
        minActiveGhostDistance = min(activeGhostDistances)
        if minActiveGhostDistance > 0:
            evaluationScore -= 15 / minActiveGhostDistance  # Increased penalty for being close to active ghosts

    # Factor 3: Distance to the closest capsule (to encourage eating capsules)
    if capsuleDistances := [manhattanDistance(pacmanPos, capPos) for capPos in capsules]:
        minCapsuleDistance = min(capsuleDistances)
        evaluationScore += 10 / minCapsuleDistance  # Encouragement for closer capsules

    # Factor 4: Distance to the closest scared ghost (higher reward for closer distance to scared ghosts)
    if scaredGhostDistances:
        minScaredGhostDistance = min(scaredGhostDistances)
        evaluationScore += 80.0 / minScaredGhostDistance  # Higher reward for being close to scared ghosts

    # Factor 5: Number of remaining food pellets (higher penalty for more remaining food)
    numFoodRemaining = len(food.asList())
    evaluationScore -= 4 * numFoodRemaining  # Increased penalty for more remaining food pellets

    return evaluationScore


# Abbreviation
better = betterEvaluationFunction
