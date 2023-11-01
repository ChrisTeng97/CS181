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
        # Collect legal moves and child states
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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostPos = [ghost.getPosition() for ghost in newGhostStates 
                    if ghost.scaredTimer == 0]
        if len(ghostPos) > 0:
            minGhostDist = min([util.manhattanDistance(newPos, ghost) for ghost in ghostPos])
        else:
            minGhostDist = 1e6
        # Lose the game
        if minGhostDist == 0:
            return -200.0
        # Eat the food
        if newPos in currentGameState.getFood().asList():
            return 200.0
        minFoodDist = min([util.manhattanDistance(newPos, food) for food in newFood.asList()])
        # Calculate the score
        loss = 0
        if minGhostDist <= 5:
                loss = 100 / minGhostDist
        profit = 100 / minFoodDist
        return profit - loss

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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        scores = [self.minimax(gameState.getNextState(0, action), 0, 1) 
                    for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) 
                        if scores[index] == bestScore]
        return legalMoves[random.choice(bestIndices)]

    # Minimax algorithm, return the score of the state
    def minimax(self, gameState: GameState, depth: int, agentIndex: int):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, depth)
        else:
            return self.minValue(gameState, depth, agentIndex)

    # For Pacman, return the max score of the state
    def maxValue(self, gameState: GameState, depth: int):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = -1e9
        for action in gameState.getLegalActions(0):
            v = max(v, self.minimax(gameState.getNextState(0, action), depth, 1))
        return v

    # For ghosts, return the min score of the state
    def minValue(self, gameState: GameState, depth: int, agentIndex: int):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = 1e9
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = min(v, self.minimax(gameState.getNextState(agentIndex, action), depth + 1, 0))
            else:
                v = min(v, self.minimax(gameState.getNextState(agentIndex, action), depth, agentIndex + 1))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        scores = [self.minimax(gameState.getNextState(0, action), 0, 1, 
                    alpha = -1e9, beta = 1e9) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) 
                        if scores[index] == bestScore]
        return legalMoves[random.choice(bestIndices)]

    # Minimax algorithm with alpha-beta pruning, return the score of the state
    def minimax(self, gameState: GameState, depth: int, agentIndex: int, 
                alpha: float = -1e9, beta: float = 1e9):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex, alpha, beta)
        else:
            return self.minValue(gameState, depth, agentIndex, alpha, beta)

    # For Pacman, return the max score of the state
    def maxValue(self, gameState: GameState, depth: int, agentIndex: int, 
                alpha: float = -1e9, beta: float = 1e9):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = -1e9
        for action in gameState.getLegalActions(0):
            v = max(v, self.minimax(gameState.getNextState(0, action), 
                    depth, 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    # For ghosts, return the min score of the state
    def minValue(self, gameState: GameState, depth: int, agentIndex: int, 
                alpha: float = -1e9, beta: float = 1e9):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = 1e9
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = min(v, self.minimax(gameState.getNextState(agentIndex, action), 
                        depth + 1, 0, alpha, beta))
            else:
                v = min(v, self.minimax(gameState.getNextState(agentIndex, action), 
                        depth, agentIndex + 1, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

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
        legalMoves = gameState.getLegalActions()
        scores = [self.expectimax(gameState.getNextState(0, action), 0, 1)
                    for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores))
                        if scores[index] == bestScore]
        return legalMoves[random.choice(bestIndices)]

    # Expectimax algorithm, return the score of the state
    def expectimax(self, gameState: GameState, depth: int, agentIndex: int):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, depth)
        else:
            return self.expValue(gameState, depth, agentIndex)

    # For Pacman, return the max score of the state
    def maxValue(self, gameState: GameState, depth: int):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = -1e9
        for action in gameState.getLegalActions(0):
            v = max(v, self.expectimax(gameState.getNextState(0, action), depth, 1))
        return v

    # For ghosts, return the expected score of the state
    def expValue(self, gameState: GameState, depth: int, agentIndex: int):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = 0
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            if agentIndex == gameState.getNumAgents() - 1:
                v += self.expectimax(gameState.getNextState(agentIndex, action), 
                        depth + 1, 0)
            else:
                v += self.expectimax(gameState.getNextState(agentIndex, action), 
                        depth, agentIndex + 1)
        return v / len(legalActions)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 1
    if currentGameState.isLose():
        return -1
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newCapsules = []
    if currentGameState.getCapsules() != []:
        newCapsules = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
    newGhostPos = [ghost.getPosition() for ghost in newGhostStates 
                    if ghost.scaredTimer == 0]
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    loss = 0
    profit = 0
    if len(newGhostPos) > 0:
        minGhostDist = min([util.manhattanDistance(newPos, ghost) for ghost in newGhostPos])
    else:
        minGhostDist = 1e6
    # Lose the game
    if minGhostDist == 0:
        return -1
    # Eat the food
    if newPos in newFood:
        return 1
    minFoodDist = min([util.manhattanDistance(newPos, food) for food in newFood])
    if newCapsules != []:
        minCapsuleDist = min([util.manhattanDistance(newPos, capsule) for capsule in newCapsules])
        if minCapsuleDist < 5:
            profit += 150 / minCapsuleDist
    # Calculate the score
    if minGhostDist <= 5:
            loss += 100 / minGhostDist
    profit += 100 / minFoodDist
    return profit - loss

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState: GameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()