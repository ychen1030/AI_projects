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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        if action == Directions.STOP:
            score -= 50

        for i in range(0, len(newGhostStates)):
            if newScaredTimes[i] == 0 and manhattanDistance(newGhostPos[i], newPos) <= 1:
                score -= 1000

        score += 100/findClosestFood(currentGameState, newPos)
        return score

def findClosestFood(currentGameState, pac):
    dis = 100000
    for pos in currentGameState.getFood().asList():
        temp = manhattanDistance(pos, pac)
        if temp < dis:
            dis = temp

    if dis == 0:
        return 0.5

    return dis

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        legalMoves = gameState.getLegalActions()
        scores = [self.minimax(gameState.generateSuccessor(0, action),
                self.depth, 1, gameState.getNumAgents()) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        return legalMoves[random.choice(bestIndices)]

    def minimax(self, gameState, depth, agentIndex, agentTotal):

        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            return max([self.minimax(gameState.generateSuccessor(0, action),
                    depth, agentIndex + 1, agentTotal) for action in legalMoves])
        elif agentIndex == agentTotal - 1:
            return min([self.minimax(gameState.generateSuccessor(agentIndex, action),
                    depth - 1, 0, agentTotal) for action in legalMoves])
        else:
            return min([self.minimax(gameState.generateSuccessor(agentIndex, action),
                    depth, agentIndex + 1, agentTotal) for action in legalMoves])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        beta = float("inf")
        bestAction = None
        legalMoves = gameState.getLegalActions()

        for action in legalMoves:
            child = self.alphabeta(gameState.generateSuccessor(0, action), self.depth,
                                   1, gameState.getNumAgents(), alpha, beta)
            if child > alpha:
                alpha = child
                bestAction = action
        return bestAction


    def alphabeta(self, gameState, depth, agentIndex, agentTotal, a, b):
        """
            Minimax search with alpha-beta pruning.
        """
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            v = float("-inf")
            for action in legalMoves:
                v = max(v, self.alphabeta(gameState.generateSuccessor(0, action),
                                    depth, agentIndex + 1, agentTotal, a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v
        else:
            v = float("inf")
            for action in legalMoves:
                if agentIndex == agentTotal - 1:
                    v = min(v, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                                                depth - 1, 0, agentTotal, a, b))
                else:
                    v = min(v, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                                                depth, agentIndex + 1, agentTotal, a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        scores = [self.expectimax(gameState.generateSuccessor(0, action),
                               self.depth, 1, gameState.getNumAgents()) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        return legalMoves[random.choice(bestIndices)]

    def expectimax(self, gameState, depth, agentIndex, agentTotal):

        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            return max([self.expectimax(gameState.generateSuccessor(0, action),
                    depth, agentIndex + 1, agentTotal) for action in legalMoves])
        elif agentIndex == agentTotal - 1:
            scores = [self.expectimax(gameState.generateSuccessor(agentIndex, action),
                    depth - 1, 0, agentTotal) for action in legalMoves]
            return sum(scores)/float(len(scores))
        else:
            scores = [self.expectimax(gameState.generateSuccessor(agentIndex, action),
                    depth, agentIndex + 1, agentTotal) for action in legalMoves]
            return sum(scores)/float(len(scores))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    numFood = currentGameState.getNumFood()
    allCap = currentGameState.getCapsules()
    score = 0
    foodNumWeight = -1000.0
    foodDisWeight = -50.0

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    # score += foodNumWeight*numFood + foodDisWeight*closestFood(currentGameState, pos)
    score += foodNumWeight * numFood + foodDisWeight * findClosestFood(currentGameState, pos)
    score += -1000.0*len(allCap)

    if ghostStates[0].scaredTimer != 0:
        ghostPos = currentGameState.getGhostPosition(1)
        if manhattanDistance(ghostPos, pos) == 0:
            score += 20000.0

    return score

# def closestFood(currentGameState, pos):
#     dis = float("inf")
#     for x in currentGameState.getFood().asList():
#         temp = searchAgents.mazeDistance(x, pos, currentGameState)
#         dis = min(dis, temp)
#         if dis == 0:
#             return 0.5
#
#     return dis




# Abbreviation
better = betterEvaluationFunction

