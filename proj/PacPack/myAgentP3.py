# myAgentP3.py
# ---------
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
# This file was based on the starter code for student bots, and refined 
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#########
# Agent #
#########
class myAgentP3(CaptureAgent):
  """
  Students' Names: Yingying Chen, Chen Chen
  Phase Number: 3
  Description of Bot: Use feature representation.
  Reward for closest food, has food, and penalty for
  being too close to the ghost and being close to the
  other pacman.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    # Make sure you do not delete the following line. 
    # If you would like to use Manhattan distances instead 
    # of maze distances in order to save on initialization 
    # time, please take a look at:
    # CaptureAgent.registerInitialState in captureAgents.py.
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

  def chooseAction(self, gameState):
    """
    Picks among best actions.
    """
    if self.toBroadcast and len(self.toBroadcast) > 0:
      action = self.toBroadcast.pop(0)

      if action in gameState.getLegalActions(self.index):
        ghosts = [gameState.getAgentPosition(ghost) for ghost in gameState.getGhostTeamIndices()]
        pacman = gameState.getAgentPosition(self.index)
        closestGhost = min(self.distancer.getDistance(pacman, ghost) for ghost in ghosts) \
          if len(ghosts) > 0 else 1.0

        if closestGhost > 2:
          return action

    currentAction = self.chooseActionHelper(gameState)
    futureActions = self.generatePlan(gameState.generateSuccessor(self.index, currentAction), 3)

    self.toBroadcast = futureActions
    return currentAction

  def chooseActionHelper(self, state):
    actions = state.getLegalActions(self.index)
    filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), state, self.index)

    qValues = [self.evaluate(state, action) for action in filteredActions]
    return filteredActions[qValues.index(max(qValues))]

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights()
    return features * weights

  def getFeatures(self, state, action):
    nextState = state.generateSuccessor(self.index, action)
    newPos = nextState.getAgentPosition(self.index)
    foods = state.getFood().asList()
    features = util.Counter()

    if len(foods) == 0:
      features['isWin'] = 1
    else:
      ghosts = [state.getAgentPosition(ghost) for ghost in state.getGhostTeamIndices()]
      others = [state.getAgentPosition(other) for other in state.getPacmanTeamIndices() if other != self.index]

      closestFood = min(self.distancer.getDistance(newPos, food) for food in foods) + 1.0
      closestGhost = min(self.distancer.getDistance(newPos, ghost) for ghost in ghosts) + 1.0
      closestPac = min(self.distancer.getDistance(newPos, other) for other in others) + 1.0

      features['closestFood'] = closestFood
      features['hasFood'] = state.hasFood(newPos[0], newPos[1])
      features['teammateDistance'] = closestPac
      features['closeToGhost'] = 1 if closestGhost <= 2 else 0
    return features

  def getWeights(self):
    return {'closestFood': -500,
            'hasFood': 500000,
            'teammateDistance': 250,
            'closeToGhost': -200,
            'isWin': 999999}

  def generatePlan(self, state, plan_length):
    plan = []
    other = [other for other in state.getPacmanTeamIndices() if other != self.index]
    other = other[0]
    for i in range(plan_length):
      if self.receivedBroadcast and len(self.receivedBroadcast) > i:
        action = self.receivedBroadcast[i]
        if action in state.getLegalActions(other):
          state = state.generateSuccessor(other, action)

      action = self.chooseActionHelper(state)
      plan.append(action)
      state = state.generateSuccessor(self.index, action)
    return plan


def actionsWithoutStop(legalActions):
  """
  Filters actions by removing the STOP action
  """
  legalActions = list(legalActions)
  if Directions.STOP in legalActions:
    legalActions.remove(Directions.STOP)
  return legalActions

def actionsWithoutReverse(legalActions, gameState, agentIndex):
  """
  Filters actions by removing REVERSE, i.e. the opposite action to the previous one
  """
  legalActions = list(legalActions)
  reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
  if len (legalActions) > 1 and reverse in legalActions:
    legalActions.remove(reverse)
  return legalActions
