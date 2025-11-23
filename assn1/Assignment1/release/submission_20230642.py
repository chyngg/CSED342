## ID: 20230642 NAME: Lee, Chaeyeong
######################################################################################
# Problem 2a
# minimax value of the root node: 5
# pruned edges (in order): 5, 5, 5, 5, 6, 0, 0, 7, 6, 6, 6, 7, 2, 2, 3, 4, 4, 1, 4, 1, 4, 6, 4, 6, 2, 2
######################################################################################

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
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    legalMoves = gameState.getLegalActions()

    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
    bestVal, bestAction = float("-inf"), Directions.STOP
    actions = gameState.getLegalActions(0)
    if not actions:
      return Directions.STOP

    for a in actions:
      succ = gameState.generateSuccessor(0, a)
      v = self.val(succ, agentIndex=1, depth=0)
      if v > bestVal:
        bestVal, bestAction = v, a
    return bestAction

  def val(self, state, agentIndex, depth):
    if depth == self.depth or state.isWin() or state.isLose():
      return self.evaluationFunction(state)

    if agentIndex == 0:
      return self.max_val(state, depth)
    else:
      return self.min_val(state, agentIndex, depth)

  def max_val(self, state, depth):
    v = float("-inf")
    actions = state.getLegalActions(0)
    if not actions:
      return self.evaluationFunction(state)

    for a in actions:
      succ = state.generateSuccessor(0, a)
      v = max(v, self.val(succ, agentIndex=1, depth=depth))
    return v

  def min_val(self, state, agentIndex, depth):
    v = float("inf")
    actions = state.getLegalActions(agentIndex)
    if not actions:
      return self.evaluationFunction(state)

    lastGhost = (agentIndex == state.getNumAgents() - 1)
    for a in actions:
      succ = state.generateSuccessor(agentIndex, a)
      nextAgent = 0 if lastGhost else agentIndex + 1
      nextDepth = depth + 1 if lastGhost else depth
      v = min(v, self.val(succ, agentIndex=nextAgent, depth=nextDepth))
    return v
    # END_YOUR_ANSWER

######################################################################################
# Problem 2b: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    alpha, beta = float("-inf"), float("inf")
    bestVal, bestAction = float("-inf"), Directions.STOP

    actions = gameState.getLegalActions(0)
    if not actions:
      return Directions.STOP

    for a in actions:
      succ = gameState.generateSuccessor(0, a)
      v = self.val(succ, agentIndex=1, depth=0, alpha=alpha, beta=beta)
      if v > bestVal:
        bestVal, bestAction = v, a
      alpha = max(alpha, bestVal)
    return bestAction


  def val(self, state, agentIndex, depth, alpha, beta):
    if depth == self.depth or state.isWin() or state.isLose():
      return self.evaluationFunction(state)

    if agentIndex == 0:
      return self.max_val(state, depth, alpha, beta)
    else:
      return self.min_val(state, agentIndex, depth, alpha, beta)


  def max_val(self, state, depth, alpha, beta):
    v = float("-inf")
    actions = state.getLegalActions(0)
    if not actions:
      return self.evaluationFunction(state)

    for a in actions:
      succ = state.generateSuccessor(0, a)
      v = max(v, self.val(succ, agentIndex=1, depth=depth, alpha=alpha, beta=beta))
      if v >= beta:
        return v
      alpha = max(alpha, v)
    return v

  def min_val(self, state, agentIndex, depth, alpha, beta):
    v = float("inf")
    actions = state.getLegalActions(agentIndex)
    if not actions:
      return self.evaluationFunction(state)

    lastGhost = (agentIndex == state.getNumAgents() - 1)
    for a in actions:
      succ = state.generateSuccessor(agentIndex, a)
      nextAgent = 0 if lastGhost else agentIndex + 1
      nextDepth = depth + 1 if lastGhost else depth

      v = min(v, self.val(succ, agentIndex=nextAgent, depth=nextDepth,
                             alpha=alpha, beta=beta))
      if v <= alpha:
        return v
      beta = min(beta, v)
    return v
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
    bestVal, bestAction = float("-inf"), Directions.STOP
    actions = gameState.getLegalActions(0)
    if not actions:
      return Directions.STOP

    for a in actions:
      succ = gameState.generateSuccessor(0, a)
      v = self._value(succ, agentIndex=1, depth=0)
      if v > bestVal:
        bestVal, bestAction = v, a
    return bestAction

  def _value(self, state, agentIndex, depth):
    if depth == self.depth or state.isWin() or state.isLose():
      return self.evaluationFunction(state)

    if agentIndex == 0:
      return self._max_value(state, depth)
    else:
      return self._exp_value(state, agentIndex, depth)

  def _max_value(self, state, depth):
    v = float("-inf")
    actions = state.getLegalActions(0)
    if not actions:
      return self.evaluationFunction(state)

    for a in actions:
      succ = state.generateSuccessor(0, a)
      v = max(v, self._value(succ, agentIndex=1, depth=depth))
    return v

  def _exp_value(self, state, agentIndex, depth):
    actions = state.getLegalActions(agentIndex)
    if not actions:
      return self.evaluationFunction(state)

    lastGhost = (agentIndex == state.getNumAgents() - 1)
    p = 1.0 / float(len(actions))
    expected = 0.0
    for a in actions:
      succ = state.generateSuccessor(agentIndex, a)
      nextAgent = 0 if lastGhost else agentIndex + 1
      nextDepth = depth + 1 if lastGhost else depth
      expected += p * self._value(succ, agentIndex=nextAgent, depth=nextDepth)
    return expected

  # END_YOUR_ANSWER

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  pos = currentGameState.getPacmanPosition()
  food = currentGameState.getFood().asList()
  capsules = currentGameState.getCapsules()
  ghosts = currentGameState.getGhostStates()
  score = currentGameState.getScore()

  if currentGameState.isWin():
    return 1e9
  if currentGameState.isLose():
    return -1e9

  if food:
    dists = sorted(manhattanDistance(pos, f) for f in food)
    K = 3
    pull = sum(2.8 / (d + 1.0) for d in dists[:K])
  else:
    pull = 8.0
  food_left_pen = 1.6 * len(food)

  if capsules:
    cap_min = min(manhattanDistance(pos, c) for c in capsules)
    cap_pull = 2.2 / (cap_min + 1.0)
  else:
    cap_min = None
    cap_pull = 0.0
  cap_left_pen = 1.0 * len(capsules)

  nearest_active = float('inf')
  risk = 0.0
  hunt = 0.0
  for g in ghosts:
    gd = manhattanDistance(pos, g.getPosition())
    if g.scaredTimer > 0:
      slack = g.scaredTimer - gd
      if slack >= 0:
        hunt += 12.0 + 2.0 * slack
      else:
        hunt += 1.5 / (gd + 1.0)
    else:
      nearest_active = min(nearest_active, gd)

  if nearest_active != float('inf'):
    d = nearest_active
    if d <= 1:
      risk += 700.0
    elif d == 2:
      risk += 110.0
    elif d == 3:
      risk += 35.0
    risk += 14.0 / ((d + 1.0) * (d + 1.0))

    danger_boost = max(0, 4 - min(d, 4))
    cap_pull *= (1.4 + 0.4 * danger_boost)
    pull *= (1.0 - 0.08 * danger_boost)

  value = score + pull + cap_pull + hunt - risk - food_left_pen - cap_left_pen
  return value

  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction

