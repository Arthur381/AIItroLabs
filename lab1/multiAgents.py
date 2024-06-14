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
# Revised by TAs from Intro to AI class (2024 Spring) of PKU.


from util import manhattanDistance
from game import Directions
import random, util
from math import sqrt, log
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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]#回归最高分对应的节点
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)#多个最好的，随机选一个

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]#返回一个action

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        #the freeze time is finite if pacman eats a power pellet

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #get successor
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        # much food in an aslist
        newGhostStates = successorGameState.getGhostStates()
        #ghosts' position and others
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          GameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          GameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          GameState.getNumAgents():
            Returns the total number of agents in the game
          GameState.isWin(), GameState.isLose():
            Returns whether or not the game state is a terminal state
        """
        def maximizer(state, depth, index_of_agent):
            maxiAction = None #agent need to get max
            # condition for termination of recursive method calls
            def terminal_condition(state,depth):
                "*** YOUR CODE HERE ***"
                if state.isLose() == True or depth==0 or state.isWin()==True:
                    return True
                return False
            if terminal_condition(state,depth) == True:
                return (self.evaluationFunction(state), maxiAction)#初始化时action为None
            # initialize value
            "*** YOUR CODE HERE ***"

            legal_actions=state.getLegalActions(index_of_agent)#得到这个 agent 的所有可能的行动
            assert len(legal_actions)!=0, "Wrong"
            scores = [minimizer(state.generateSuccessor(index_of_agent, Act), depth, (index_of_agent + 1))[0] for Act in
                      legal_actions]
            maxScore=max(scores)
            max_index = [index for index in range(len(legal_actions)) if scores[index] == maxScore]
            maxiAction = legal_actions[max_index[0]]
            value = maxScore
            #就是该节点的值
            # for every legal action, update value and maxiAction
            "*** YOUR CODE HERE ***"
            return (value, maxiAction)

        def minimizer(state, depth, index_of_agent):#返回元组
            miniAction = None
            def terminal_condition(state,depth):
                "*** YOUR CODE HERE ***"
                if state.isWin()==True or depth==0 or state.isLose()==True:
                    return True
                return False
            if terminal_condition(state,depth) == True:
                return (self.evaluationFunction(state), miniAction)
            # initialize value
            "*** YOUR CODE HERE ***"
            legal_actions=state.getLegalActions(index_of_agent)
            if index_of_agent==state.getNumAgents()-1:#此时到了最后一个ghost
                scores = [maximizer(state.generateSuccessor(index_of_agent, Act),(depth-1),0)[0] for Act in legal_actions]
            else:#不是最后一个ghost，那么肯定还有一层minimizer
                scores=[minimizer(state.generateSuccessor(index_of_agent, Act),depth,index_of_agent+1)[0] for Act in legal_actions]
            minScore=min(scores)
            # for every legal action, update value and miniAction
            "*** YOUR CODE HERE ***"
            min_index=[index for index in range(len(legal_actions)) if scores[index]==minScore]
            miniAction=legal_actions[min_index[0]]
            value=minScore
            return (value, miniAction)
        action = maximizer(gameState, self.depth, 0)[1]#pacman is zero forever
        return action 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maximizer(state, depth, index_of_agent, alpha, beta):
            maxiAction = None
            #condition for termination of recursive method calls
            def terminal_condition(state,depth):
                "*** YOUR CODE HERE ***"
                return state.isLose() == True or depth==0 or state.isWin()==True
            if terminal_condition(state,depth) == True:
                return (self.evaluationFunction(state), maxiAction)
            # initialize value
            legal_actions=state.getLegalActions(index_of_agent)#得到这个 agent 的所有可能的行动
            scores=list()
            value=-1000000
            for i in range (len(legal_actions)):
                cur_score=minimizer(state.generateSuccessor(index_of_agent, legal_actions[i]), depth, (index_of_agent + 1),alpha,beta)[0]
                if cur_score>value:
                    maxiAction=legal_actions[i]
                    value=cur_score
                if cur_score>beta:#只有比传入的 beta 小，才有可能被选择
                    return (value, maxiAction)
                if cur_score>alpha:# if the node isn't be deleted, renew the alpha
                    alpha=cur_score
            "*** YOUR CODE HERE ***"
            # for every legal action, update value, maxiAction and alpha:
            "*** YOUR CODE HERE ***"
            return (value, maxiAction)
        
        def minimizer(state, depth, index_of_agent, alpha, beta):#alpha & beta can not return back
            miniAction = None
            def terminal_condition(state,depth):
                "*** YOUR CODE HERE ***"
                return state.isLose() == True or depth==0 or state.isWin()==True
            if terminal_condition(state,depth) == True:# if the depth is enough I can get real value
                return (self.evaluationFunction(state), miniAction)
            #initialize value
            legal_actions = state.getLegalActions(index_of_agent)
            value=1000000
            for i in range (len(legal_actions)):
                if index_of_agent == state.getNumAgents() - 1:  # 此时到了最后一个ghost
                    scores = maximizer(state.generateSuccessor(index_of_agent, legal_actions[i]), (depth - 1), 0,alpha,beta)[0]
                else:  # 不是最后一个ghost，那么肯定还有一层minimizer
                    scores =minimizer(state.generateSuccessor(index_of_agent, legal_actions[i]), depth, index_of_agent + 1,alpha,beta)[0]
                if scores<value:
                    value=scores
                    miniAction=legal_actions[i]
                if scores<alpha:
                    return (value,miniAction)
                if scores<beta: #如果比 alpha 小一定将来会被 alpha 限制住；
                    beta=scores
            # for every legal action, update value and miniAction
            # for every legal action, update value, miniAction and beta
            "*** YOUR CODE HERE ***"
            return (value, miniAction)
        # initialize alpha/beta
        alpha = -1000000
        beta = 1000000
        action = maximizer(gameState, self.depth, 0, alpha, beta)[1]
        return action 


class MCTSAgent(MultiAgentSearchAgent):

    def getAction(self, gameState, mcts_time_limit=10):
        class Node:
            #难点： 如何区别不可访问的节点
            def __init__(self, data):
                self.north = None                   # 选择当前action为“north”对应的节点, <class 'Node'>
                self.east = None                    # 选择当前action为“east”对应的节点, <class 'Node'>
                self.west = None                    # 选择当前action为“west”对应的节点, <class 'Node'>
                self.south = None                   # 选择当前action为“south”对应的节点, <class 'Node'>
                self.stop = None                    # 选择当前action为“stop”对应的节点, <class 'Node'>
                self.parent = None                  # 父节点, <class 'Node'>
                self.statevalue = data[0]           # 该节点对应的游戏状态, <class 'GameState' (defined in pacman.py)>
                self.numerator = data[1]            # 该节点的分数
                self.denominator = data[2]          # 该节点的访问次数

        def Selection(cgs, cgstree):#选择
            '''
                cgs: current game state, <class 'GameState' (defined in pacman.py)>
                cgstree: current game state tree, <class 'Node'>
                
                YOUR CORE HERE (~30 lines or fewer)
                1. You have to find a node that is not completely expanded (e.g., node.north is None)
                2. When you find the node, return its corresponding game state and the node itself.
                3. You should use best_UCT() to find the best child of a node each time.

            '''
            legals = cgs.getLegalActions(0)
            if len(legals)==0:
                   return(cgs, cgstree)
            def uncompletely_expanded(node):#至少有一个未拓展
                return (node.north is None and "North" in legals) or\
                    (node.east is None and "East" in legals) or (node.south is None and "South" in legals) or\
                    (node.west is None and "West" in legals) or (node.stop is None and "Stop" in legals)
            #传入的一定是 Legalaction
            if uncompletely_expanded(cgstree) is False:#未找到一个未被拓展的节点,已经被完全拓展过
                children=[]#在 best_UCT 中有判断None
                if "North" in legals:
                    children.append((cgstree.north,"North"))#tuple
                if "East" in legals:
                    children.append((cgstree.east,"East"))
                if "West" in legals:
                    children.append((cgstree.west,"West"))
                if "South" in legals:
                    children.append((cgstree.south,"South"))
                if "Stop" in legals:
                    children.append((cgstree.stop,"Stop"))
                #assert len(children) >=1
                best_node_state,best_action=best_UCT(children)
                #print(children)
                if best_action=="North":
                    later_node=cgstree.north
                elif best_action=="East":
                    later_node=cgstree.east
                elif best_action=="West":
                    later_node=cgstree.west
                elif best_action=="South":
                    later_node=cgstree.south
                else:
                    later_node=cgstree.stop
                return Selection(best_node_state, later_node)
            return (cgs, cgstree)

        def Expansion(cgstree):#拓展
            legal_actions = cgstree.statevalue.getLegalActions(0)# pacman's legal actions
            '''
                YOUR CORE HERE (~20 lines or fewer)
                1. You should expand the current game state tree node by adding all of its children.
                2. You should use Node() to create a new node for each child.
                3. You can traverse the legal_actions to find all the children of the current game state tree node.
            '''
            def add_child(tree,child,action):
                #if tree.action is None:# haven't been expanded before
                if action=="North":
                    tree.north=child
                elif action=="South":
                    tree.south=child
                elif action=="East":
                    tree.east=child
                elif action == "West":
                    tree.west = child
                elif action == "Stop":
                    tree.stop = child
                else:
                    return
            for l_a in legal_actions:#可执行的动作
                A_child=Node([cgstree.statevalue.generateSuccessor(0,l_a),0,1])#初始化
                A_child.parent=cgstree
                add_child(cgstree,A_child,l_a)


        def Simulation(cgs, cgstree):#模拟
            '''
                This implementation is different from the one taught during the lecture.
                All the nodes during a simulation trajectory are expanded.
                We choose to more quickly expand our game tree (and hence pay more memory) to get a faster MCTS improvement in return.
            '''
            simulation_score = 0
            while cgstree.statevalue.isWin() is False and cgstree.statevalue.isLose() is False:
                cgs, cgstree = Selection(cgs, cgstree)
                Expansion(cgstree)
            '''
                YOUR CORE HERE (~4 lines)
                You should modify the simulation_score of the current game state.
            '''
            if cgs.isWin() is True:
                simulation_score=1
            else:
                simulation_score=0
            return simulation_score, cgstree

        def Backpropagation(cgstree, simulation_score):#回溯
            while cgstree.parent is not None:
                '''
                    YOUR CORE HERE (~3 lines)
                    You should recursively update the numerator and denominator of the game states until you reaches the root of the tree.
                '''
                cgstree.numerator+=simulation_score
                cgstree.denominator+=1
                cgstree=cgstree.parent
            return cgstree

        # 根据UCT算法选择最好的子节点及其对应的action。你不需要修改这个函数。
        def best_UCT(children, random_prob=0.3):#0.3为超参数
            '''
                children: list of tuples, each tuple contains a child node and the action that leads to it
                random_prob: the probability of choosing a random action when UCT values are the same

                return: the best child node's game state and the action that leads to it
            '''
            i = 0
            while i < len(children):#children 里的 action 是最终决策
                if children[i][0] is None or children[i][1] == 'Stop':
                    children.pop(i)#删除无效节点，之后不会重新排序
                else:
                    i = i+1

            children_UCT = []
            for i in range(len(children)):
                #ssert  children[i][0].denominator==1
                value = ((children[i][0].numerator / children[i][0].denominator) + sqrt(2) * sqrt(
                    ((log(children[i][0].parent.denominator))/log(2.71828)) / children[i][0].denominator)), children[i][1]
            #此时 c=sqrt(2)
                children_UCT.append(value)

            max_index = 0
            equal_counter = 1

            for i in range(len(children_UCT)-1):
                if children_UCT[i][0] == children_UCT[i+1][0]:
                    equal_counter = equal_counter + 1
            
            # 如果所有的UCT值都相等，用启发式函数来选择
            if equal_counter == len(children_UCT):
                
                # 有random_prob的概率随机选择
                decision_maker = random.randint(1, 101)
                if decision_maker < (1 - random_prob) * 100:
                    eval_list = [] #用两个list对照存放
                    max_index_list = []
                    for i in range(len(children)):
                        eval_list.append(HeuristicFunction(
                            children[i][0].statevalue))
                    max_index_list.append(eval_list.index(max(eval_list)))#可能有多个这样的 max ，heuristic值
                    maxval = eval_list.pop(max_index_list[-1])#弹出最后一个
                    eval_list.insert(max_index_list[-1], -9999)#插入一个非常小的值
                    while maxval in eval_list:
                        max_index_list.append(eval_list.index(max(eval_list)))#找到最大的，插入 index 表中
                        eval_list.pop(max_index_list[-1])#从 eval_list 中弹出这个最大的
                        eval_list.insert(max_index_list[-1], -9999)# 插入一个非常小的值
                    #最终目的：把 >= maxval 的所有节点都去除， 从其指标集中随机选择一个值
                    max_index = random.choice(max_index_list)
                else:
                    max_index = random.randint(0, len(children)-1)
            
            # 否则选最好的UCT对应的节点
            else:
                maximumvalueofUCT = -9999
                for i in range(len(children_UCT)):
                    if children_UCT[i][0] > maximumvalueofUCT:
                        max_index = i
                        maximumvalueofUCT = children_UCT[i][0]
            return (children[max_index][0].statevalue, children[max_index][1])

        # 样例启发式函数，你不需要修改。这个函数会返回一个游戏状态的分数。
        def HeuristicFunction(currentGameState):
            new_position = currentGameState.getPacmanPosition()
            new_food = currentGameState.getFood().asList()

            food_distance_min = float('inf')
            for food in new_food:
                food_distance_min = min(
                    food_distance_min, manhattanDistance(new_position, food))

            ghost_distance = 0
            ghost_positions = currentGameState.getGhostPositions()

            for i in ghost_positions:
                ghost_distance = manhattanDistance(new_position, i)
                if (ghost_distance < 1):
                    return -float('inf')

            food = currentGameState.getNumFood()
            pellet = len(currentGameState.getCapsules())

            food_coefficient = 999999
            pellet_coefficient = 19999
            food_distance_coefficient = 999

            game_rewards = 0
            if currentGameState.isLose():
                game_rewards = game_rewards - 99999
            elif currentGameState.isWin():
                game_rewards = game_rewards + 99999

            answer = (1.0 / (food + 1) * food_coefficient) + ghost_distance + (
                1.0 / (food_distance_min + 1) * food_distance_coefficient) + (
                1.0 / (pellet + 1) * pellet_coefficient) + game_rewards

            return answer

        def endSelection(cgstree):
            children = []
            destin = (cgstree.north, "North")
            children.append(destin)
            destin = (cgstree.east, "East")
            children.append(destin)
            destin = (cgstree.south, "South")
            children.append(destin)
            destin = (cgstree.west, "West")
            children.append(destin)
            destin = (cgstree.stop, "Stop")
            children.append(destin)
            action = best_UCT(children, random_prob=0.0)[1]
            return action
        
        '''YOUR CODE HERE (~1-2 line)‘’‘
        ’‘’initialize root node cgstree (class Node)'''
        cgstree=Node([gameState,0,1])

        for _ in range(mcts_time_limit):#在时间限制内搜索
            gameState, cgstree = Selection(gameState, cgstree)                  # 根据当前的游戏状态和搜索树，选择一个最好的子节点
            Expansion(cgstree)                                                  # 扩展这个选到的节点
            simulation_score, cgstree = Simulation(gameState, cgstree)          # 从这个节点开始模拟
            cgstree = Backpropagation(cgstree, simulation_score)                # 将模拟的结果回溯到根节点，cgstree为根节点
            gameState = cgstree.statevalue                              
        
        return endSelection(cgstree)#作出最终选择
