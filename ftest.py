import sys
import copy
sys.path.append('aima-python')
from games import *
import math


def get_default_state(p1d, p2d):
    state = {"total_dice": p1d + p2d, "player_1_dice": [None for i in range(p1d)], "player_2_dice": [None for i in range(p2d)], "history": [], "previous_bid": None, "player_to_bid": 1, "winner": None}

    return state

def determinize_state_random(state, player):
    determined_state = copy.deepcopy(state)
    roll_dice(determined_state)
    if player == 1:
        determined_state["player_1_dice"] = state["player_1_dice"]
    else:
        determined_state["player_2_dice"] = state["player_2_dice"]

    return determined_state
    

def roll_dice(state):
    for i in range(len(state["player_1_dice"])):
        state["player_1_dice"][i] = random.randint(1, 3)
    for i in range(len(state["player_2_dice"])):
        state["player_2_dice"][i] = random.randint(1, 3)


def get_valid_bids(previous_bid, total_dice):
    valid_bids = []
    if previous_bid is None:
        for i in range(1, total_dice + 1):
            for j in range(1, 4):
                valid_bids.append((i,j))
        return valid_bids

    
    (freq, num) = previous_bid
    for i in range(freq + 1,  total_dice + 1):
        valid_bids.append((i, num))

    for i in range(freq, total_dice):
        for j in range(num + 1, 4):
            valid_bids.append((i, j))

    valid_bids.append((-1, -1))

    return valid_bids

def apply_bid(state, bid):
    if bid[0] == -1: 
        state["winner"] = assign_winner(state)
    state["history"].append(bid)
    state["previous_bid"] = bid
    if state["player_to_bid"]  == 1:
        state["player_to_bid"] = 2
    else:
        state["player_to_bid"] = 1


def assign_winner(state):
    (freq, num) = state["previous_bid"]
    total_dice = state["player_1_dice"] + state["player_2_dice"]
    if total_dice.count(num) < freq:
        if state["player_to_bid"] == 1:
            return 1
        else:
            return 2
    else:
        if state["player_to_bid"] == 1:
            return 2
        else:
            return 1





class LiarsDice(Game):
    """Similar to Fig52Game but bigger. Useful for visualisation"""

    def __init__(self, player_1_dice=3, player_2_dice=3, k=3):
        self.player_1_dice = player_1_dice
        self.player_2_dice = player_2_dice
        self.initial = get_default_state(player_1_dice, player_2_dice)
        roll_dice(self.initial)

    def actions(self, state):
        return get_valid_bids(state["previous_bid"], state["total_dice"])

    def result(self, state, move):
        s2 = copy.deepcopy(state)
        apply_bid(s2, move)
        return s2

    def utility(self, state, player):
        if not self.terminal_test(state):
            return 0
        if state["winner"] == player:
            return 1
        else:
            return -1



    def terminal_test(self, state):
        if state["winner"] is None:
            return False
        return True
            

    def to_move(self, state):
        return state["player_to_bid"]


def monte_carlo_tree_search(state, game, N=1000):
    def select(n):
        """select a leaf node in the tree"""
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        """expand the leaf node by adding all its children states"""
        if not n.children and not game.terminal_test(n.state):
            n.children = {MCT_Node(state=game.result(n.state, action), parent=n): action
                          for action in game.actions(n.state)}
        return select(n)

    def simulate(game, state):
        """simulate the utility of current state by random picking a step"""
        player = game.to_move(state)
        while not game.terminal_test(state):
            action = random.choice(list(game.actions(state)))
            state = game.result(state, action)
        v = game.utility(state, player)
        return -v

    def backprop(n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        # if utility == 0:
        #     n.U += 0.5
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=state)

    for i in range(N):
        # print(i)
        leaf = select(root)
        child = expand(leaf)
        result = simulate(game, child.state)
        backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state)

class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""

    def __init__(self, parent=None, state=None, U=0, N=0):
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children = {}
        self.actions = None


def ucb(n, C=1.4):
    return np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)

def determinized_monte_carlo(state, game):
    actions = {}
    for i in range(10):
        determined_state = determinize_state_random(state, game.to_move(state))
        return monte_carlo_tree_search(determined_state, game)

    normalizing_sum = 0
    for i in actions.keys():
        normalizing_sum += actions[i]

    for i in actions.keys():
        actions[i] /= normalizing_sum

    return max(actions, key = actions.get)

def information_set_monte_carlo(state, game, N = 100, M = 10):
    state = determinize_state_random(state, game.to_move(state))
    def select(n):
        """select a leaf node in the tree"""
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        """expand the leaf node by adding all its children states"""
        if not n.children and not game.terminal_test(n.state):
            n.children = {MCT_Node(state=game.result(n.state, action), parent=n): action
                          for action in game.actions(n.state)}
        return select(n)

    def simulate(game, state):
        """simulate the utility of current state by random picking a step"""
        player = game.to_move(state)
        while not game.terminal_test(state):
            action = random.choice(list(game.actions(state)))
            state = game.result(state, action)
        v = game.utility(state, player)
        return -v

    def backprop(n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        # if utility == 0:
        #     n.U += 0.5
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=state)


    for i in range(N):

        # print(i)
        leaf = select(root)
        child = expand(leaf)
        result = simulate(game, determinize_state_random(child.state, game.to_move(child.state)))
        backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state)
   



def run_simulation(p1d, p2d, p1, p2, display = True):

    game = LiarsDice(p1d, p2d)

    real_state = game.initial
    if display:
        print(real_state)

    while not game.terminal_test(real_state):
        if game.to_move(real_state) == 1:
            action = p1(real_state, game)
            real_state = game.result(real_state, action)
            if display:
                print(action, real_state)
        else:
            action = p2(real_state, game)
            real_state = game.result(real_state, action)
            if display:
                print(action, real_state)

    return real_state["winner"]

def random_agent(state, game):
    return random.choice(game.actions(state)) 

def conservative_agent(state, game):
    for bid in sorted(game.actions(state), key = lambda x: x[0]):
        (freq, num) = bid
        player = game.to_move(state)
        dice = state["player_1_dice"]
        if player == 2:
            dice = state["player_2_dice"]

        if dice.count(num) >= freq:
            return bid

    return (-1, -1)
            

def real_agent(state, game):
    if game.to_move(state) == 1:
        print("Your rolls:", state["player_1_dice"]) 

    else:
        print("Your rolls:", state["player_1_dice"]) 

    print("Enemy last move: ", state["previous_bid"])
    freq = -2
    num = -2
    while (freq, num) not in game.actions(state):
        print("Valid moves: ", game.actions(state))
        freq = int(input("enter freq "))
        num = int(input("enter num "))

        if (freq, num) == (-1, -1):
            print("Player 1 dice: ", state["player_1_dice"])
            print("Player 2 dice: ", state["player_2_dice"])


    return (freq, num)



def state_to_key(state):
    key = [state["total_dice"], tuple(state["history"]),]
    if state["player_to_bid"] == 1:
        key.append(tuple(state["player_1_dice"]))
    else:
        key.append(tuple(state["player_2_dice"]))

    return tuple(key)

def play_round_cfr(player1, player2, state, game, values):
    winner = None
    p1_states = []
    p2_states = []
    if state["player_to_bid"] == 1:
        p1_states.append(state_to_key(state))
    else:
        p2_states.append(state_to_key(state))
    while not game.terminal_test(state):
        if state["player_to_bid"] == 1:
            bid = player1(state)
            new_state = game.result(state, bid)
            p1_states.append(state_to_key(new_state))
        else: 
            bid = player2(state)
            new_state = game.result(state, bid)
            p2_states.append(state_to_key(new_state))

        state = new_state

    winner = state["winner"]

    if winner == 1:
        for state in p1_states:
            if state in values:
                values[state] += 1 
            else:
                values[state] = 1
        for state in p2_states:
            if state in values:
                values[state] -= 1
            else:
                values[state] = -1

    else:
        for state in p1_states:
            if state in values:
                values[state] -= 1 
            else:
                values[state] = -1
        for state in p2_states:
            if state in values:
                values[state] += 1
            else:
                values[state] = 1
    
    return winner


def regret_min_agent(state, game):
    values = regret_min_player_datagen(state, game)
    moves = []
    regrets = []
    if state_to_key(state) in values:
        state_value = values[state_to_key(state)]

        for bid in get_valid_bids(state["previous_bid"], state["total_dice"]):
            s2 = copy.deepcopy(state)
            apply_bid(s2, bid)
            if state_to_key(s2) in values:
                state_i_value = values[state_to_key(s2)]
                regret = state_i_value - state_value
                if regret < 0:
                    regret = 0
                # regret = max(regret, 0)
                regrets.append(regret)
                moves.append(bid)

    regrets_sum = sum(regrets)
    if len(moves) == 0 or regrets_sum == 0:
        return (-1, -1)
        # print("couldnt find regret!")

    # for i in range(len(regrets)):
    #     regrets[i] = regrets[i] / regrets_sum
    # # for (move, regret) in zip(moves,regrets):
    # #     print("move ", move , " prob ", 100 * regret) 
    # # print(sum(regrets))
    # return random.choices(moves, weights=regrets)[0]

    return moves[np.argmax(regrets)]

def regret_min_player_datagen(state, game):
    values = {}

    for i in range(1000):
        s2= copy.deepcopy(state)
        s2 = determinize_state_random(state, game.to_move(state))

        def random_player(state):
            return random.choice(get_valid_bids(state["previous_bid"], state["total_dice"])) 
            
        play_round_cfr(random_player, random_player, s2, game, values)

    # print(values)
    return values


agents = {"PIMC": determinized_monte_carlo, "ISMCTS": information_set_monte_carlo, "MCCFR": regret_min_agent, "CONSERVATIVE": conservative_agent, "RANDOM": random_agent}


for p1_strategy in agents:
    for p2_strategy in agents:
        p1w = 0
        p2w = 0

        print(f"P1 ({p1_strategy}) vs P2 ({p2_strategy}) [NEUTRAL]")
        for i in range(1000):
            winner = run_simulation(3, 3, agents[p1_strategy], agents[p2_strategy], display=False)

            if winner == 1:
                p1w += 1
            else:
                p2w += 1

        print(p1w * 10, p2w * 10)

        p1w = 0
        p2w = 0

        print(f"P1 ({p1_strategy}) vs P2 ({p2_strategy}) [FAVORED]")
        for i in range(1000):
            winner = run_simulation(3, 1, agents[p1_strategy], agents[p2_strategy], display=False)

            if winner == 1:
                p1w += 1
            else:
                p2w += 1

        print(p1w , p2w)




