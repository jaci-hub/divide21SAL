'''

Name: Jacinto Jeje Matamba Quimua

Date: 10/30/2025



This script creates a state-action lookup document for the game Divide21 when the number of digits in the initial number are: 2 and 3



The Minimax algorithm (with full depth) is used to accomplish this.

'''



from functools import lru_cache

import json

import math

import random

import copy





def get_all_legal_actions(game_state):

    '''

    Generate all legal actions for a given game state

    

    the game state is structured as:

        (1) "dynamic number" (str): the current number

        (2) "available digits per index" (list of dicts): digits available at each index

        (3) "players" (list): the list of player objects (like dictionaries) that have attributes such as id, score, etc

        (4) "turn" (int): the id of the player with the current turn

    

    return:

        it returns a list of dicts with the structure:

            (1) division (bool): true/false

            (2) digit (int): if division=true, then it is the divisor, else it is the new digit in the index chosen

            (3) index (int): if division=true, then it is None, else the index where the digit will be overwriten

    '''

    all_legal_actions = []

    number_of_digits = len(game_state["dynamic number"])



    # (1) Digit change actions

    for index in range(number_of_digits):

        available_digits = game_state["available digits per index"][index]

        for digit in available_digits:

            action = {

                "division": False,

                "digit": digit,

                "index": index

            }

            all_legal_actions.append(action)

    

    # (2) Division actions

    for divisor in range(2, 10):

        if int(game_state["dynamic number"]) % divisor == 0:

            action = {

                "division": True,

                "digit": divisor,

                "index": None

            }

            all_legal_actions.append(action)



    return all_legal_actions





def game_over(game_state):

    # (1) quotient 1

    if int(game_state["dynamic number"]) == 1:

        return True

    # (2) max points

    for player in game_state["players"]:

        if player['score'] >= 9*len(game_state["dynamic number"]):

            return True

    # (3) only one player left without -max points or less

    count = 0

    for player in game_state["players"]:

        if player['score'] <= -9*len(game_state["dynamic number"]):

            count += 1

    if count == len(game_state["players"]) - 1:

        return True

    

    return False





def index_available_digit_list_is_empty(index, available_digits_per_index):

    if not available_digits_per_index:

        return True



    index_available_digit_list = available_digits_per_index[index]

    return len(index_available_digit_list) == 0





def update_available_digits_per_index(index, digits, dynamic_number, available_digits_per_index):

    if index is None: # division was performed

        # Loop through all indices.

        # Check if any list became empty AFTER the quotient digits were removed.

        for i in range(digits):

            if index_available_digit_list_is_empty(i, available_digits_per_index):

                # This list is empty, so we must refill it.

                current_digit = int(dynamic_number[i]) # Get digit from the NEW number

                all_digits = [d for d in range(10) if d != current_digit]

                all_digits = remove_all_prohibited_digits_at_given_index_from_given_list(i, all_digits, dynamic_number)

                available_digits_per_index[i] = all_digits

                

    else: # division was not performed (this is the digit-change logic)

        # This part was already correct.

        # If the list at this index is not empty, do nothing.

        if not index_available_digit_list_is_empty(index, available_digits_per_index):

            return available_digits_per_index



        # The list at this index IS empty, so refill it.

        current_digit = int(dynamic_number[index])

        all_digits = [d for d in range(10) if d != current_digit]

        all_digits = remove_all_prohibited_digits_at_given_index_from_given_list(index, all_digits, dynamic_number)

        available_digits_per_index[index] = all_digits

    

    return available_digits_per_index

    



def remove_each_quotient_digit_from_available_digits_per_index(quotient_string, available_digits_per_index):

    if not available_digits_per_index:

        return



    for i in range(len(quotient_string)):

        index_available_digit_list = available_digits_per_index[i]

        if not index_available_digit_list:

            continue



        digit_to_remove = int(quotient_string[i])

        index_available_digit_list = [d for d in index_available_digit_list if d != digit_to_remove]

        available_digits_per_index[i] = index_available_digit_list

    

    return available_digits_per_index





def remove_digit_from_index_available_digits(index, digit_to_remove, available_digits_per_index):

    if not available_digits_per_index:

        return



    index_available_digit_list = available_digits_per_index[index]

    if not index_available_digit_list:

        return



    index_available_digit_list = [d for d in index_available_digit_list if d != digit_to_remove]

    available_digits_per_index[index] = index_available_digit_list

    

    return available_digits_per_index





def simulate_action(action, game_state):

    '''

    Simulate a given action or move in a given game state, and return the new game state

    '''

    

    # deepcopy to prevent state mutation

    new_game_state = copy.deepcopy(game_state) 

    

    # unpack action

    division = action["division"]

    digit = action["digit"]

    index = action["index"]

    

    # unpack game state (from the new copied state)

    dynamic_number = new_game_state["dynamic number"]

    turn = new_game_state["turn"]

    num_of_digits = len(dynamic_number)

    available_digits_per_index = new_game_state["available digits per index"]

    

    # perform division

    if division:

        num = int(dynamic_number)

        new_num = num // digit

        new_num = str(new_num)

        # keep the original number of digits

        if len(new_num) < num_of_digits:

            new_num = "0"*(num_of_digits - len(new_num)) + new_num

        new_game_state["dynamic number"] = new_num

        # update the list of available digits per index

        #   (1) remove each quotient digit from available digits per index

        new_game_state["available digits per index"] = remove_each_quotient_digit_from_available_digits_per_index(new_game_state["dynamic number"], available_digits_per_index)

        #   (2) update available digits per index

        new_dynamic_number = new_game_state["dynamic number"] # Get the NEW number

        new_game_state["available digits per index"] = update_available_digits_per_index(index, num_of_digits, new_dynamic_number, new_game_state["available digits per index"])

        # update player score

        new_game_state["players"][turn]["score"] += digit

    # digit change

    else:

        num_str = list(dynamic_number)

        num_str[index] = str(digit)

        new_game_state["dynamic number"] = "".join(num_str)

        # update the list of available digits per index

        #   (1) remove digit from index available digits

        new_game_state["available digits per index"] = remove_digit_from_index_available_digits(index, digit, available_digits_per_index)

        #   (2) update available digits per index

        new_dynamic_number = new_game_state["dynamic number"] # Get the NEW number

        new_game_state["available digits per index"] = update_available_digits_per_index(index, num_of_digits, new_dynamic_number, new_game_state["available digits per index"])

        # update player turn

        new_game_state["turn"] = (new_game_state["turn"] + 1) % len(new_game_state["players"])



    return new_game_state





def _get_hashable_state(game_state):

    '''

    converts the game_state var, which is a mutable dict state, into an immutable, hashable tuple.

    '''

    

    # Convert available_digits_per_index (dict of lists) to a tuple of tuples

    # We must sort the keys (indices) to ensure {0: [1], 1: [2]} and {1: [2], 0: [1]} hash to the same value

    available_digits_tuple = tuple(

        sorted(

            (index, tuple(sorted(digits))) for index, digits in game_state["available digits per index"].items()

        )

    )

    

    # Convert players (list of dicts) to a simple tuple of scores

    scores_tuple = tuple(player['score'] for player in game_state["players"])

    

    return (

        game_state["dynamic number"],

        available_digits_tuple,

        scores_tuple,

        game_state["turn"]

    )





@lru_cache(maxsize=None)

def minimax_full_depth(hashable_state, maximizing):

    '''

    Compute the full game tree for a given game state

    '''

    

    # NOTE: We can't use the original 'game_state' dict here, as it's not cached.

    # We must re-build a temporary one if needed, or pass all components.

    # For 'game_over' and 'get_all_legal_actions', we need the dict.

    # This is inefficient, but necessary for the cache to work with your existing code.

    

    # --- Reconstruct the game_state dict from the hashable_state tuple ---

    dynamic_number, available_digits_tuple, scores_tuple, turn = hashable_state

    

    available_digits_per_index = {item[0]: list(item[1]) for item in available_digits_tuple}

    players = [{'id': i, 'score': score} for i, score in enumerate(scores_tuple)]

    

    game_state = {

        "dynamic number": dynamic_number,

        "available digits per index": available_digits_per_index,

        "players": players,

        "turn": turn

    }

    # --- End reconstruction ---



    if game_over(game_state):

        # win for current player

        return float("inf") if maximizing else -float("inf")



    actions = get_all_legal_actions(game_state)

    if not actions:

        # No legal action = loss for current player

        return -float("inf") if maximizing else float("inf")



    if maximizing:

        value = -float("inf")

        for action in actions:

            new_game_state = simulate_action(action, game_state)

            # --- Convert new state to hashable before recursive call ---

            new_hashable_state = _get_hashable_state(new_game_state)

            value = max(value, minimax_full_depth(new_hashable_state, False))

        return value

    else:

        value = float("inf")

        for action in actions:

            new_game_state = simulate_action(action, game_state)

            # --- Convert new state to hashable before recursive call ---

            new_hashable_state = _get_hashable_state(new_game_state)

            value = min(value, minimax_full_depth(new_hashable_state, True))

        return value





def get_best_action(game_state):

    '''

    return the optimal action for a given game state

    '''

    best_val = -float("inf")

    best_action = None

    

    actions = get_all_legal_actions(game_state)

    # Handle case with no moves

    if not actions:

        return None, -float("inf") # No best move, it's a loss



    for action in actions:

        new_game_state = simulate_action(action, game_state)

        

        # --- Convert to hashable state to call the cached function ---

        hashable_state = _get_hashable_state(new_game_state)

        val = minimax_full_depth(hashable_state, False)

        

        if val > best_val:

            best_val = val

            best_action = action

    return best_action, best_val





def get_prohibited_digit_list_at_index(index, dynamic_number):

    prohibited = set()

    # no leading zero

    if index == 0:

        prohibited.add(0)



    # can’t make number 0 or 1

    for d in [0, 1]:

        modified = dynamic_number[:index] + str(d) + dynamic_number[index + 1:]

        if int(modified) in (0, 1):

            prohibited.add(d)



    return list(prohibited)





def remove_all_prohibited_digits_at_given_index_from_given_list(index, digit_list, dynamic_number):

    prohibited_digits = set(get_prohibited_digit_list_at_index(index, dynamic_number))

    return [d for d in digit_list if d not in prohibited_digits]





def setup_available_digit_list_per_index(digits, dynamic_number):

    available_digit_list_per_index = {}



    for i in range(digits):

        current_digit = int(dynamic_number[i])

        all_digits = [d for d in range(10) if d != current_digit]

        filtered_digits = remove_all_prohibited_digits_at_given_index_from_given_list(i, all_digits, dynamic_number)

        available_digit_list_per_index[i] = filtered_digits



    return available_digit_list_per_index





def get_all_game_states_for_number(num):
    '''
    return all the possible game states for a given number.
    the only attribute value that changes is the players attribute because we need to compute the game state for
        all score combination, for example, when the score is 0-0, 0-1, 1-1, 0-2, ..., (9*digits-1)-(9*digits-1)
    '''
    digits = len(num)
    game_states = []
    
    base_available_digits = setup_available_digit_list_per_index(digits, num)
    
    max_score = 9*digits
    for score0 in range(max_score + 1): 
        for score1 in range(max_score + 1):
            game_state = {
                "dynamic number": num,
                "available digits per index": copy.deepcopy(base_available_digits), 
                "players": [{'id': 0, 'score': score0}, {'id': 1, 'score': score1}], # two players by default
                "turn": 0
            }
            game_states.append(game_state)
    
    return game_states


def get_minimax_data(digits):
    min_int = 10**(digits - 1)
    max_int = 10**digits
    
    data = {}
    for num in range(min_int, max_int):
        print(f"Solving for number: {num}...") # Added a print for progress
        game_states = get_all_game_states_for_number(str(num))
        for game_state in game_states:
            action, val = get_best_action(game_state)
            outcome = True if val == float("inf") else False
            
            # --- FIX: The key MUST be the hashable state ---
            state_key = _get_hashable_state(game_state)
            
            # Convert key to string for JSON
            # JSON keys must be strings. A tuple (10, (...), (0,0), 0) becomes "('10', ((...), (...)), (0, 0), 0)"
            data[str(state_key)] = {
                # "state": game_state, # Don't need to save this, it's the key
                "action": action,
                "win": outcome
            }

    return data


def make_document():
    for digit in [2, 3]:
        minimax_data = get_minimax_data(digit)

        with open(f"divide21_{digit}_digits_state_action_lookup.json", "w") as f:
            json.dump(minimax_data, f, indent=2)


if __name__ == "__main__":

    # num="850" # uncomment for testing

    # digits = len(num) # uncomment for testing

    # # ------------ START test the output of all legal actions START ------------

    # available_digits_per_index = setup_available_digit_list_per_index(digits, num)

    # game_state = {

    #     "dynamic number": num,

    #     "available digits per index": available_digits_per_index,

    #     "players": [{'id': 0, 'score': 0}, {'id': 1, 'score': 0}],

    #     "turn": 0

    # }

    # actions = get_all_legal_actions(game_state)

    # all_legal_actions = {"all legal actions": actions}

    

    # with open(f"all_legal_actions_{num}.json", "w") as f:

    #     json.dump(all_legal_actions, f, indent=2)

    # # ------------ END test the output of all legal actions END ------------

    

    # # ------------ START test the output of all game states START ------------

    # game_states = get_all_game_states_for_number(num)

    # all_game_states = {"all game states": game_states}

    

    # with open(f"all_game_states_{num}.json", "w") as f:

    #     json.dump(all_game_states, f, indent=2)

    # # ------------ END test the output of all game states END ------------

    

    # # ------------ START test the output of simulate action START ------------

    # simulated_action = {}

    # available_digits_per_index = setup_available_digit_list_per_index(digits, num)

    # game_state = {

    #     "dynamic number": num,

    #     "available digits per index": available_digits_per_index,

    #     "players": [{'id': 0, 'score': 0}, {'id': 1, 'score': 0}],

    #     "turn": 0

    # }

    # simulated_action["game state BEFORE action"] = game_state

    

    # actions = get_all_legal_actions(game_state)

    # action = random.choice(actions)

    # simulated_action["action"] = action

    

    # new_game_state = simulate_action(action, game_state)

    # simulated_action["game state AFTER action"] = new_game_state

    

    # with open(f"simulated_action_{num}.json", "w") as f:

    #     json.dump(simulated_action, f, indent=2)

    # # ------------ END test the output of simulate action END ------------

    

    make_document()

