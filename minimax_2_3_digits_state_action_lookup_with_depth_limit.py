"""
Full (reachable) minimax solver for Divide21 (2-digit starts).
- Starts from scores (0,0)
- Uses alpha-beta pruning + transposition table
- Detects cycles (visited states per path)
- Avoids deepcopy loops via safe manual copying/cloning

Author: Jacinto Jeje Matamba Quimua
Date: 2025-10-30
"""

from functools import lru_cache
import json
import math
import sys
import time
from typing import Dict, Tuple, List

# Optional: increase recursion limit a bit (safe after cycle detection)
sys.setrecursionlimit(pow(10, 5))

# -------------------------
# Game utilities & rules
# -------------------------

def get_digits(x: str) -> List[int]:
    return [int(c) for c in x]

def digits_to_str(digits: List[int]) -> str:
    return "".join(str(d) for d in digits)

def setup_available_digit_list_per_index(digits: int, dynamic_number: str) -> Dict[int, List[int]]:
    """
    Return a fresh dict mapping index -> list of allowed replacement digits
    given the current number. (No leading zero; avoid making value 0 or 1.)
    """
    out = {}
    for i in range(digits):
        curr = int(dynamic_number[i])
        cand = [d for d in range(10) if d != curr]
        # remove digits that would make the number equal 0 or 1, or leading zero
        filtered = []
        for d in cand:
            if i == 0 and d == 0:
                continue
            trial = dynamic_number[:i] + str(d) + dynamic_number[i+1:]
            if int(trial) in (0, 1):
                continue
            filtered.append(d)
        out[i] = filtered
    return out

def get_all_legal_actions(game_state: Dict) -> List[Dict]:
    """
    Return a list of actions; action is dict:
      {"division": bool, "digit": int, "index": int|None}
    """
    actions = []
    dyn = game_state["dynamic number"]
    nd = len(dyn)

    # digit-change moves
    for i in range(nd):
        for d in game_state["available digits per index"][i]:
            actions.append({"division": False, "digit": d, "index": i})

    # division moves by 2..9
    x = int(dyn)
    for k in range(2, 10):
        if x % k == 0:
            actions.append({"division": True, "digit": k, "index": None})
    return actions

def game_over(game_state: Dict) -> bool:
    dyn = int(game_state["dynamic number"])
    if dyn == 1:
        return True
    nd = len(game_state["dynamic number"])
    max_points = 9 * nd
    # someone reached max positive points
    for p in game_state["players"]:
        if p["score"] >= max_points:
            return True
    # only one player remains above -max (opponent all <= -max)
    count_bad = 0
    for p in game_state["players"]:
        if p["score"] <= -max_points:
            count_bad += 1
    if count_bad == len(game_state["players"]) - 1:
        return True
    return False

# -------------------------
# State copying & simulation
# -------------------------

def clone_game_state(game_state: Dict) -> Dict:
    """
    Make a safe, independent copy of game_state (no shared lists/dicts).
    This avoids deepcopy coping with accidental cycles.
    """
    return {
        "dynamic number": str(game_state["dynamic number"]),
        "available digits per index": {i: list(lst) for i, lst in game_state["available digits per index"].items()},
        "players": [{"id": p["id"], "score": p["score"]} for p in game_state["players"]],
        "turn": int(game_state["turn"])
    }

def remove_each_quotient_digit_from_available_digits_per_index(quotient_string: str, available_digits_per_index: Dict[int, List[int]]):
    """
    Return a new dict (cloned) where for each index we remove the digit that appears
    in the quotient at that index (if present in the list).
    """
    out = {i: list(lst) for i, lst in available_digits_per_index.items()}
    for i, ch in enumerate(quotient_string):
        d = int(ch)
        if i in out:
            out[i] = [x for x in out[i] if x != d]
    return out

def remove_digit_from_index_available_digits(index: int, digit_to_remove: int, available_digits_per_index: Dict[int, List[int]]):
    out = {i: list(lst) for i, lst in available_digits_per_index.items()}
    if index in out:
        out[index] = [d for d in out[index] if d != digit_to_remove]
    return out

def remove_all_prohibited_digits_at_given_index_from_given_list(index: int, digit_list: List[int], dynamic_number: str) -> List[int]:
    """
    Given candidate digits, filter out those that would be illegal (leading zero or make number 0/1).
    """
    out = []
    for d in digit_list:
        if index == 0 and d == 0:
            continue
        trial = dynamic_number[:index] + str(d) + dynamic_number[index+1:]
        if int(trial) in (0, 1):
            continue
        out.append(d)
    return out

def update_available_digits_per_index(index, digits, dynamic_number, available_digits_per_index):
    """
    Refill any empty lists caused after operations. Always returns a new dict (no in-place edits).
    If index is None (division case), we consider every index and refill empty lists using the
    new dynamic_number (quotient) digits.
    If index is an integer (digit-change case), only refill that index if empty.
    """
    out = {i: list(lst) for i, lst in available_digits_per_index.items()}

    if index is None:
        for i in range(digits):
            if not out.get(i):  # empty or missing
                current_digit = int(dynamic_number[i])
                all_digits = [d for d in range(10) if d != current_digit]
                filtered = remove_all_prohibited_digits_at_given_index_from_given_list(i, all_digits, dynamic_number)
                out[i] = filtered
    else:
        if not out.get(index):  # refill only if empty
            current_digit = int(dynamic_number[index])
            all_digits = [d for d in range(10) if d != current_digit]
            filtered = remove_all_prohibited_digits_at_given_index_from_given_list(index, all_digits, dynamic_number)
            out[index] = filtered
    return out

def simulate_action(action: Dict, game_state: Dict) -> Dict:
    """
    Safely simulate an action from a given game_state; return a new independent game_state.
    """
    state = clone_game_state(game_state)  # safe clone
    division = action["division"]
    digit = action["digit"]
    index = action["index"]
    dyn = state["dynamic number"]
    nd = len(dyn)
    turn = state["turn"]

    if division:
        num = int(dyn)
        new_num = str(num // digit)
        if len(new_num) < nd:
            new_num = "0" * (nd - len(new_num)) + new_num
        state["dynamic number"] = new_num

        # remove quotient digits from available lists
        state["available digits per index"] = remove_each_quotient_digit_from_available_digits_per_index(new_num, state["available digits per index"])
        # update (refill any emptied lists)
        state["available digits per index"] = update_available_digits_per_index(None, nd, state["dynamic number"], state["available digits per index"])
        # update score (player gets digit)
        state["players"][turn]["score"] += digit
        # NOTE: player continues to play in your rule if they wish — but in our search we treat the move as single
        # atomic action that changes state and hands turn to opponent only for digit-change moves.
        # To reflect "same player may continue dividing" you'd need chain-generation as single moves;
        # here we model single-step division and the minimax will consider further divisions as subsequent moves by same player
        # only if you implement chain-as-one-move logic. (We keep single-step for simplicity and correctness.)
    else:
        # digit change: replace digit at index, remove that digit from that index's available list,
        # refill if necessary, and advance turn
        digits_list = list(state["dynamic number"])
        digits_list[index] = str(digit)
        state["dynamic number"] = "".join(digits_list)

        state["available digits per index"] = remove_digit_from_index_available_digits(index, digit, state["available digits per index"])
        state["available digits per index"] = update_available_digits_per_index(index, nd, state["dynamic number"], state["available digits per index"])
        # swap turn
        state["turn"] = (turn + 1) % len(state["players"])

    return state

# -------------------------
# Hashable representation
# -------------------------

def _get_hashable_state(game_state: Dict) -> Tuple:
    """
    Convert the game_state into a canonical, hashable tuple used as key in caches.
    """
    dyn = game_state["dynamic number"]
    # sorted (index, tuple(sorted(digits))) ensures determinism
    adi = tuple(sorted((i, tuple(sorted(lst))) for i, lst in game_state["available digits per index"].items()))
    scores = tuple(p["score"] for p in game_state["players"])
    turn = int(game_state["turn"])
    return (dyn, adi, scores, turn)

# -------------------------
# Minimax with alpha-beta + transposition + cycle detection
# -------------------------

# Transposition table: maps (hashable_state, maximizing) -> value
TRANSPOSITION: Dict[Tuple, float] = {}

DEPTH_LIMIT = 25  # safety cutoff; increase if you have plenty RAM/CPU

def minimax(hashable_state: Tuple, maximizing: bool, depth: int = 0, alpha: float = -1e9, beta: float = 1e9, visited: frozenset = frozenset()) -> float:
    """
    Minimax with alpha-beta, cycle detection (visited), and transposition table.
    - hashable_state: tuple representation from _get_hashable_state
    - maximizing: True if current node is maximizing player
    - depth: recursion depth (safety)
    - visited: frozenset of states in current path (for cycle detection)
    """
    # cycle detection
    if hashable_state in visited:
        # cycle -> treat as draw/neutral to avoid infinite loops
        return 0.0

    # transposition lookup
    key = (hashable_state, maximizing)
    if key in TRANSPOSITION:
        return TRANSPOSITION[key]

    # safety cutoff
    if depth > DEPTH_LIMIT:
        return 0.0

    # reconstruct small game_state for convenience
    dyn, adi, scores, turn = hashable_state
    available_digits_per_index = {i: list(lst) for i, lst in adi}
    players = [{"id": i, "score": s} for i, s in enumerate(scores)]
    game_state = {
        "dynamic number": dyn,
        "available digits per index": available_digits_per_index,
        "players": players,
        "turn": turn
    }

    # terminal?
    if game_over(game_state):
        val = float("inf") if maximizing else -float("inf")
        TRANSPOSITION[key] = val
        return val

    actions = get_all_legal_actions(game_state)
    if not actions:
        val = -float("inf") if maximizing else float("inf")
        TRANSPOSITION[key] = val
        return val

    # add to visited
    new_visited = visited | {hashable_state}

    if maximizing:
        value = -float("inf")
        for action in actions:
            child_state = simulate_action(action, game_state)
            child_hash = _get_hashable_state(child_state)
            v = minimax(child_hash, False, depth + 1, alpha, beta, new_visited)
            if v > value:
                value = v
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break  # beta cutoff
        TRANSPOSITION[key] = value
        return value
    else:
        value = float("inf")
        for action in actions:
            child_state = simulate_action(action, game_state)
            child_hash = _get_hashable_state(child_state)
            v = minimax(child_hash, True, depth + 1, alpha, beta, new_visited)
            if v < value:
                value = v
            if value < beta:
                beta = value
            if alpha >= beta:
                break  # alpha cutoff
        TRANSPOSITION[key] = value
        return value

# -------------------------
# Helpers to compute best action from initial state (0,0)
# -------------------------

def get_best_action_from_state(game_state: Dict) -> Tuple[Dict, float]:
    actions = get_all_legal_actions(game_state)
    if not actions:
        return None, -float("inf")
    best_val = -float("inf")
    best_action = None
    for action in actions:
        child = simulate_action(action, game_state)
        h = _get_hashable_state(child)
        v = minimax(h, False, depth=1, alpha=-1e9, beta=1e9, visited=frozenset({_get_hashable_state(game_state)}))
        if v > best_val:
            best_val = v
            best_action = action
    return best_action, best_val

# -------------------------
# Main solve loop (2 digits)
# -------------------------

def solve_for_digits(digits: int):
    start = 10**(digits - 1)
    end = 10**digits
    results = {}

    for num in range(start, end):
        s = str(num)
        # initial available digits per index
        adi = setup_available_digit_list_per_index(digits, s)
        initial_state = {
            "dynamic number": s,
            "available digits per index": adi,
            "players": [{"id": 0, "score": 0}, {"id": 1, "score": 0}],
            "turn": 0
        }
        action, val = get_best_action_from_state(initial_state)
        outcome = "WIN" if val == float("inf") else ("LOSE" if val == -float("inf") else "DRAW")
        results[num] = {"best_action": action, "value": val, "outcome": outcome}
        # small progress print
        if num % 10 == 0:
            print(f"digits={digits} solved up to {num} (last outcome {outcome})")
    return results

def main():
    print("Solving Divide21 for 2 digits (starting scores = 0-0). This may take a while...")
    d=2
    TRANSPOSITION.clear()
    start_time = time.perf_counter()
    res = solve_for_digits(d)
    fname = f"divide21_{d}_digits_state_action_lookup.json"
    with open(fname, "w") as f:
        json.dump(res, f, indent=2)
    end_time = time.perf_counter()
    print('Time (s): ' + str(end_time - start_time))

if __name__ == "__main__":
    main()
