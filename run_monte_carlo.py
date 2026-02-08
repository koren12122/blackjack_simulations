"""
Monte Carlo EV worker for parallel GitHub Actions runs.

Each worker runs the full MC simulation independently and saves
its own checkpoint pickle. A separate merge script combines all
worker results into a single pickle.

Usage:
    python run_monte_carlo.py --worker-id 0 --total-hands 3500000 --output-dir results
"""

import argparse
import os
import pickle
import numpy as np
from collections import defaultdict
from card_counting_blackjack_env import CardCountingBlackjackEnv

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DEFAULT_TOTAL_HANDS = 3_500_000
LOG_INTERVAL = 100_000

TC_MIN, TC_MAX = -5, 5

ACTION_NAMES = {0: 'S', 1: 'H', 2: 'D', 3: 'P', 4: 'R'}

RULES = {
    'surrender': 'Early',
    'dealer_hits_soft_17': False,
    'dealer_peeks': False,
    'blackjack_payout': 1.5,
    'num_decks': 6,
    'cut_card_limit': 0.9,
}

# ─────────────────────────────────────────────
# Helpers (same as monte_carlo_ev.py)
# ─────────────────────────────────────────────

def discretize_obs(obs: np.ndarray) -> tuple:
    """Convert continuous observation to discrete state tuple."""
    player_sum = int(obs[0])
    dealer_up = int(obs[1])
    usable_ace = int(obs[2])
    true_count = int(np.clip(round(obs[3]), TC_MIN, TC_MAX))
    can_split = int(obs[4])
    can_double = int(obs[5])
    can_surrender = int(obs[6])
    return (player_sum, dealer_up, usable_ace, true_count,
            can_split, can_double, can_surrender)


def get_legal_action_ids(obs: np.ndarray) -> list:
    """Return list of legal action IDs from the observation flags."""
    actions = [0, 1]  # Stand + Hit always legal
    player_sum = int(obs[0])
    if player_sum >= 21:
        actions = [0]  # Only stand
    if obs[5] > 0.5:   # can_double
        actions.append(2)
    if obs[4] > 0.5:   # can_split
        actions.append(3)
    if obs[6] > 0.5:   # can_surrender
        actions.append(4)
    return actions


def _lookup_ev_action(state, legal, ev_sums, ev_counts, min_n=20):
    """Return the legal action with highest EV, or None if insufficient data."""
    if state not in ev_counts:
        return None
    best_action = None
    best_ev = -np.inf
    for a in legal:
        n = ev_counts[state].get(a, 0)
        if n >= min_n:
            ev = ev_sums[state][a] / n
            if ev > best_ev:
                best_ev = ev
                best_action = a
    return best_action


def _basic_strategy_action(obs: np.ndarray) -> int:
    """Approximate basic strategy for hit/stand decisions (fallback)."""
    player_sum = int(obs[0])
    dealer_up  = int(obs[1])
    usable_ace = int(obs[2])

    if usable_ace:
        if player_sum >= 19:
            return 0
        if player_sum == 18:
            if dealer_up >= 9 or dealer_up == 1:
                return 1
            return 0
        return 1

    if player_sum >= 17:
        return 0
    if player_sum <= 11:
        return 1

    if player_sum == 12:
        if 4 <= dealer_up <= 6:
            return 0
        return 1
    if 2 <= dealer_up <= 6:
        return 0
    return 1


def follow_up_action(obs, ev_sums, ev_counts):
    """Pick the best known action for the current state."""
    state = discretize_obs(obs)
    legal = get_legal_action_ids(obs)

    best = _lookup_ev_action(state, legal, ev_sums, ev_counts)
    if best is not None:
        return best

    psum, dup, ua, tc, csplit, cdbl, csurr = state
    for cd in (0, 1):
        for cs in (0, 1):
            alt = (psum, dup, ua, tc, csplit, cd, cs)
            if alt == state:
                continue
            best = _lookup_ev_action(alt, legal, ev_sums, ev_counts)
            if best is not None:
                return best

    return _basic_strategy_action(obs)


def play_out(env, obs, ev_sums, ev_counts) -> float:
    """Play out a hand from the current state to termination."""
    total_reward = 0.0
    done = False
    while not done:
        action = follow_up_action(obs, ev_sums, ev_counts)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


# ─────────────────────────────────────────────
# Lightweight snapshot / restore
# ─────────────────────────────────────────────

def _snapshot(env):
    """Capture the minimal mutable state of the env."""
    return (
        env.shoe.copy(),
        env.shoe_position,
        env.running_count,
        env.player_hand.copy(),
        env.dealer_hand.copy(),
        env.is_split_hand,
        env.doubled,
        env.first_action,
        [h[:] for h in env.split_hands],
        [{k: (v[:] if isinstance(v, list) else v) for k, v in h.items()}
         for h in env.completed_hands],
        env.num_splits,
        env.split_from_aces,
        env._hole_card_counted,
    )


def _restore(env, snap):
    """Restore env to a previously captured snapshot."""
    (shoe, shoe_pos, running_count, player_hand, dealer_hand,
     is_split_hand, doubled, first_action,
     split_hands, completed_hands,
     num_splits, split_from_aces, hole_card_counted) = snap

    env.shoe = shoe.copy()
    env.shoe_position = shoe_pos
    env.running_count = running_count
    env.player_hand = player_hand[:]
    env.dealer_hand = dealer_hand[:]
    env.is_split_hand = is_split_hand
    env.doubled = doubled
    env.first_action = first_action
    env.split_hands = [h[:] for h in split_hands]
    env.completed_hands = [{k: (v[:] if isinstance(v, list) else v) for k, v in h.items()}
                           for h in completed_hands]
    env.num_splits = num_splits
    env.split_from_aces = split_from_aces
    env._hole_card_counted = hole_card_counted


# ─────────────────────────────────────────────
# Save worker checkpoint
# ─────────────────────────────────────────────

def save_worker_checkpoint(ev_sums, ev_counts, total_actions_tested,
                           hands_done, output_path):
    """Save raw accumulators for this worker."""
    data = {
        'ev_sums': {s: dict(a) for s, a in ev_sums.items()},
        'ev_counts': {s: dict(a) for s, a in ev_counts.items()},
        'total_actions_tested': total_actions_tested,
        'hands_done': hands_done,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Worker checkpoint saved to {output_path}")


# ─────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────

def run_worker(worker_id: int, total_hands: int, output_dir: str):
    """Run a single Monte Carlo worker."""

    # Seed based on worker ID for diverse shoe shuffles
    np.random.seed(worker_id * 12345 + 42)

    env = CardCountingBlackjackEnv(rules=RULES)

    ev_sums = defaultdict(lambda: defaultdict(float))
    ev_counts = defaultdict(lambda: defaultdict(int))
    total_actions_tested = 0

    print(f"[Worker {worker_id}] Monte Carlo EV — {total_hands:,} hands")
    print(f"[Worker {worker_id}] Rules: {RULES}")
    print(f"[Worker {worker_id}] TC range: [{TC_MIN}, {TC_MAX}]\n")

    for hand_idx in range(total_hands):
        obs, info = env.reset()
        state = discretize_obs(obs)
        legal_actions = get_legal_action_ids(obs)

        snap = _snapshot(env)

        for action in legal_actions:
            _restore(env, snap)
            obs_a, reward_a, terminated, truncated, _ = env.step(action)
            total_reward = reward_a

            if not (terminated or truncated):
                total_reward += play_out(env, obs_a, ev_sums, ev_counts)

            ev_sums[state][action] += total_reward
            ev_counts[state][action] += 1
            total_actions_tested += 1

        # Advance the real shoe
        _restore(env, snap)
        done = False
        while not done:
            action = follow_up_action(obs, ev_sums, ev_counts)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if (hand_idx + 1) % LOG_INTERVAL == 0:
            n_states = len(ev_counts)
            print(
                f"[Worker {worker_id}] Hand {hand_idx + 1:>10,} | "
                f"States: {n_states:,} | "
                f"Actions tested: {total_actions_tested:,}"
            )

    env.close()

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"mc_worker_{worker_id}.pkl")
    save_worker_checkpoint(ev_sums, ev_counts, total_actions_tested,
                           total_hands, output_path)

    print(f"\n[Worker {worker_id}] Done. {len(ev_counts):,} states, "
          f"{total_actions_tested:,} action samples.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo EV worker")
    parser.add_argument("--worker-id", type=int, required=True,
                        help="Unique worker ID (used for seeding)")
    parser.add_argument("--total-hands", type=int, default=DEFAULT_TOTAL_HANDS,
                        help=f"Number of hands to simulate (default: {DEFAULT_TOTAL_HANDS:,})")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save worker pickle (default: results)")
    args = parser.parse_args()

    run_worker(args.worker_id, args.total_hands, args.output_dir)
