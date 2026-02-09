"""
DQN training + evaluation script for GitHub Actions.

Trains a DQN agent on the CardCountingBlackjackEnv, evaluates it
by bucketing average reward by pre-deal true count, and saves the
evaluation results as a pickle. If an existing results pickle is
present, merges evaluation counts into it.

Usage:
    python run_dqn.py --output-dir results
    python run_dqn.py --output-dir results --timesteps 2000000 --eval-hands 300000
"""

import argparse
import os
import pickle
import numpy as np
from collections import defaultdict
from stable_baselines3 import DQN
from card_counting_blackjack_env import CardCountingBlackjackEnv

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DEFAULT_TIMESTEPS = 2_000_000
DEFAULT_EVAL_HANDS = 300_000

RULES = {
    'surrender': 'Early',
    'dealer_hits_soft_17': False,
    'dealer_peeks': False,
    'blackjack_payout': 1.5,
    'num_decks': 6,
    'cut_card_limit': 0.9,
}


# ─────────────────────────────────────────────
# Helpers for fair evaluation
# ─────────────────────────────────────────────

def _hi_lo_effect(card: int) -> int:
    """Hi-Lo count value for a single card."""
    if 2 <= card <= 6:
        return 1
    if card >= 10 or card == 1:
        return -1
    return 0


def _pre_deal_tc(env) -> float:
    """
    True count BEFORE the current hand's cards were dealt.
    This is the count a card counter would use for bet sizing.
    """
    rc_pre = env.running_count
    for card in env.player_hand:
        rc_pre -= _hi_lo_effect(card)
    rc_pre -= _hi_lo_effect(env.dealer_hand[0])
    remaining_pre = len(env.shoe) - env.shoe_position + 4
    decks_pre = max(remaining_pre / 52, 0.5)
    return rc_pre / decks_pre


# ─────────────────────────────────────────────
# Merge logic for DQN evaluation results
# ─────────────────────────────────────────────

def merge_dqn_results(existing_path: str, new_buckets: dict) -> dict:
    """
    Merge new tc_buckets into an existing results pickle.

    Each bucket is tc -> [total_reward, hand_count].
    Merging sums rewards and counts.
    """
    if os.path.exists(existing_path):
        with open(existing_path, 'rb') as f:
            existing = pickle.load(f)
        print(f"Merging with existing results from {existing_path}")
        for tc, (rw, cnt) in new_buckets.items():
            if tc in existing:
                existing[tc][0] += rw
                existing[tc][1] += cnt
            else:
                existing[tc] = [rw, cnt]
        return existing
    else:
        return dict(new_buckets)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_dqn(timesteps: int, eval_hands: int, output_dir: str, checkpoint: str = None):
    env = CardCountingBlackjackEnv(rules=RULES)

    print("Starting DQN training...")
    print(f"Rules: {RULES}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Eval hands: {eval_hands:,}")
    if checkpoint:
        print(f"Checkpoint: {checkpoint}")
    print()

    # ── Train ──
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint from {checkpoint}...")
        model = DQN.load(checkpoint, env=env, device="auto")
        print("Checkpoint loaded successfully.")
        print("Continuing training...\n")
    else:
        if checkpoint:
            print(f"Warning: Checkpoint {checkpoint} not found. Training from scratch.\n")
        policy_kwargs = dict(net_arch=[256, 256])
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
            buffer_size=50_000,
            learning_starts=1000,
            exploration_fraction=0.2,
            target_update_interval=250,
            learning_rate=0.0005,
            device="auto",
        )
    
    model.learn(total_timesteps=timesteps)
    print("Training finished.\n")


    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "dqn_blackjack_counter")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # ── Evaluate ──
    print(f"\nEvaluating strategy per Pre-Deal True Count ({eval_hands:,} hands)...")

    tc_buckets = defaultdict(lambda: [0.0, 0])
    obs, _ = env.reset()

    for _ in range(eval_hands):
        pre_tc = _pre_deal_tc(env)
        tc_at_start = int(np.clip(round(pre_tc), -5, 5))

        hand_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            hand_reward += reward
            done = terminated or truncated

        tc_buckets[tc_at_start][0] += hand_reward
        tc_buckets[tc_at_start][1] += 1

        if done:
            obs, _ = env.reset()

    env.close()

    # ── Print results ──
    print(f"\n{'TC':>4} | {'Hands':>9} | {'Avg EV':>9} | {'Cumul. EV':>11}")
    print("-" * 42)

    total_reward = 0.0
    total_hands = 0

    for tc in sorted(tc_buckets.keys()):
        rw, cnt = tc_buckets[tc]
        avg = rw / cnt if cnt > 0 else 0.0
        total_reward += rw
        total_hands += cnt
        cum_avg = total_reward / total_hands if total_hands > 0 else 0.0
        print(f"{tc:>+4d} | {cnt:>9,} | {avg:>+9.5f} | {cum_avg:>+11.5f}")

    overall = total_reward / total_hands if total_hands > 0 else 0.0
    print("-" * 42)
    print(f"{'ALL':>4} | {total_hands:>9,} | {overall:>+9.5f} |")

    # ── Save / merge results ──
    results_path = os.path.join(output_dir, "dqn_eval_results.pkl")
    # Convert defaultdict to regular dict for serialization
    new_buckets = {tc: list(vals) for tc, vals in tc_buckets.items()}

    merged = merge_dqn_results(results_path, new_buckets)

    with open(results_path, 'wb') as f:
        pickle.dump(merged, f)
    print(f"\nEvaluation results saved to {results_path}")

    return merged, model  # Return model for extended evaluation


def extended_evaluation(model, num_hands: int, output_dir: str):
    """Run extended evaluation with detailed statistics."""
    env = CardCountingBlackjackEnv(rules=RULES)
    
    print(f"\n{'='*80}")
    print(f"EXTENDED EVALUATION ({num_hands:,} hands)")
    print(f"{'='*80}\n")
    
    tc_buckets = defaultdict(lambda: [0.0, 0])
    action_counts = defaultdict(int)
    action_names = {0: 'Stand', 1: 'Hit', 2: 'Double', 3: 'Split', 4: 'Surrender'}
    hand_rewards = []
    obs, _ = env.reset()

    for hand_idx in range(num_hands):
        pre_tc = _pre_deal_tc(env)
        tc_at_start = int(np.clip(round(pre_tc), -5, 5))

        hand_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            hand_reward += reward
            done = terminated or truncated

        tc_buckets[tc_at_start][0] += hand_reward
        tc_buckets[tc_at_start][1] += 1
        hand_rewards.append(hand_reward)

        if done:
            obs, _ = env.reset()
        
        if (hand_idx + 1) % 50000 == 0:
            current_ev = sum(hand_rewards) / len(hand_rewards)
            print(f"  Hand {hand_idx + 1:>10,} | Running EV: {current_ev:>+8.5f}")

    env.close()

    # Calculate statistics
    overall_ev = sum(hand_rewards) / len(hand_rewards)
    std_dev = np.std(hand_rewards)
    std_error = std_dev / np.sqrt(num_hands)

    # Print detailed results
    print(f"\n{'OVERALL RESULTS':-^80}")
    print(f"Total hands: {num_hands:,}")
    print(f"EV per hand: {overall_ev:+.5f}")
    print(f"Std deviation: {std_dev:.5f}")
    print(f"Std error: {std_error:.5f}")
    print(f"95% CI: [{overall_ev - 1.96*std_error:+.5f}, {overall_ev + 1.96*std_error:+.5f}]")
    
    ev_pct = overall_ev * 100
    ci_lower = (overall_ev - 1.96*std_error) * 100
    ci_upper = (overall_ev + 1.96*std_error) * 100
    print(f"House edge: {-ev_pct:+.3f}% (95% CI: [{-ci_upper:+.3f}%, {-ci_lower:+.3f}%])")

    # Action distribution
    print(f"\n{'ACTION DISTRIBUTION':-^80}")
    total_actions = sum(action_counts.values())
    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        pct = (count / total_actions) * 100
        action_name = action_names.get(action_id, f"Action {action_id}")
        print(f"{action_name:<12} | {count:>12,} | {pct:>9.2f}%")

    # Results by TC
    print(f"\n{'RESULTS BY TRUE COUNT':-^80}")
    print(f"{'TC':>4} | {'Hands':>10} | {'Total Reward':>12} | {'EV per Hand':>11} | {'House Edge':>11}")
    print("-" * 80)
    
    for tc in sorted(tc_buckets.keys()):
        rw, cnt = tc_buckets[tc]
        tc_ev = rw / cnt if cnt > 0 else 0
        tc_he = -tc_ev * 100
        print(f"{tc:>+4d} | {cnt:>10,} | {rw:>+12.2f} | {tc_ev:>+11.5f} | {tc_he:>+10.3f}%")
    
    print("=" * 80)
    
    # Save extended results
    ext_results_path = os.path.join(output_dir, "dqn_extended_eval.pkl")
    ext_results = {
        'overall_ev': overall_ev,
        'std_error': std_error,
        'tc_results': dict(tc_buckets),
        'action_counts': dict(action_counts),
        'num_hands': num_hands
    }
    with open(ext_results_path, 'wb') as f:
        pickle.dump(ext_results, f)
    print(f"\nExtended evaluation results saved to {ext_results_path}\n")

    return ext_results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Blackjack training + evaluation")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help=f"DQN training timesteps (default: {DEFAULT_TIMESTEPS:,})")
    parser.add_argument("--eval-hands", type=int, default=DEFAULT_EVAL_HANDS,
                        help=f"Evaluation hands (default: {DEFAULT_EVAL_HANDS:,})")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results (default: results)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint model to continue training from")
    parser.add_argument("--extended-eval", type=int, default=None,
                        help="Run extended evaluation with this many hands after training")
    args = parser.parse_args()

    merged, model = run_dqn(args.timesteps, args.eval_hands, args.output_dir, args.checkpoint)
    
    # Run extended evaluation if requested
    if args.extended_eval:
        extended_evaluation(model, args.extended_eval, args.output_dir)
