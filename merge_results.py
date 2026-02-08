"""
Merge multiple Monte Carlo worker pickle files into a single combined result.

For each state+action pair the merge sums ev_sums and ev_counts across all
workers.  If an existing combined pickle already exists, the new worker data
is merged into it (incremental accumulation).

Usage:
    python merge_results.py --input-dir results --output results/mc_ev_combined.pkl
    python merge_results.py --input-dir results --output results/mc_ev_combined.pkl \
                            --existing results/mc_ev_combined.pkl
"""

import argparse
import glob
import os
import pickle
from collections import defaultdict


# ─────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────

def load_worker_pickle(path: str):
    """Load a single worker checkpoint pickle.

    Returns (ev_sums, ev_counts, total_actions_tested, hands_done).
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    ev_sums = defaultdict(lambda: defaultdict(float))
    ev_counts = defaultdict(lambda: defaultdict(int))

    for s, actions in data['ev_sums'].items():
        for a, val in actions.items():
            ev_sums[s][a] = val
    for s, actions in data['ev_counts'].items():
        for a, val in actions.items():
            ev_counts[s][a] = val

    return (ev_sums, ev_counts,
            data.get('total_actions_tested', 0),
            data.get('hands_done', 0))


def load_combined_pickle(path: str):
    """Load an existing combined pickle (same format as worker checkpoint)."""
    return load_worker_pickle(path)


# ─────────────────────────────────────────────
# Merge logic
# ─────────────────────────────────────────────

def merge_into(target_sums, target_counts, source_sums, source_counts):
    """Merge source accumulators into target (in-place).

    For Monte Carlo EV estimation:
        EV(s,a) = sum_of_rewards / count
    Merging by summing both numerator and denominator is mathematically
    correct for combining independent samples.
    """
    for state in source_sums:
        for action in source_sums[state]:
            target_sums[state][action] += source_sums[state][action]
            target_counts[state][action] += source_counts[state][action]


def build_ev_table(ev_sums, ev_counts):
    """Derive the mean EV table from raw accumulators."""
    ev_table = {}
    for state in ev_sums:
        ev_table[state] = {}
        for action in ev_sums[state]:
            n = ev_counts[state][action]
            if n > 0:
                ev_table[state][action] = ev_sums[state][action] / n
    return ev_table


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge Monte Carlo worker pickles into one combined result")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing mc_worker_*.pkl files")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for the combined output pickle")
    parser.add_argument("--existing", type=str, default=None,
                        help="Path to existing combined pickle to merge into "
                             "(if not set, checks if --output already exists)")
    args = parser.parse_args()

    # ── Discover worker files ──
    pattern = os.path.join(args.input_dir, "mc_worker_*.pkl")
    worker_files = sorted(glob.glob(pattern))

    if not worker_files:
        print(f"No worker files found matching {pattern}")
        return

    print(f"Found {len(worker_files)} worker file(s):")
    for wf in worker_files:
        print(f"  - {wf}")

    # ── Initialize accumulators ──
    combined_sums = defaultdict(lambda: defaultdict(float))
    combined_counts = defaultdict(lambda: defaultdict(int))
    total_actions = 0
    total_hands = 0

    # ── Load existing combined pickle if available ──
    existing_path = args.existing or args.output
    if os.path.exists(existing_path):
        print(f"\nLoading existing combined results from {existing_path}")
        ex_sums, ex_counts, ex_actions, ex_hands = load_combined_pickle(existing_path)
        merge_into(combined_sums, combined_counts, ex_sums, ex_counts)
        total_actions += ex_actions
        total_hands += ex_hands
        print(f"  Existing: {len(ex_counts):,} states, "
              f"{ex_hands:,} hands, {ex_actions:,} actions")

    # ── Merge each worker ──
    for wf in worker_files:
        print(f"\nMerging {wf} ...")
        w_sums, w_counts, w_actions, w_hands = load_worker_pickle(wf)
        merge_into(combined_sums, combined_counts, w_sums, w_counts)
        total_actions += w_actions
        total_hands += w_hands
        print(f"  Worker: {len(w_counts):,} states, "
              f"{w_hands:,} hands, {w_actions:,} actions")

    # ── Save combined checkpoint ──
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    combined_data = {
        'ev_sums': {s: dict(a) for s, a in combined_sums.items()},
        'ev_counts': {s: dict(a) for s, a in combined_counts.items()},
        'total_actions_tested': total_actions,
        'hands_done': total_hands,
    }
    with open(args.output, 'wb') as f:
        pickle.dump(combined_data, f)

    print(f"\n{'='*60}")
    print(f"Combined results saved to {args.output}")
    print(f"  Total states:  {len(combined_counts):,}")
    print(f"  Total hands:   {total_hands:,}")
    print(f"  Total actions: {total_actions:,}")

    # ── Also save the derived EV table ──
    ev_table = build_ev_table(combined_sums, combined_counts)
    ev_table_path = args.output.replace('.pkl', '_ev_table.pkl')
    with open(ev_table_path, 'wb') as f:
        pickle.dump(ev_table, f)
    print(f"  EV table:      {ev_table_path} ({len(ev_table):,} states)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
