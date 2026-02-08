"""
Card Counting Blackjack Environment for Gymnasium.

A custom gym.Env implementation designed to train RL agents to count cards
using the Hi-Lo system with configurable casino rules and continuous shoe logic.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class CardCountingBlackjackEnv(gym.Env):
    """
    Blackjack environment with Hi-Lo card counting and continuous shoe.
    
    Observation Space (Box, shape=(7,)):
        [player_sum, dealer_up_card, usable_ace, true_count, 
         can_split, can_double, can_surrender]
    
    Action Space (Discrete(5)):
        0: Stand, 1: Hit, 2: Double, 3: Split, 4: Surrender
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    # Action constants
    STAND = 0
    HIT = 1
    DOUBLE = 2
    SPLIT = 3
    SURRENDER = 4
    
    def __init__(self, rules: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        super().__init__()
        
        # Default rules
        self.rules = {
            'dealer_hits_soft_17': True,
            'double_after_split': True,
            'surrender': 'Early',  # 'None', 'Late', 'Early'
            'dealer_peeks': False,
            'blackjack_payout': 1.5,
            'num_decks': 6,
            'cut_card_limit': 0.6,
            'max_splits': 3,         # Max re-splits (up to 4 hands total)
            'resplit_aces': False,    # Can you re-split if you get another ace after splitting aces?
            'hit_split_aces': False,  # Can you hit after splitting aces? (False = one card only)
            'surrender_vs_ace': False # Can you surrender when dealer shows an Ace?
        }
        if rules:
            self.rules.update(rules)
        
        self.render_mode = render_mode
        self.num_decks = self.rules['num_decks']
        self.cut_card_limit = self.rules['cut_card_limit']
        
        # Observation space: 7 continuous values
        # [player_sum, dealer_up_card, usable_ace, true_count, can_split, can_double, can_surrender]
        self.observation_space = spaces.Box(
            low=np.array([4, 1, 0, -20, 0, 0, 0], dtype=np.float32),
            high=np.array([31, 10, 1, 20, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Initialize shoe and counting
        self._init_shoe()
        
        # Game state
        self.player_hand = []
        self.dealer_hand = []
        self.is_split_hand = False
        self.doubled = False
        self.first_action = True
        self._hole_card_counted = False  # Track whether dealer hole card has been counted
        
        # Split tracking
        self.split_hands = []       # Queue of hands waiting to be played
        self.completed_hands = []   # Results: list of dicts {hand, value, busted, doubled, surrendered}
        self.num_splits = 0         # How many times the player has split this episode
        self.split_from_aces = False  # True when the current hand(s) came from splitting aces
        
    def _init_shoe(self) -> None:
        """Initialize and shuffle a new shoe of cards."""
        # Card values: 1=Ace, 2-10=face value (10 includes J,Q,K)
        # Each deck has 4 of each card 1-9, and 16 tens (10,J,Q,K)
        single_deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4)
        self.shoe = np.tile(single_deck, self.num_decks)
        np.random.shuffle(self.shoe)
        self.shoe_position = 0
        self.running_count = 0
        
    def _should_reshuffle(self) -> bool:
        """Check if shoe penetration exceeds cut card limit."""
        penetration = self.shoe_position / len(self.shoe)
        return penetration >= self.cut_card_limit
    
    def _deal_card(self, count: bool = True) -> int:
        """Deal a card from the shoe and optionally update running count.
        
        Args:
            count: If True, update the Hi-Lo running count for this card.
                   Set to False for face-down cards (dealer hole card).
        """
        if self.shoe_position >= len(self.shoe):
            self._init_shoe()
            
        card = self.shoe[self.shoe_position]
        self.shoe_position += 1
        if count:
            self._update_count(card)
        return int(card)
    
    def _update_count(self, card: int) -> None:
        """Update running count using Hi-Lo system."""
        if 2 <= card <= 6:
            self.running_count += 1
        elif card >= 10 or card == 1:  # 10s and Aces
            self.running_count -= 1
        # 7, 8, 9 are neutral (0)
    
    def _get_true_count(self) -> float:
        """Calculate true count (running count / remaining decks)."""
        cards_remaining = len(self.shoe) - self.shoe_position
        decks_remaining = max(cards_remaining / 52, 0.5)  # Avoid division issues
        true_count = self.running_count / decks_remaining
        return np.clip(true_count, -20, 20)
    
    def _get_hand_value(self, hand: list) -> Tuple[int, bool]:
        """
        Calculate hand value and whether there's a usable ace.
        
        Returns:
            (total, usable_ace): Total value and whether ace counts as 11
        """
        total = sum(hand)
        usable_ace = False
        
        # Check if we can use an ace as 11
        if 1 in hand and total + 10 <= 21:
            total += 10
            usable_ace = True
            
        return total, usable_ace
    
    def _is_blackjack(self, hand: list) -> bool:
        """Check if hand is a natural blackjack (2 cards, value 21)."""
        if len(hand) != 2:
            return False
        return self._get_hand_value(hand)[0] == 21
    
    def _is_pair(self) -> bool:
        """Check if player has a splittable pair."""
        if len(self.player_hand) != 2:
            return False
        # Cards must have same value (10s are all equal)
        return self.player_hand[0] == self.player_hand[1]
    
    def _get_legal_actions(self) -> Dict[str, bool]:
        """Determine which actions are currently legal."""
        player_sum, _ = self._get_hand_value(self.player_hand)
        
        legal = {
            'stand': True,
            'hit': player_sum < 21,
            'double': False,
            'split': False,
            'surrender': False
        }
        
        # ── Ace-split restriction: one card only, no hit/double ──
        if self.split_from_aces and not self.rules['hit_split_aces']:
            legal['hit'] = False
            legal['double'] = False
            legal['split'] = False
            legal['surrender'] = False
            return legal
        
        # Double: only on first two cards
        if self.first_action and len(self.player_hand) == 2:
            # If split hand, check double_after_split rule
            if self.is_split_hand:
                legal['double'] = self.rules['double_after_split']
            else:
                legal['double'] = True
        
        # Split: only if pair, first action, and haven't exceeded max splits
        if (self.first_action and self._is_pair()
                and self.num_splits < self.rules['max_splits']):
            # Block re-split of aces if rule disallows it
            is_ace_pair = (self.player_hand[0] == 1)
            if is_ace_pair and self.split_from_aces and not self.rules['resplit_aces']:
                legal['split'] = False
            else:
                legal['split'] = True
        
        # Surrender: only on first action of the original hand (not after split)
        if self.first_action and len(self.player_hand) == 2 and not self.is_split_hand:
            surrender_rule = self.rules['surrender']
            
            if surrender_rule in ('Late', 'Early'):
                # Block surrender vs Ace if rule says so
                if self.dealer_hand[0] == 1 and not self.rules['surrender_vs_ace']:
                    legal['surrender'] = False
                else:
                    legal['surrender'] = True
            # 'None': surrender stays False
        
        return legal
    
    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        player_sum, usable_ace = self._get_hand_value(self.player_hand)
        dealer_up = self.dealer_hand[0] if self.dealer_hand else 1
        true_count = self._get_true_count()
        
        legal = self._get_legal_actions()
        
        obs = np.array([
            float(player_sum),
            float(dealer_up),
            float(usable_ace),
            float(true_count),
            float(legal['split']),
            float(legal['double']),
            float(legal['surrender'])
        ], dtype=np.float32)
        
        return obs
    
    def _count_hole_card(self) -> None:
        """Count the dealer's hole card when it is revealed.
        
        Should be called exactly once per hand, when the dealer flips
        over their face-down card (start of dealer play, or on BJ check).
        """
        if not self._hole_card_counted and len(self.dealer_hand) >= 2:
            self._update_count(self.dealer_hand[1])
            self._hole_card_counted = True
    
    def _dealer_play(self) -> int:
        """
        Dealer plays out their hand according to rules.
        
        Returns:
            Final dealer hand value
        """
        # Reveal hole card — count it now
        self._count_hole_card()
        
        while True:
            dealer_sum, usable_ace = self._get_hand_value(self.dealer_hand)
            
            if dealer_sum > 21:
                break
            elif dealer_sum > 17:
                break
            elif dealer_sum == 17:
                # Check H17 rule
                if self.rules['dealer_hits_soft_17'] and usable_ace:
                    self.dealer_hand.append(self._deal_card())
                else:
                    break
            else:
                self.dealer_hand.append(self._deal_card())
        
        return self._get_hand_value(self.dealer_hand)[0]
    
    def _calculate_reward(self, player_sum: int, dealer_sum: int) -> float:
        """Calculate reward based on hand outcomes."""
        multiplier = 2.0 if self.doubled else 1.0
        
        # Player bust
        if player_sum > 21:
            return -1.0 * multiplier
        
        # Dealer bust
        if dealer_sum > 21:
            return 1.0 * multiplier
        
        # Compare hands
        if player_sum > dealer_sum:
            return 1.0 * multiplier
        elif player_sum < dealer_sum:
            return -1.0 * multiplier
        else:
            return 0.0  # Push
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for a new episode.
        
        Reshuffles shoe only if penetration exceeds cut card limit.
        """
        super().reset(seed=seed)
        
        # Check if we need to reshuffle
        if self._should_reshuffle():
            self._init_shoe()
        
        # Reset game state
        self.player_hand = []
        self.dealer_hand = []
        self.is_split_hand = False
        self.doubled = False
        self.first_action = True
        self._hole_card_counted = False
        self.split_hands = []
        self.completed_hands = []
        self.num_splits = 0
        self.split_from_aces = False
        
        # Deal initial cards: player, dealer_up, player, dealer_hole
        # The hole card is face-down — a real card counter can't see it,
        # so we do NOT update the running count for the hole card yet.
        # It will be counted when the dealer reveals it during play.
        self.player_hand.append(self._deal_card())           # player card 1 (visible)
        self.dealer_hand.append(self._deal_card())           # dealer up card (visible)
        self.player_hand.append(self._deal_card())           # player card 2 (visible)
        self.dealer_hand.append(self._deal_card(count=False))  # dealer hole card (hidden)
        
        # Check for dealer blackjack if peek rule is enabled
        if self.rules['dealer_peeks'] and self._is_blackjack(self.dealer_hand):
            # Dealer has blackjack - episode ends immediately
            player_bj = self._is_blackjack(self.player_hand)
            reward = 0.0 if player_bj else -1.0
            
            info = {
                'player_hand': self.player_hand.copy(),
                'dealer_hand': self.dealer_hand.copy(),
                'dealer_blackjack': True,
                'player_blackjack': player_bj,
                'running_count': self.running_count,
                'true_count': self._get_true_count()
            }
            
            # Return with terminated=True through step-like response
            # But reset should return obs, info - we handle BJ in first step
            pass
        
        # Check for player blackjack
        info = {
            'player_hand': self.player_hand.copy(),
            'dealer_hand': [self.dealer_hand[0]],  # Only show up card
            'running_count': self.running_count,
            'true_count': self._get_true_count()
        }
        
        return self._get_obs(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute an action.
        
        Args:
            action: 0=Stand, 1=Hit, 2=Double, 3=Split, 4=Surrender
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        truncated = False
        
        legal = self._get_legal_actions()
        player_sum, _ = self._get_hand_value(self.player_hand)
        
        # ── Blackjack checks (only on the very first action of the original hand) ──
        if self.first_action and len(self.player_hand) == 2 and not self.is_split_hand:
            # Dealer blackjack with peek
            if self.rules['dealer_peeks'] and self._is_blackjack(self.dealer_hand):
                self._count_hole_card()  # Reveal hole card for BJ
                player_bj = self._is_blackjack(self.player_hand)
                reward = 0.0 if player_bj else -1.0
                return self._get_obs(), reward, True, False, {
                    'player_hand': self.player_hand.copy(),
                    'dealer_hand': self.dealer_hand.copy(),
                    'outcome': 'push' if player_bj else 'dealer_blackjack',
                    'running_count': self.running_count,
                    'true_count': self._get_true_count()
                }
            
            # Player natural blackjack
            if self._is_blackjack(self.player_hand):
                self._count_hole_card()  # Reveal hole card
                dealer_sum = self._dealer_play()
                if self._is_blackjack(self.dealer_hand):
                    reward = 0.0
                else:
                    reward = self.rules['blackjack_payout']
                return self._get_obs(), reward, True, False, {
                    'player_hand': self.player_hand.copy(),
                    'dealer_hand': self.dealer_hand.copy(),
                    'outcome': 'blackjack' if reward > 0 else 'push',
                    'running_count': self.running_count,
                    'true_count': self._get_true_count()
                }
        
        # ── Validate action (invalid → Stand + penalty) ──
        illegal_penalty = 0.0
        if action == self.DOUBLE and not legal['double']:
            illegal_penalty = -0.2
            action = self.STAND
        elif action == self.SPLIT and not legal['split']:
            illegal_penalty = -0.2
            action = self.STAND
        elif action == self.SURRENDER and not legal['surrender']:
            illegal_penalty = -0.2
            action = self.STAND
        elif action == self.HIT and not legal['hit']:
            illegal_penalty = -0.2
            action = self.STAND
        
        # ── Execute action ──
        hand_done = False  # True when current hand is resolved
        
        if action == self.STAND:
            hand_done = True
            
        elif action == self.HIT:
            self.first_action = False
            self.player_hand.append(self._deal_card())
            player_sum, _ = self._get_hand_value(self.player_hand)
            if player_sum >= 21:  # bust or 21 → hand auto-finishes
                hand_done = True
                
        elif action == self.DOUBLE:
            self.doubled = True
            self.first_action = False
            self.player_hand.append(self._deal_card())
            hand_done = True  # Only one card on double
            
        elif action == self.SPLIT:
            self.num_splits += 1
            self.is_split_hand = True
            
            # Split the pair into two hands
            card_a = self.player_hand[0]
            card_b = self.player_hand[1]
            splitting_aces = (card_a == 1)
            
            if splitting_aces:
                self.split_from_aces = True
            
            # Build hand A (current) and hand B (queued)
            hand_a = [card_a, self._deal_card()]
            hand_b = [card_b, self._deal_card()]
            
            # ── Split aces with one-card-only rule ──
            if splitting_aces and not self.rules['hit_split_aces']:
                # Both hands auto-stand immediately (no player action)
                self.player_hand = hand_a
                self.split_hands.append(hand_b)
                self.doubled = False
                # Auto-finish hand A → triggers hand B → triggers resolve
                return self._finish_current_hand(self.STAND)
            
            # ── Normal split ──
            self.player_hand = hand_a
            self.split_hands.append(hand_b)
            self.first_action = True
            self.doubled = False
            
            # If hand A is 21, auto-finish it immediately
            if self._get_hand_value(self.player_hand)[0] == 21:
                hand_done = True
            else:
                # Hand A continues — agent keeps acting
                info = self._build_info(action, terminated=False)
                return self._get_obs(), 0.0, False, False, info
                
        elif action == self.SURRENDER:
            hand_done = True  # Flagged as surrendered in _finish_current_hand
        
        # ── If current hand is resolved, run the finish/transition logic ──
        if hand_done:
            obs, reward, terminated, truncated, info = self._finish_current_hand(action)
            reward += illegal_penalty
            if illegal_penalty != 0.0:
                info['illegal_action_penalty'] = illegal_penalty
            return obs, reward, terminated, truncated, info
        
        # Hand still in play (hit without bust)
        self.first_action = False
        info = self._build_info(action, terminated=False)
        return self._get_obs(), 0.0, False, False, info
    
    # ──────────────────────────────────────────────────────────────────
    #  Split helpers
    # ──────────────────────────────────────────────────────────────────
    
    def _finish_current_hand(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Record the result of the current hand and either transition to the
        next split hand or resolve all hands against the dealer.
        """
        player_sum, _ = self._get_hand_value(self.player_hand)
        
        self.completed_hands.append({
            'hand': self.player_hand.copy(),
            'value': player_sum,
            'busted': player_sum > 21,
            'doubled': self.doubled,
            'surrendered': (action == self.SURRENDER)
        })
        
        # If there are more split hands to play, transition to the next one
        if self.split_hands:
            self.player_hand = self.split_hands.pop(0)
            self.first_action = True
            self.doubled = False
            
            # Auto-finish if next hand already totals 21
            if self._get_hand_value(self.player_hand)[0] == 21:
                return self._finish_current_hand(self.STAND)
            
            # Auto-finish if split aces with one-card-only rule
            if self.split_from_aces and not self.rules['hit_split_aces']:
                return self._finish_current_hand(self.STAND)
            
            info = self._build_info(action, terminated=False)
            return self._get_obs(), 0.0, False, False, info
        
        # All hands played — resolve against dealer
        return self._resolve_all_hands(action)
    
    def _resolve_all_hands(self, last_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Dealer plays once, then compute the combined reward for every
        completed hand.
        """
        # Dealer plays only if at least one hand is still alive
        any_alive = any(
            not h['busted'] and not h['surrendered']
            for h in self.completed_hands
        )
        
        # Always reveal (count) the hole card at hand end for accurate
        # running count in the continuous shoe.
        self._count_hole_card()
        
        if any_alive:
            dealer_sum = self._dealer_play()
        else:
            dealer_sum = self._get_hand_value(self.dealer_hand)[0]
        
        total_reward = 0.0
        hand_results = []
        
        for h in self.completed_hands:
            if h['surrendered']:
                total_reward -= 0.5
                hand_results.append({'hand': h['hand'], 'outcome': 'surrender', 'reward': -0.5})
                continue
            
            multiplier = 2.0 if h['doubled'] else 1.0
            pv = h['value']
            
            if pv > 21:
                r = -1.0 * multiplier
                outcome = 'bust'
            elif dealer_sum > 21:
                r = 1.0 * multiplier
                outcome = 'win'
            elif pv > dealer_sum:
                r = 1.0 * multiplier
                outcome = 'win'
            elif pv < dealer_sum:
                r = -1.0 * multiplier
                outcome = 'lose'
            else:
                r = 0.0
                outcome = 'push'
            
            total_reward += r
            hand_results.append({'hand': h['hand'], 'outcome': outcome, 'reward': r})
        
        info = {
            'all_hands': hand_results,
            'dealer_hand': self.dealer_hand.copy(),
            'dealer_sum': dealer_sum,
            'total_reward': total_reward,
            'num_splits': self.num_splits,
            'action_taken': ['stand', 'hit', 'double', 'split', 'surrender'][last_action],
            'running_count': self.running_count,
            'true_count': self._get_true_count(),
            'penetration': self.shoe_position / len(self.shoe)
        }
        
        return self._get_obs(), total_reward, True, False, info
    
    def _build_info(self, action: int, terminated: bool) -> Dict:
        """Build the standard info dict for non-terminal steps."""
        return {
            'player_hand': self.player_hand.copy(),
            'dealer_hand': self.dealer_hand.copy() if terminated else [self.dealer_hand[0]],
            'action_taken': ['stand', 'hit', 'double', 'split', 'surrender'][action],
            'running_count': self.running_count,
            'true_count': self._get_true_count(),
            'penetration': self.shoe_position / len(self.shoe),
            'hands_remaining': len(self.split_hands),
            'hands_completed': len(self.completed_hands)
        }
    
    def render(self) -> None:
        """Render the current game state."""
        if self.render_mode != "human":
            return
            
        player_sum, usable_ace = self._get_hand_value(self.player_hand)
        dealer_up = self.dealer_hand[0]
        
        print(f"\n{'='*40}")
        if self.num_splits > 0:
            # Show completed hands
            for i, h in enumerate(self.completed_hands):
                status = "BUST" if h['busted'] else ("SURR" if h['surrendered'] else f"{h['value']}")
                dbl = " (2x)" if h['doubled'] else ""
                print(f"  Hand {i+1}: {h['hand']} = {status}{dbl}  [done]")
            # Show current hand
            idx = len(self.completed_hands) + 1
            print(f"► Hand {idx}: {self.player_hand} = {player_sum}" +
                  (" (soft)" if usable_ace else "") +
                  ("  [playing]" if self.split_hands or True else ""))
            # Show queued hands
            for j, qh in enumerate(self.split_hands):
                qv, _ = self._get_hand_value(qh)
                print(f"  Hand {idx+j+1}: {qh} = {qv}  [waiting]")
        else:
            print(f"Player Hand: {self.player_hand} = {player_sum}" +
                  (" (soft)" if usable_ace else ""))
        print(f"Dealer Shows: {dealer_up}")
        print(f"Running Count: {self.running_count}")
        print(f"True Count: {self._get_true_count():.2f}")
        print(f"Penetration: {self.shoe_position / len(self.shoe) * 100:.1f}%")
        print(f"{'='*40}")
    
    def close(self) -> None:
        """Clean up resources."""
        pass


# Register the environment
if __name__ == "__main__":
    # Quick test
    env = CardCountingBlackjackEnv(render_mode="human")
    
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    env.render()
    
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Action: {action}, Reward: {reward}")
        env.render()
    
    print(f"\nFinal info: {info}")
    env.close()
