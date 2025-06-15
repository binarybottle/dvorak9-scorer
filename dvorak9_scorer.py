# dvorak9_scorer.py
"""
Dvorak-9 scoring model implementation for keyboard layout evaluation.

This script implements the 9 evaluation criteria derived from Dvorak's "Typing Behavior" 
book and patent (1936) for evaluating keyboard layouts based on bigram typing behavior.

The 9 scoring criteria for typing bigrams are:
1. Hands - favor alternating hands over same hand
2. Fingers - avoid same finger repetition  
3. Skip fingers - favor non-adjacent fingers over adjacent (same hand)
4. Don't cross home - avoid crossing over the home row (hurdling)
5. Same row - favor typing within the same row
6. Home row - favor using the home row
7. Columns - favor fingers staying in their designated columns
8. Strum - favor inward rolls over outward rolls (same hand)
9. Strong fingers - favor stronger fingers over weaker ones

Example usage:
    python dvorak9_scorer.py --items "abc" --positions "FDJ" --text "abacaba"
    python dvorak9_scorer.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details
"""

import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# QWERTY keyboard layout with (row, finger, hand) mapping
# Rows: 0=number, 1=top, 2=home, 3=bottom
# Fingers: 1=index, 2=middle, 3=ring, 4=pinky  
# Hands: L=left, R=right
QWERTY_LAYOUT = {
    # Number row (row 0)
    '1': (0, 4, 'L'), '2': (0, 3, 'L'), '3': (0, 2, 'L'), '4': (0, 1, 'L'), '5': (0, 1, 'L'),
    '6': (0, 1, 'R'), '7': (0, 1, 'R'), '8': (0, 2, 'R'), '9': (0, 3, 'R'), '0': (0, 4, 'R'),
    
    # Top row (row 1)
    'Q': (1, 4, 'L'), 'W': (1, 3, 'L'), 'E': (1, 2, 'L'), 'R': (1, 1, 'L'), 'T': (1, 1, 'L'),
    'Y': (1, 1, 'R'), 'U': (1, 1, 'R'), 'I': (1, 2, 'R'), 'O': (1, 3, 'R'), 'P': (1, 4, 'R'),
    
    # Home row (row 2) 
    'A': (2, 4, 'L'), 'S': (2, 3, 'L'), 'D': (2, 2, 'L'), 'F': (2, 1, 'L'), 'G': (2, 1, 'L'),
    'H': (2, 1, 'R'), 'J': (2, 1, 'R'), 'K': (2, 2, 'R'), 'L': (2, 3, 'R'), ';': (2, 4, 'R'),
    
    # Bottom row (row 3)
    'Z': (3, 4, 'L'), 'X': (3, 3, 'L'), 'C': (3, 2, 'L'), 'V': (3, 1, 'L'), 'B': (3, 1, 'L'),
    'N': (3, 1, 'R'), 'M': (3, 1, 'R'), ',': (3, 2, 'R'), '.': (3, 3, 'R'), '/': (3, 4, 'R'),
    
    # Additional common keys
    "'": (2, 4, 'R'), '[': (1, 4, 'R'), ']': (1, 4, 'R'), '\\': (1, 4, 'R'),
    '-': (0, 4, 'R'), '=': (0, 4, 'R'),
}

# Define finger strength (1=index, 2=middle are strong; 3=ring, 4=pinky are weak)
STRONG_FINGERS = {1, 2}
WEAK_FINGERS = {3, 4}

# Define home row
HOME_ROW = 2

# Define finger column assignments for detecting lateral movement
# Each finger has a "home column" - middle columns T,G,B,Y,H,N require lateral movement
FINGER_COLUMNS = {
    # Left hand columns (4=leftmost to 1=rightmost)
    'L': {
        4: ['Q', 'A', 'Z', '1'],           # Pinky column
        3: ['W', 'S', 'X', '2'],           # Ring column  
        2: ['E', 'D', 'C', '3'],           # Middle column
        1: ['R', 'F', 'V', '4', '5']       # Index column (excludes T, G, B - lateral movement)
    },
    # Right hand columns (1=leftmost to 4=rightmost)  
    'R': {
        1: ['U', 'J', 'M', '6', '7'],      # Index column (excludes Y, H, N - lateral movement)
        2: ['I', 'K', ',', '8'],           # Middle column
        3: ['O', 'L', '.', '9'],           # Ring column
        4: ['P', ';', '/', '0', "'", '[', ']', '\\', '-', '=']  # Pinky column
    }
}

def get_key_info(key: str) -> Tuple[int, int, str]:
    """Get (row, finger, hand) for a key."""
    key = key.upper()
    if key in QWERTY_LAYOUT:
        return QWERTY_LAYOUT[key]
    else:
        # Default for unknown keys
        return (2, 1, 'R')

def is_finger_in_column(key: str, finger: int, hand: str) -> bool:
    """Check if a key is in the designated column for a finger."""
    key = key.upper()
    if hand in FINGER_COLUMNS and finger in FINGER_COLUMNS[hand]:
        return key in FINGER_COLUMNS[hand][finger]
    return False

class Dvorak9Scorer:
    """Implements the Dvorak-9 scoring model for keyboard layout evaluation."""
    
    def __init__(self, layout_mapping: Dict[str, str], text: str):
        """
        Initialize scorer with layout mapping and text.
        
        Args:
            layout_mapping: Dict mapping characters to QWERTY positions (e.g., {'a': 'F', 'b': 'D'})
            text: Text to analyze for bigram scoring
        """
        self.layout_mapping = layout_mapping
        self.text = text.lower()
        self.bigrams = self._extract_bigrams()

    def _extract_bigrams(self) -> List[Tuple[str, str]]:
        """Extract all consecutive character pairs from text that exist in layout mapping."""
        bigrams = []
        
        # Filter text to only characters in our layout
        filtered_chars = [char for char in self.text if char in self.layout_mapping]
        
        # Create bigrams from consecutive characters
        for i in range(len(filtered_chars) - 1):
            char1, char2 = filtered_chars[i], filtered_chars[i + 1]
            bigrams.append((char1, char2))
        
        return bigrams

    def score_bigram(self, char1: str, char2: str) -> Dict[str, float]:
        """Score a single bigram according to the 9 Dvorak criteria."""
        pos1 = self.layout_mapping[char1]
        pos2 = self.layout_mapping[char2]
        
        row1, finger1, hand1 = get_key_info(pos1)
        row2, finger2, hand2 = get_key_info(pos2)
        
        scores = {}
        
        # 1. Hands - favor alternating hands
        scores['hands'] = 1.0 if hand1 != hand2 else 0.0
        
        # 2. Fingers - avoid same finger repetition
        if hand1 != hand2:
            scores['fingers'] = 1.0  # Different hands = different fingers
        else:
            scores['fingers'] = 0.0 if finger1 == finger2 else 1.0
        
        # 3. Skip fingers - favor skipping more fingers (same hand only)
        if hand1 != hand2:
            scores['skip_fingers'] = 1.0  # Different hands
        elif finger1 == finger2:
            scores['skip_fingers'] = 0.0  # Same finger
        else:
            finger_gap = abs(finger1 - finger2)
            if finger_gap == 1:
                scores['skip_fingers'] = 0.0  # Adjacent fingers (skip 0)
            elif finger_gap == 2:
                scores['skip_fingers'] = 0.5  # Skip 1 finger
            else:  # finger_gap == 3
                scores['skip_fingers'] = 1.0  # Skip 2 fingers (index to pinky)
        
        # 4. Don't cross home - avoid hurdling over home row
        if hand1 != hand2:
            scores['dont_cross_home'] = 1.0  # Different hands always score well
        else:
            # Check for hurdling (top to bottom or bottom to top, skipping home)
            if (row1 == 1 and row2 == 3) or (row1 == 3 and row2 == 1):
                scores['dont_cross_home'] = 0.0  # Hurdling over home row
            else:
                scores['dont_cross_home'] = 1.0  # No hurdling
        
        # 5. Same row - favor staying in same row
        scores['same_row'] = 1.0 if row1 == row2 else 0.0
        
        # 6. Home row - favor using home row
        home_count = sum(1 for row in [row1, row2] if row == HOME_ROW)
        if home_count == 2:
            scores['home_row'] = 1.0      # Both in home row
        elif home_count == 1:
            scores['home_row'] = 0.5      # One in home row
        else:
            scores['home_row'] = 0.0      # Neither in home row
        
        # 7. Columns - favor fingers staying in their designated columns
        in_column1 = is_finger_in_column(pos1, finger1, hand1)
        in_column2 = is_finger_in_column(pos2, finger2, hand2)
        
        if in_column1 and in_column2:
            scores['columns'] = 1.0       # Both in correct columns
        elif in_column1 or in_column2:
            scores['columns'] = 0.5       # One in correct column
        else:
            scores['columns'] = 0.0       # Neither in correct column
        
        # 8. Strum - favor inward rolls (outer to inner fingers)
        if hand1 != hand2:
            scores['strum'] = 1.0         # Different hands
        elif finger1 == finger2:
            scores['strum'] = 0.0         # Same finger
        else:
            # Inward roll: from higher finger number to lower (4→3→2→1)
            if finger1 > finger2:
                scores['strum'] = 1.0     # Inward roll
            else:
                scores['strum'] = 0.0     # Outward roll
        
        # 9. Strong fingers - favor index and middle fingers
        strong_count = sum(1 for finger in [finger1, finger2] if finger in STRONG_FINGERS)
        if strong_count == 2:
            scores['strong_fingers'] = 1.0    # Both strong fingers
        elif strong_count == 1:
            scores['strong_fingers'] = 0.5    # One strong finger
        else:
            scores['strong_fingers'] = 0.0    # Both weak fingers
        
        return scores

    def calculate_all_scores(self) -> Dict:
        """Calculate mean scores across all bigrams for each criterion."""
        if not self.bigrams:
            return self._empty_scores()
        
        # Initialize accumulators
        criterion_sums = defaultdict(float)
        criterion_counts = defaultdict(int)
        bigram_details = []
        
        # Score each bigram
        for char1, char2 in self.bigrams:
            bigram_scores = self.score_bigram(char1, char2)
            bigram_details.append((char1, char2, bigram_scores))
            
            for criterion, score in bigram_scores.items():
                criterion_sums[criterion] += score
                criterion_counts[criterion] += 1
        
        # Calculate mean scores
        mean_scores = {}
        for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                         'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
            if criterion_counts[criterion] > 0:
                mean_scores[criterion] = criterion_sums[criterion] / criterion_counts[criterion]
            else:
                mean_scores[criterion] = 0.0
        
        # Calculate total score (sum of all 9 criteria)
        total_score = sum(mean_scores.values())
        
        return {
            'scores': mean_scores,
            'total': total_score,
            'bigram_count': len(self.bigrams),
            'bigram_details': bigram_details
        }

    def _empty_scores(self) -> Dict:
        """Return empty score structure when no bigrams available."""
        criteria = ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                   'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']
        return {
            'scores': {criterion: 0.0 for criterion in criteria},
            'total': 0.0,
            'bigram_count': 0,
            'bigram_details': []
        }

    def get_detailed_breakdown(self) -> Dict:
        """Get detailed breakdown by criterion showing good/bad examples."""
        if not self.bigrams:
            return {}
        
        breakdown = defaultdict(lambda: {'good': [], 'bad': [], 'neutral': []})
        
        for char1, char2 in self.bigrams:
            bigram_scores = self.score_bigram(char1, char2)
            pos1, pos2 = self.layout_mapping[char1], self.layout_mapping[char2]
            
            for criterion, score in bigram_scores.items():
                example = f"{char1}{char2} ({pos1}{pos2})"
                
                if score >= 0.8:
                    breakdown[criterion]['good'].append(example)
                elif score <= 0.2:
                    breakdown[criterion]['bad'].append(example)
                else:
                    breakdown[criterion]['neutral'].append(example)
        
        return dict(breakdown)

def print_results(results: Dict, show_details: bool = False):
    """Print formatted results."""
    scores = results['scores']
    
    print("Dvorak-9 Scoring Results")
    print("=" * 60)
    
    criteria_info = [
        ('hands', '1. Hands (favor alternating hands)'),
        ('fingers', '2. Fingers (avoid same finger)'),
        ('skip_fingers', '3. Skip fingers (favor non-adjacent)'),
        ('dont_cross_home', '4. Don\'t cross home (avoid hurdling)'),
        ('same_row', '5. Same row (stay in same row)'),
        ('home_row', '6. Home row (favor home row use)'),
        ('columns', '7. Columns (stay in finger columns)'),
        ('strum', '8. Strum (favor inward rolls)'),
        ('strong_fingers', '9. Strong fingers (favor index/middle)')
    ]
    
    for key, name in criteria_info:
        score = scores[key]
        print(f"{name:<45} | {score:6.3f}")
    
    print("-" * 60)
    print(f"{'Total Score (sum of all 9)':<45} | {results['total']:6.3f}")
    print(f"{'Average Score':<45} | {results['total']/9:6.3f}")
    print(f"{'Bigrams analyzed':<45} | {results['bigram_count']:6d}")
    
    if show_details:
        print("\nDetailed Breakdown:")
        print("=" * 60)
        
        breakdown = {}
        if results['bigram_details']:
            # Rebuild breakdown from bigram details
            for char1, char2, bigram_scores in results['bigram_details']:
                for criterion, score in bigram_scores.items():
                    if criterion not in breakdown:
                        breakdown[criterion] = {'good': [], 'bad': [], 'neutral': []}
                    
                    example = f"{char1}{char2}"
                    if score >= 0.8:
                        breakdown[criterion]['good'].append(example)
                    elif score <= 0.2:
                        breakdown[criterion]['bad'].append(example)
                    else:
                        breakdown[criterion]['neutral'].append(example)
        
        for key, name in criteria_info:
            print(f"\n{name}:")
            if key in breakdown:
                b = breakdown[key]
                print(f"  Good examples (≥0.8): {', '.join(b['good'][:10])}")
                if len(b['good']) > 10:
                    print(f"    ... and {len(b['good'])-10} more")
                print(f"  Bad examples (≤0.2): {', '.join(b['bad'][:10])}")
                if len(b['bad']) > 10:
                    print(f"    ... and {len(b['bad'])-10} more")





# Enhanced Dvorak9 scorer with bigram-level combination weighting

def get_combination_weights():
    """Return weights for different feature combinations based on empirical analysis."""
    return {
        # 5-way combinations (strongest)
        ('fingers', 'same_row', 'home_row', 'strum', 'strong_fingers'): -0.148,
        
        # 4-way combinations
        ('fingers', 'same_row', 'home_row', 'strum'): -0.147,
        ('fingers', 'same_row', 'home_row', 'strong_fingers'): -0.141,
        ('fingers', 'same_row', 'strum', 'strong_fingers'): -0.138,
        
        # 3-way combinations (most robust)
        ('fingers', 'same_row', 'strum'): -0.124,
        ('fingers', 'same_row', 'home_row'): -0.119,
        ('fingers', 'strum', 'strong_fingers'): -0.115,
        
        # 2-way combinations
        ('fingers', 'same_row'): -0.106,
        ('fingers', 'strum'): -0.102,
        ('same_row', 'strum'): -0.098,
        
        # Individual features (base effects)
        ('fingers',): -0.088,
        ('strum',): -0.087,
        ('same_row',): -0.088,
        ('home_row',): -0.058,
        ('strong_fingers',): -0.045,
        ('skip_fingers',): -0.038,
        ('columns',): -0.025,
        ('dont_cross_home',): -0.018,
        ('hands',): +0.090,  # Positive = bad for typing speed
        
        # Default for no features
        (): 0.0
    }

def identify_bigram_combination(bigram_scores, threshold=0.8):
    """
    Identify which feature combination a bigram exhibits.
    
    Args:
        bigram_scores: Dict of feature scores for a single bigram
        threshold: Minimum score to consider a feature "active"
    
    Returns:
        Tuple of active feature names, sorted for consistent lookup
    """
    active_features = []
    
    for feature, score in bigram_scores.items():
        if score >= threshold:
            active_features.append(feature)
    
    return tuple(sorted(active_features))

def score_bigram_weighted(bigram_scores, combination_weights):
    """
    Score a single bigram using combination-specific weights.
    
    Args:
        bigram_scores: Dict of 9 feature scores for the bigram
        combination_weights: Dict mapping combinations to empirical weights
    
    Returns:
        Weighted score for this bigram
    """
    # Identify which combination this bigram exhibits
    combination = identify_bigram_combination(bigram_scores)
    
    # Try to find exact match first
    if combination in combination_weights:
        weight = combination_weights[combination]
        # Calculate combination strength (how well does bigram exhibit this combination)
        combination_strength = sum(bigram_scores[feature] for feature in combination) / len(combination) if combination else 0
        return weight * combination_strength
    
    # Fall back to best partial match
    best_weight = 0.0
    best_score = 0.0
    
    for combo, weight in combination_weights.items():
        if not combo:  # Skip empty combination
            continue
            
        # Check if this bigram exhibits this combination (partial match allowed)
        if all(feature in bigram_scores for feature in combo):
            combo_strength = sum(bigram_scores[feature] for feature in combo) / len(combo)
            
            # Penalize partial matches
            match_completeness = len(set(combo) & set(identify_bigram_combination(bigram_scores))) / len(combo)
            adjusted_score = weight * combo_strength * match_completeness
            
            if abs(adjusted_score) > abs(best_score):
                best_score = adjusted_score
    
    return best_score

class EnhancedDvorak9Scorer(Dvorak9Scorer):
    """Enhanced scorer using bigram-level combination weighting."""
    
    def __init__(self, layout_mapping, text):
        super().__init__(layout_mapping, text)
        self.combination_weights = get_combination_weights()
    
    def calculate_combination_scores(self):
        """Calculate layout score using bigram-level combination weighting."""
        if not self.bigrams:
            return {
                'total_weighted_score': 0.0,
                'bigram_count': 0,
                'combination_breakdown': {},
                'individual_scores': self._empty_scores()['scores']
            }
        
        total_weighted_score = 0.0
        combination_counts = defaultdict(int)
        combination_score_sums = defaultdict(float)
        bigram_details = []
        
        # Score each bigram with combination weighting
        for char1, char2 in self.bigrams:
            bigram_scores = self.score_bigram(char1, char2)
            weighted_score = score_bigram_weighted(bigram_scores, self.combination_weights)
            combination = identify_bigram_combination(bigram_scores)
            
            total_weighted_score += weighted_score
            combination_counts[combination] += 1
            combination_score_sums[combination] += weighted_score
            
            bigram_details.append({
                'bigram': f"{char1}{char2}",
                'scores': bigram_scores,
                'combination': combination,
                'weighted_score': weighted_score
            })
        
        # Calculate combination breakdown
        combination_breakdown = {}
        for combo, count in combination_counts.items():
            combination_breakdown[combo] = {
                'count': count,
                'total_contribution': combination_score_sums[combo],
                'average_score': combination_score_sums[combo] / count if count > 0 else 0,
                'percentage': count / len(self.bigrams) * 100
            }
        
        # Also calculate traditional individual scores for comparison
        individual_results = super().calculate_all_scores()
        
        return {
            'total_weighted_score': total_weighted_score,
            'average_weighted_score': total_weighted_score / len(self.bigrams),
            'bigram_count': len(self.bigrams),
            'combination_breakdown': combination_breakdown,
            'individual_scores': individual_results['scores'],
            'bigram_details': bigram_details
        }

def print_combination_results(results):
    """Print formatted results from combination scoring."""
    print("Enhanced Dvorak-9 Combination Scoring Results")
    print("=" * 70)
    
    print(f"Total Weighted Score: {results['total_weighted_score']:8.3f}")
    print(f"Average per Bigram:   {results['average_weighted_score']:8.3f}")
    print(f"Bigrams Analyzed:     {results['bigram_count']:8d}")
    
    print("\nCombination Breakdown:")
    print("-" * 70)
    print(f"{'Combination':<35} {'Count':<8} {'Contrib':<10} {'Avg':<8} {'%':<6}")
    print("-" * 70)
    
    # Sort by total contribution
    sorted_combos = sorted(results['combination_breakdown'].items(), 
                          key=lambda x: abs(x[1]['total_contribution']), 
                          reverse=True)
    
    for combo, stats in sorted_combos[:15]:  # Top 15 combinations
        combo_str = '+'.join(combo) if combo else 'none'
        if len(combo_str) > 34:
            combo_str = combo_str[:31] + '...'
        
        print(f"{combo_str:<35} {stats['count']:<8} {stats['total_contribution']:<10.3f} "
              f"{stats['average_score']:<8.3f} {stats['percentage']:<6.1f}")

# Example usage
if __name__ == "__main__":
    # Test with a simple layout
    layout_mapping = {'a': 'F', 'b': 'D', 'c': 'J'}
    text = "abacaba"
    
    # Traditional scoring
    traditional_scorer = Dvorak9Scorer(layout_mapping, text)
    traditional_results = traditional_scorer.calculate_all_scores()
    
    print("Traditional Scoring:")
    print(f"Total: {traditional_results['total']:.3f}")
    print()
    
    # Enhanced combination scoring
    enhanced_scorer = EnhancedDvorak9Scorer(layout_mapping, text)
    combination_results = enhanced_scorer.calculate_combination_scores()
    
    print_combination_results(combination_results)




def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate Dvorak-9 layout scores for keyboard evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic scoring
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --text "abacaba"
  
  # Score layout with details  
  python dvorak9_scorer.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details
  
  # Use items as text if no text provided
  python dvorak9_scorer.py --items "abc" --positions "FDJ"
        """
    )
    
    parser.add_argument("--items", required=True,
                       help="String of characters (e.g., 'etaoinsrhldcumfp')")
    parser.add_argument("--positions", required=True,
                       help="String of QWERTY positions (e.g., 'FDESRJKUMIVLA;OW')")
    parser.add_argument("--text",
                       help="Text to analyze (default: uses items string)")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed breakdown with examples")
    parser.add_argument("--csv", action="store_true",
                       help="Output in CSV format")
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        if len(args.items) != len(args.positions):
            print(f"Error: Character count ({len(args.items)}) != Position count ({len(args.positions)})")
            return
        
        # Create layout mapping
        layout_mapping = dict(zip(args.items.lower(), args.positions.upper()))
        
        # Use provided text or items string
        text = args.text if args.text else args.items
        
        # Initialize scorer and calculate
        scorer = Dvorak9Scorer(layout_mapping, text)
        results = scorer.calculate_all_scores()
        
        if args.csv:
            # CSV output
            print("criterion,score")
            for criterion, score in results['scores'].items():
                print(f"{criterion},{score:.6f}")
            print(f"total,{results['total']:.6f}")
        else:
            # Human-readable output
            print(f"Layout: {args.items} → {args.positions}")
            print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print()
            
            print_results(results, args.details)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()