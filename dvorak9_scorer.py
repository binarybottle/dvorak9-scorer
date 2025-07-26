# dvorak9_scorer.py
"""
Dvorak-9 empirical scoring model for keyboard layout evaluation.

This script implements the 9 evaluation criteria derived from Dvorak's "Typing Behavior" 
book and patent (1936) with empirical weights based on analysis of real typing performance data.

The 9 scoring criteria for typing bigrams are (0-1, higher = better performance):
1. Hands - favor alternating hands over same hand
2. Fingers - avoid same finger repetition  
3. Skip fingers - favor non-adjacent fingers over adjacent (same hand)
4. Don't cross home - avoid crossing over the home row (hurdling)
5. Same row - favor typing within the same row
6. Home row - favor using the home row
7. Columns - favor fingers staying in their designated columns
8. Strum - favor inward rolls over outward rolls (same hand)
9. Strong fingers - favor stronger fingers over weaker ones

The combined layout score is calculated as the negative of the empirical weighted combination score,
so that HIGHER scores indicate BETTER layouts (consistent with individual criteria).

Requires a weights CSV file.
  - NOTE: The empirical weights derived from 136M+ keystroke dataset analysis with FDR correction
          tested 511 combinations, which should cover most real bigrams reasonably well.
  - NOTE: The empirical weights derived from the comfort scores keystroke dataset analysis with FDR correction
          tested 511 combinations, which should cover most real bigrams reasonably well.

# Basic scoring
poetry run python3 dvorak9_scorer.py --items "qwertyuiopasdfghjkl;zxcvbnm,./" --positions "QWERTYUIOPASDFGHJKL;ZXCVBNM,./"  --weights-csv "weights/combinations_weights_from_speed_significant.csv"
poetry run python3 dvorak9_scorer.py --items "qwertyuiopasdfghjkl;zxcvbnm,./" --positions "QWERTYUIOPASDFGHJKL;ZXCVBNM,./"  --weights-csv "weights/combinations_weights_from_comfort_significant.csv"

python dvorak9_scorer.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW"

# With detailed breakdown
python dvorak9_scorer.py --items "abc" --positions "FDJ" --details

# CSV export for analysis
python dvorak9_scorer.py --items "abc" --positions "FDJ" --csv > results.csv

# Use items as text if no text provided
python dvorak9_scorer.py --items "abc" --positions "FDJ" --text "abacaba"
python dvorak9_scorer.py --items "abc" --positions "FDJ" --text-file "sample_text.txt"

# Return just the 10 scores (average weighted + 9 individual scores)
python dvorak9_scorer.py --items "abc" --positions "FDJ" --ten-scores

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
        4: ['Q', 'A', 'Z', '1'],                                 # Pinky column
        3: ['W', 'S', 'X', '2'],                                 # Ring column  
        2: ['E', 'D', 'C', '3'],                                 # Middle column
        1: ['R', 'F', 'V', '4'] #'T', 'G', 'B', '5']             # Index column
    },
    # Right hand columns (1=leftmost to 4=rightmost)  
    'R': {
        1: ['U', 'J', 'M', '7'], #'Y', 'H', 'N', '6'],           # Index column
        2: ['I', 'K', ',', '8'],                                 # Middle column
        3: ['O', 'L', '.', '9'],                                 # Ring column
        4: ['P', ';', '/', '0']  #"'", '[', ']', '\\', '-', '='] # Pinky column
    }
}

def get_key_info(key: str) -> Tuple[int, int, str]:
    """Get (row, finger, hand) for a key."""
    key = key.upper()
    if key in QWERTY_LAYOUT:
        return QWERTY_LAYOUT[key]
    else:
        return None

def is_finger_in_column(key: str, finger: int, hand: str) -> bool:
    """Check if a key is in the designated column for a finger."""
    key = key.upper()
    if hand in FINGER_COLUMNS and finger in FINGER_COLUMNS[hand]:
        return key in FINGER_COLUMNS[hand][finger]
    return False

def score_bigram_dvorak9(bigram: str) -> Dict[str, float]:
    """
    Calculate all 9 Dvorak criteria scores for a bigram.
        
    Args:
        bigram: Two-character string (e.g., "th", "er")
        
    Returns:
        Dict with keys: hands, fingers, skip_fingers, dont_cross_home, same_row, 
                       home_row, columns, strum, strong_fingers
        Values are 0-1 where higher = better for typing speed according to Dvorak principles
    """
    if len(bigram) != 2:
        # Return neutral scores for invalid input
        return {criterion: 0.5 for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                                                'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']}
    
    char1, char2 = bigram[0].upper(), bigram[1].upper()
    
    # Get key information
    row1, finger1, hand1 = get_key_info(char1)
    row2, finger2, hand2 = get_key_info(char2)
    
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
        scores['skip_fingers'] = 1.0      # Different hands is good
    elif finger1 == finger2:
        scores['skip_fingers'] = 0.0      # Same finger is bad
    else:
        finger_gap = abs(finger1 - finger2)
        if finger_gap == 1:
            scores['skip_fingers'] = 0    # Adjacent fingers is bad
        elif finger_gap == 2:
            scores['skip_fingers'] = 0.5  # Skipping 1 finger is good
        elif finger_gap == 3:
            scores['skip_fingers'] = 1.0  # Skipping 2 fingers is better
    
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
    in_column1 = is_finger_in_column(char1, finger1, hand1)
    in_column2 = is_finger_in_column(char2, finger2, hand2)
    
    if in_column1 and in_column2:
        scores['columns'] = 1.0       # Both in correct columns
    elif in_column1 or in_column2:
        scores['columns'] = 0.5       # One in correct column
    else:
        scores['columns'] = 0.0       # Neither in correct column
    
    # 8. Strum - favor inward rolls (outer to inner fingers)
    if hand1 != hand2:
        scores['strum'] = 1.0         # Different hands get full score
    elif finger1 == finger2:
        scores['strum'] = 0.0         # Same finger gets zero
    else:
        # Inward roll: from higher finger number to lower (4→3→2→1)
        # This represents rolling from pinky toward index finger
        if finger1 > finger2:
            scores['strum'] = 1.0     # Inward roll (e.g., pinky to ring, ring to middle)
        else:
            scores['strum'] = 0.0     # Outward roll (e.g., index to middle, middle to ring)
    
    # 9. Strong fingers - favor index and middle fingers
    strong_count = sum(1 for finger in [finger1, finger2] if finger in STRONG_FINGERS)
    if strong_count == 2:
        scores['strong_fingers'] = 1.0    # Both strong fingers
    elif strong_count == 1:
        scores['strong_fingers'] = 0.5    # One strong finger
    else:
        scores['strong_fingers'] = 0.0    # Both weak fingers
    
    return scores

def load_combination_weights(csv_path: str = "weights/combinations_weights_from_speed_significant.csv"):
    """
    Load empirical correlation weights for different feature combinations from CSV file.
    
    Args:
        csv_path: Path to CSV file containing combination correlations
        
    Returns:
        Dict mapping combination tuples to correlation values
        
    The CSV should have 'combination' and 'correlation' columns.
    These correlations are derived from analysis of 136M+ keystroke dataset
    with FDR correction applied to 529 statistical tests.
    """
    import csv
    import os
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Combination weights file not found: {csv_path}")
    
    combination_weights = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse combination name
                combination_str = row.get('combination', '').strip()
                
                # Get correlation value
                correlation_str = row.get('correlation', '').strip()
                if not correlation_str:
                    continue  # Skip rows without correlation values
                
                try:
                    correlation = float(correlation_str)
                except ValueError:
                    continue  # Skip rows with invalid correlation values
                
                # Convert combination string to tuple
                if combination_str.lower() in ['none', 'empty', '']:
                    combination = ()
                else:
                    # For individual features, just use the feature name
                    if '+' in combination_str:
                        # Multi-feature combination like "fingers+same_row+strum"
                        features = [f.strip() for f in combination_str.split('+')]
                        combination = tuple(sorted(features))
                    else:
                        # Single feature like "fingers"
                        combination = (combination_str,)
                
                combination_weights[combination] = correlation
                
    except Exception as e:
        raise ValueError(f"Error parsing combination weights CSV: {e}")
    
    # Ensure we have at least an empty combination
    if () not in combination_weights:
        combination_weights[()] = 0.0
    
    print(f"Loaded {len(combination_weights)} combination weights from {csv_path}")
    
    return combination_weights

def identify_bigram_combination(bigram_scores, threshold=0):
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
    Score a single bigram using empirical combination-specific weights.
    
    Args:
        bigram_scores: Dict of 9 feature scores for the bigram
        combination_weights: Dict mapping combinations to empirical weights
    
    Returns:
        Weighted score for this bigram (negative = good for typing speed)
    """
    # Identify which combination this bigram exhibits
    combination = identify_bigram_combination(bigram_scores)
    
    # Try to find exact match first
    if combination in combination_weights:
        weight = combination_weights[combination]
        # Calculate combination strength (how well does bigram exhibit this combination)
        combination_strength = sum(bigram_scores[feature] for feature in combination) / len(combination) if combination else 0
        return weight * combination_strength
    
    print(f"Warning: No exact combination match for {combination}. Using best partial match.")

    # Fall back to best partial match
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

class Dvorak9Scorer:
    """
    Dvorak-9 scorer using empirical combination weighting.
    
    Implements the 9 evaluation criteria derived from Dvorak's work with
    empirical weights derived from analysis of real typing performance data.
    """
    
    def __init__(self, layout_mapping: Dict[str, str], text: str, weights_csv: str = "weights/combinations_weights_from_speed_significant.csv"):
        """
        Initialize scorer with layout mapping and text.
        
        Args:
            layout_mapping: Dict mapping characters to QWERTY positions (e.g., {'a': 'F', 'b': 'D'})
            text: Text to analyze for bigram scoring
            weights_csv: Path to CSV file containing empirical combination weights
        """
        self.layout_mapping = layout_mapping
        self.text = text.lower()
        self.bigrams = self._extract_bigrams()
        self.combination_weights = load_combination_weights(weights_csv)

    def _extract_bigrams(self) -> List[Tuple[str, str]]:
        """Extract all consecutive character pairs from text that exist in layout mapping.
        
        Spaces and other non-layout characters act as word boundaries that break bigram chains.
        """
        bigrams = []
        
        # Split text into words first (respecting boundaries)
        import re
        words = re.findall(r'\S+', self.text)  # Split on whitespace
        
        for word in words:
            # Filter each word to only characters in our layout
            filtered_chars = [char for char in word.lower() if char in self.layout_mapping]
            
            # Create bigrams from consecutive characters within each word
            for i in range(len(filtered_chars) - 1):
                char1, char2 = filtered_chars[i], filtered_chars[i + 1]
                bigrams.append((char1, char2))
        
        return bigrams

    def score_bigram(self, char1: str, char2: str) -> Dict[str, float]:
        """Score a single bigram according to the 9 Dvorak criteria."""
        # Map characters through layout to QWERTY positions
        pos1 = self.layout_mapping.get(char1, char1.upper())
        pos2 = self.layout_mapping.get(char2, char2.upper())
        
        # Use the canonical scoring function
        return score_bigram_dvorak9(pos1 + pos2)

    def calculate_scores(self):
        """Calculate layout score using empirical combination weighting."""
        if not self.bigrams:
            return {
                'layout_score': 0.0,  # 0 is neutral for higher = better
                'average_weighted_score': 0.0,
                'total_weighted_score': 0.0,
                'bigram_count': 0,
                'individual_scores': {},
                'combination_breakdown': {},
                'bigram_details': []
            }
        
        total_weighted_score = 0.0
        combination_counts = defaultdict(int)
        combination_score_sums = defaultdict(float)
        bigram_details = []
        
        # Initialize accumulators for individual unweighted scores
        criterion_sums = defaultdict(float)
        criterion_counts = defaultdict(int)
        
        # Score each bigram with combination weighting
        for char1, char2 in self.bigrams:
            bigram_scores = self.score_bigram(char1, char2)
            weighted_score = score_bigram_weighted(bigram_scores, self.combination_weights)
            combination = identify_bigram_combination(bigram_scores)
            
            total_weighted_score += -weighted_score  # FLIP SIGN
            combination_counts[combination] += 1
            combination_score_sums[combination] += -weighted_score  # FLIP SIGN
            
            # Accumulate individual criterion scores (unweighted)
            for criterion, score in bigram_scores.items():
                criterion_sums[criterion] += score
                criterion_counts[criterion] += 1
            
            bigram_details.append({
                'bigram': f"{char1}{char2}",
                'scores': bigram_scores,
                'combination': combination,
                'weighted_score': -weighted_score  # FLIP SIGN
            })
        
        # Calculate mean individual scores (unweighted, 0-1 scale)
        individual_scores = {}
        for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                         'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
            if criterion_counts[criterion] > 0:
                individual_scores[criterion] = criterion_sums[criterion] / criterion_counts[criterion]
            else:
                individual_scores[criterion] = 0.0
        
        # Calculate combination breakdown
        combination_breakdown = {}
        for combo, count in combination_counts.items():
            combination_breakdown[combo] = {
                'count': count,
                'total_contribution': combination_score_sums[combo],  # Already flipped above
                'average_score': combination_score_sums[combo] / count if count > 0 else 0,  # Already flipped
                'percentage': count / len(self.bigrams) * 100
            }

        # Calculate average weighted score (the main metric)
        average_weighted_score = total_weighted_score / len(self.bigrams)

        return {
            'layout_score': average_weighted_score,  # Primary metric
            'average_weighted_score': average_weighted_score,
            'total_weighted_score': total_weighted_score,
            'bigram_count': len(self.bigrams),
            'individual_scores': individual_scores,
            'combination_breakdown': combination_breakdown,
            'bigram_details': bigram_details
        }

    def get_detailed_breakdown(self) -> Dict:
        """Get detailed breakdown by criterion showing good/bad examples."""
        if not self.bigrams:
            return {}
        
        breakdown = defaultdict(lambda: {'good': [], 'bad': [], 'neutral': []})
        
        for char1, char2 in self.bigrams:
            bigram_scores = self.score_bigram(char1, char2)
            pos1 = self.layout_mapping.get(char1, char1.upper())
            pos2 = self.layout_mapping.get(char2, char2.upper())
            
            for criterion, score in bigram_scores.items():
                example = f"{char1}{char2} ({pos1}{pos2})"
                
                if score >= 0.8:
                    breakdown[criterion]['good'].append(example)
                elif score <= 0.2:
                    breakdown[criterion]['bad'].append(example)
                else:
                    breakdown[criterion]['neutral'].append(example)
        
        return dict(breakdown)

def print_combination_results(results):
    """Print formatted results from empirical combination scoring."""
    print("Dvorak-9 Empirical Combination Scoring Results")
    print("=" * 70)
    
    # Primary metric - empirically validated
    print(f"Empirical Score: {results['average_weighted_score']:8.3f}")
    print(f"Bigrams Analyzed: {results['bigram_count']:8d}")
    
    # Secondary metrics - individual criteria for interpretability 
    print(f"\nIndividual Criterion Breakdown (0-1 scale, higher = better):")
    print("-" * 60)
    
    criteria_info = [
        ('hands', '1. Hands (alternating)'),
        ('fingers', '2. Fingers (avoid same)'),
        ('skip_fingers', '3. Skip fingers'),
        ('dont_cross_home', '4. Don\'t cross home'),
        ('same_row', '5. Same row'),
        ('home_row', '6. Home row'),
        ('columns', '7. Columns'),
        ('strum', '8. Strum (inward rolls)'),
        ('strong_fingers', '9. Strong fingers')
    ]
    
    individual_scores = results.get('individual_scores', {})
    for key, name in criteria_info:
        score = individual_scores.get(key, 0.0)
        print(f"  {name:<25}: {score:6.3f}")
    
    # Detailed breakdown - combination contributions
    print(f"\nCombination Breakdown:")
    print("-" * 70)
    print(f"{'Combination':<35} {'Count':<8} {'Contrib':<10} {'Avg':<8} {'%':<6}")
    print("-" * 70)
    
    # Sort by total contribution magnitude
    sorted_combos = sorted(results['combination_breakdown'].items(), 
                          key=lambda x: abs(x[1]['total_contribution']), 
                          reverse=True)
    
    for combo, stats in sorted_combos[:15]:  # Top 15 combinations
        combo_str = '+'.join(combo) if combo else 'none'
        if len(combo_str) > 34:
            combo_str = combo_str[:31] + '...'
        
        print(f"{combo_str:<35} {stats['count']:<8} {stats['total_contribution']:<10.3f} "
              f"{stats['average_score']:<8.3f} {stats['percentage']:<6.1f}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate Dvorak-9 layout scores using empirical combination weighting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic scoring
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --text "abacaba"
  
  # Score layout with details  
  python dvorak9_scorer.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details
  
  # Use items as text if no text provided
  python dvorak9_scorer.py --items "abc" --positions "FDJ"
  
  # Use custom weights file
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --weights-csv "weights/combinations_weights_from_speed_significant.csv"

  # QWERTY to QWERTY
  poetry run python3 dvorak9_scorer.py --items "qwertyuiopasdfghjkl;zxcvbnm,./" --positions "QWERTYUIOPASDFGHJKL;ZXCVBNM,./"

  # Use items as text if no text provided
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --text "abacaba"
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --text-file "sample_text.txt"

  # Return just the 10 scores (total weighted + 9 individual scores)
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --ten-scores
  
"""
    )
    
    parser.add_argument("--items", required=True,
                       help="String of characters (e.g., 'etaoinsrhldcumfp')")
    parser.add_argument("--positions", required=True,
                       help="String of QWERTY positions (e.g., 'FDESRJKUMIVLA;OW')")
    parser.add_argument("--text",
                       help="Text to analyze (default: uses items string)")
    parser.add_argument("--text-file",
                       help="Path to text file to analyze (alternative to --text)")
    parser.add_argument("--weights-csv", default="weights/combinations_weights_from_speed_significant.csv",
                       help="Path to CSV file containing empirical combination weights")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed breakdown with examples")
    parser.add_argument("--csv", action="store_true",
                       help="Output in CSV format")
    parser.add_argument("--ten-scores", action="store_true",
                       help="Output only 10 scores: total weighted score followed by 9 individual scores")

    args = parser.parse_args()
    
    try:
        # Validate inputs
        if len(args.items) != len(args.positions):
            print(f"Error: Character count ({len(args.items)}) != Position count ({len(args.positions)})")
            return
        
        # Filter to only letters, keeping corresponding positions
        letter_pairs = [(char, pos) for char, pos in zip(args.items, args.positions) if char.isalpha()]
        
        if not letter_pairs:
            print("Error: No letters found in --items")
            return
        
        # Reconstruct filtered strings
        filtered_items = ''.join(pair[0] for pair in letter_pairs)
        filtered_positions = ''.join(pair[1] for pair in letter_pairs)
        
        print(f"Filtered to letters only: '{filtered_items}' → '{filtered_positions}'")
        
        # Create layout mapping
        layout_mapping = dict(zip(filtered_items.lower(), filtered_positions.upper()))

        # Determine text source (priority: text-file > text > items)
        if args.text_file:
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading text file: {e}")
                return
        elif args.text:
            text = args.text
        else:
            text = args.items

        # Calculate scores
        scorer = Dvorak9Scorer(layout_mapping, text, args.weights_csv)
        results = scorer.calculate_scores()

        if args.ten_scores:
            # Output 10 scores: average weighted score + 9 individual scores
            individual_scores = results.get('individual_scores', {})
            
            # Use average_weighted_score
            scores = [results['average_weighted_score']]
            
            # Add individual scores in consistent order
            for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                            'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
                scores.append(individual_scores.get(criterion, 0.0))
            
            print(' '.join(f"{score:.6f}" for score in scores))

        elif args.csv:
            # CSV output
            print("metric,value")
            
            # Use average as the main empirical score
            print(f"empirical_score,{results['average_weighted_score']:.6f}")            
            print(f"average_empirical_score,{results['average_weighted_score']:.6f}")
            print(f"total_empirical_score,{results['total_weighted_score']:.6f}")
            print(f"bigram_count,{results['bigram_count']}")
            
            # Individual unweighted scores
            individual_scores = results.get('individual_scores', {})
            for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                             'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
                score = individual_scores.get(criterion, 0.0)
                print(f"individual_{criterion},{score:.6f}")
            
            # Top combinations
            sorted_combos = sorted(results['combination_breakdown'].items(), 
                                  key=lambda x: abs(x[1]['total_contribution']), 
                                  reverse=True)
            for i, (combo, stats) in enumerate(sorted_combos[:5]):
                combo_str = '+'.join(combo) if combo else 'none'
                print(f"top_combo_{i+1},{combo_str}")
                print(f"top_combo_{i+1}_contribution,{stats['total_contribution']:.6f}")
                
        else:
            # Human-readable output
            print(f"Layout: {args.items} → {args.positions}")
            print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print()
            
            print_combination_results(results)
            
            if args.details:
                breakdown = scorer.get_detailed_breakdown()
                print("\nDetailed Breakdown by Criterion:")
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
                    if key in breakdown:
                        b = breakdown[key]
                        print(f"\n{name}:")
                        print(f"  Good examples (≥0.8): {', '.join(b['good'][:10])}")
                        if len(b['good']) > 10:
                            print(f"    ... and {len(b['good'])-10} more")
                        print(f"  Bad examples (≤0.2): {', '.join(b['bad'][:10])}")
                        if len(b['bad']) > 10:
                            print(f"    ... and {len(b['bad'])-10} more")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if being run directly vs imported
    import sys
    if len(sys.argv) > 1:
        # Command line usage
        main()
    else:
        # Example usage when imported/run without args
        print("Dvorak-9 Empirical Combination Scoring Example")
        print("="*50)
        
        layout_mapping = {'a': 'F', 'b': 'D', 'c': 'J'}
        text = "abacaba"
        
        try:
            scorer = Dvorak9Scorer(layout_mapping, text)
            results = scorer.calculate_scores()
            print_combination_results(results)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure weights file is in the weights directory.")
            print("This file should contain the empirical combination weights.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()