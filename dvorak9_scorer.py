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

The combined layout score is calculated using empirical combination weights with proper
sign handling: speed weights (negative correlation = faster) are flipped, comfort weights 
(positive correlation = more comfortable) are kept as-is. Higher scores = better layouts.

Requires a weights CSV file for weighted scoring, or use --no-weights for unweighted 0-1 scoring.

# Weighted scoring with speed-based weights
python dvorak9_scorer.py --items "qwertyuiopasdfghjkl;zxcvbnm,./" --positions "QWERTYUIOPASDFGHJKL;ZXCVBNM,./" --weights-csv "weights/combinations_weights_from_speed_significant.csv"

# Weighted scoring with comfort-based weights
python dvorak9_scorer.py --items "qwertyuiopasdfghjkl;zxcvbnm,./" --positions "QWERTYUIOPASDFGHJKL;ZXCVBNM,./" --weights-csv "weights/combinations_weights_from_comfort_significant.csv"

# Unweighted scoring (0-1 individual criteria only) - FIXED CHARACTER COUNT
python dvorak9_scorer.py --items "etaoinshr" --positions "FDEGJWXRT" --no-weights

# With detailed breakdown - FIXED CHARACTER COUNT
python dvorak9_scorer.py --items "etaoinshr" --positions "FDEGJWXRT" --details --no-weights

# CSV export for analysis - FIXED CHARACTER COUNT
python dvorak9_scorer.py --items "etaoinshr" --positions "FDEGJWXRT" --csv --no-weights

# Return just the 10 scores (average + 9 individual scores) - FIXED CHARACTER COUNT
python dvorak9_scorer.py --items "etaoinshr" --positions "FDEGJWXRT" --ten-scores --no-weights

"""

import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

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

def get_key_info(key: str) -> Optional[Tuple[int, int, str]]:
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
    key_info1 = get_key_info(char1)
    key_info2 = get_key_info(char2)
    
    if key_info1 is None or key_info2 is None:
        return {criterion: 0.5 for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                                                'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']}
    
    row1, finger1, hand1 = key_info1
    row2, finger2, hand2 = key_info2
    
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

def load_combination_weights(csv_path: Optional[str] = None) -> Optional[Dict[Tuple[str, ...], float]]:
    """
    Load empirical correlation weights for different feature combinations from CSV file.
    
    Args:
        csv_path: Path to CSV file containing combination correlations (optional)
        
    Returns:
        Dict mapping combination tuples to correlation values, or None if no file provided
        
    The CSV should have 'combination' and 'correlation' columns.
    These correlations are derived from analysis of 136M+ keystroke dataset
    with FDR correction applied to 529 statistical tests.
    """
    if csv_path is None:
        print("No weights file provided - using unweighted scoring")
        return None
        
    import csv
    import os
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Combination weights file not found: {csv_path}")
    
    combination_weights: Dict[Tuple[str, ...], float] = {}
    
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

def identify_bigram_combination(bigram_scores: Dict[str, float], threshold: float = 0) -> Tuple[str, ...]:
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

def score_bigram_weighted(bigram_scores: Dict[str, float], 
                         combination_weights: Dict[Tuple[str, ...], float]) -> float:
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
    Dvorak-9 scorer using empirical combination weighting or unweighted scoring.
    
    Implements the 9 evaluation criteria derived from Dvorak's work with
    optional empirical weights derived from analysis of real typing performance data.
    """
    
    def __init__(self, layout_mapping: Dict[str, str], text: str, weights_csv: Optional[str] = None):
        """
        Initialize scorer with layout mapping and text.
        
        Args:
            layout_mapping: Dict mapping characters to QWERTY positions (e.g., {'a': 'F', 'b': 'D'})
            text: Text to analyze for bigram scoring
            weights_csv: Path to CSV file containing empirical combination weights (optional)
        """
        self.layout_mapping = layout_mapping
        self.text = text.lower()
        self.bigrams = self._extract_bigrams()
        
        # Load weights if provided, otherwise use None for unweighted scoring
        if weights_csv:
            self.combination_weights = load_combination_weights(weights_csv)
            # Determine weights type based on filename for proper sign handling
            self.weights_type = self._determine_weights_type(weights_csv)
        else:
            self.combination_weights = None
            self.weights_type = None

    def _determine_weights_type(self, weights_csv: str) -> str:
        """Determine if weights are speed-based or comfort-based from filename."""
        filename = weights_csv.lower()
        if 'speed' in filename:
            return 'speed'
        elif 'comfort' in filename:
            return 'comfort'
        else:
            # Try to determine from the actual correlations in the file
            import csv
            try:
                with open(weights_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    correlations = []
                    for row in reader:
                        try:
                            corr = float(row.get('correlation', '0'))
                            correlations.append(corr)
                        except ValueError:
                            continue
                    
                    if correlations:
                        avg_corr = sum(correlations) / len(correlations)
                        # If most correlations are negative, assume speed weights
                        return 'speed' if avg_corr < 0 else 'comfort'
            except:
                pass
            
            # Default fallback
            return 'speed'

    def _extract_bigrams(self) -> List[Tuple[str, str]]:
        """Extract all consecutive character pairs from text that exist in layout mapping.
        
        Spaces and other non-layout characters act as word boundaries that break bigram chains.
        """
        bigrams = []
        
        # Split text into words first (respecting boundaries)
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

    def calculate_scores(self) -> Dict:
        """Calculate layout score using empirical combination weighting or unweighted scoring."""
        if not self.bigrams:
            return {
                'layout_score': 0.0,
                'normalized_score': 0.0,
                'theoretical_maximum': 1.0,
                'average_weighted_score': 0.0,
                'total_weighted_score': 0.0,
                'bigram_count': 0,
                'individual_scores': {},
                'combination_breakdown': {},
                'bigram_details': [],
                'scoring_mode': 'unweighted' if self.combination_weights is None else 'weighted'
            }
        
        # Initialize accumulators for individual unweighted scores
        criterion_sums = defaultdict(float)
        criterion_counts = defaultdict(int)
        bigram_details = []
        
        # If we have weights, also track weighted scoring
        if self.combination_weights is not None:
            total_weighted_score = 0.0
            combination_counts = defaultdict(int)
            combination_score_sums = defaultdict(float)
        
        # Score each bigram
        for char1, char2 in self.bigrams:
            bigram_scores = self.score_bigram(char1, char2)
            
            # Always calculate individual criterion scores (unweighted)
            for criterion, score in bigram_scores.items():
                criterion_sums[criterion] += score
                criterion_counts[criterion] += 1
            
            # Calculate weighted score if weights available
            bigram_detail = {
                'bigram': f"{char1}{char2}",
                'scores': bigram_scores
            }
            
            if self.combination_weights is not None:
                weighted_score = score_bigram_weighted(bigram_scores, self.combination_weights)
                combination = identify_bigram_combination(bigram_scores)
                
                # Apply correct sign based on weights type
                # CRITICAL: Different weights types require different sign handling
                if self.weights_type == 'speed':
                    # Speed weights: negative correlation = good (faster typing)
                    # Example: correlation = -0.14 means higher Dvorak score → faster typing
                    # Flip sign so negative becomes positive contribution to layout score
                    final_score = -weighted_score
                else:  # comfort weights
                    # Comfort weights: positive correlation = good (more comfortable) 
                    # Example: correlation = +0.44 means higher Dvorak score → more comfortable
                    # Keep original sign so positive stays positive contribution
                    final_score = weighted_score
                
                total_weighted_score += final_score
                combination_counts[combination] += 1
                combination_score_sums[combination] += final_score
                
                bigram_detail.update({
                    'combination': combination,
                    'weighted_score': final_score
                })
            else:
                # For unweighted, use sum of individual scores
                unweighted_sum = sum(bigram_scores.values())
                bigram_detail.update({
                    'combination': tuple(sorted(bigram_scores.keys())),  # All criteria
                    'weighted_score': unweighted_sum
                })
            
            bigram_details.append(bigram_detail)
        
        # Calculate mean individual scores (always available)
        individual_scores = {}
        for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                        'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
            if criterion_counts[criterion] > 0:
                individual_scores[criterion] = criterion_sums[criterion] / criterion_counts[criterion]
            else:
                individual_scores[criterion] = 0.0
        
        # Prepare results based on whether we have weights
        if self.combination_weights is not None:
            # Weighted scoring results
            average_weighted_score = total_weighted_score / len(self.bigrams)
            
            # Calculate theoretical maximum and FIXED normalized score
            theoretical_max = self.calculate_theoretical_maximum()
            
            # FIXED NORMALIZATION: Proper 0-1 clamping while preserving negative score meaning
            if theoretical_max > 0:
                raw_ratio = average_weighted_score / theoretical_max
                normalized_score = max(0.0, min(1.0, raw_ratio))
            else:
                normalized_score = 0.0
            
            combination_breakdown = {}
            for combo, count in combination_counts.items():
                combination_breakdown[combo] = {
                    'count': count,
                    'total_contribution': combination_score_sums[combo],
                    'average_score': combination_score_sums[combo] / count if count > 0 else 0,
                    'percentage': count / len(self.bigrams) * 100
                }

            return {
                'layout_score': average_weighted_score,    # Primary metric (can be negative - meaningful!)
                'normalized_score': normalized_score,      # 0-1 clamped for comparison
                'theoretical_maximum': theoretical_max,    # Max possible for this weights type
                'average_weighted_score': average_weighted_score,
                'total_weighted_score': total_weighted_score,
                'bigram_count': len(self.bigrams),
                'individual_scores': individual_scores,
                'combination_breakdown': combination_breakdown,
                'bigram_details': bigram_details,
                'scoring_mode': 'weighted',
                'weights_type': self.weights_type
            }
        else:
            # Unweighted scoring results - use average of individual scores
            average_individual_score = sum(individual_scores.values()) / len(individual_scores)
            total_individual_score = average_individual_score * len(self.bigrams)
            
            return {
                'layout_score': average_individual_score,
                'normalized_score': average_individual_score,  # Already 0-1 normalized
                'theoretical_maximum': 1.0,  # Max possible for unweighted
                'average_weighted_score': average_individual_score,
                'total_weighted_score': total_individual_score,
                'bigram_count': len(self.bigrams),
                'individual_scores': individual_scores,
                'combination_breakdown': {},  # No combinations in unweighted mode
                'bigram_details': bigram_details,
                'scoring_mode': 'unweighted'
            }
        
    def calculate_theoretical_maximum(self) -> float:
        """Calculate theoretical maximum possible score for current weights type."""
        if self.combination_weights is None:
            return 1.0  # Unweighted maximum is 1.0 (average of all 1.0 scores)
        
        # For weighted scoring, find the highest magnitude correlation
        max_magnitude = 0.0
        
        for combination, weight in self.combination_weights.items():
            if combination:  # Skip empty combination
                # Calculate what this combination would contribute with perfect scores
                weighted_score = weight * 1.0  # All criteria in combination = 1.0
                
                # Apply same sign logic as actual scoring
                if self.weights_type == 'speed':
                    final_score = -weighted_score  # Speed: negative correlation = good, so flip
                else:  # comfort
                    final_score = weighted_score   # Comfort: positive correlation = good, so keep
                
                # Track the highest positive contribution possible
                if final_score > max_magnitude:
                    max_magnitude = final_score
        
        # Return the theoretical maximum
        return max_magnitude if max_magnitude > 0 else 1.0

    def get_detailed_breakdown(self) -> Dict[str, List[Dict]]:
        """Get detailed breakdown by criterion showing all bigrams with scores."""
        if not self.bigrams:
            return {}
        
        breakdown = defaultdict(list)
        
        for char1, char2 in self.bigrams:
            bigram_scores = self.score_bigram(char1, char2)
            pos1 = self.layout_mapping.get(char1, char1.upper())
            pos2 = self.layout_mapping.get(char2, char2.upper())
            
            for criterion, score in bigram_scores.items():
                breakdown[criterion].append({
                    'bigram': f"{char1}{char2}",
                    'positions': f"{pos1}{pos2}", 
                    'score': score
                })
        
        # Sort each criterion's bigrams by score (descending)
        for criterion in breakdown:
            breakdown[criterion].sort(key=lambda x: x['score'], reverse=True)
        
        return dict(breakdown)
        """Get detailed breakdown by criterion showing all bigrams with scores."""
        if not self.bigrams:
            return {}
        
        breakdown = defaultdict(list)
        
        for char1, char2 in self.bigrams:
            bigram_scores = self.score_bigram(char1, char2)
            pos1 = self.layout_mapping.get(char1, char1.upper())
            pos2 = self.layout_mapping.get(char2, char2.upper())
            
            for criterion, score in bigram_scores.items():
                breakdown[criterion].append({
                    'bigram': f"{char1}{char2}",
                    'positions': f"{pos1}{pos2}", 
                    'score': score
                })
        
        # Sort each criterion's bigrams by score (descending)
        for criterion in breakdown:
            breakdown[criterion].sort(key=lambda x: x['score'], reverse=True)
        
        return dict(breakdown)

def print_combination_results(results: Dict) -> None:
    """Print formatted results from scoring."""
    scoring_mode = results.get('scoring_mode', 'unknown')
    weights_type = results.get('weights_type', '')
    
    if scoring_mode == 'weighted':
        weights_desc = f" ({weights_type}-based)" if weights_type else ""
        print(f"Dvorak-9 Empirical Combination Scoring Results{weights_desc}")
    else:
        print("Dvorak-9 Unweighted Individual Criteria Results")
    
    print("=" * 70)
    
    # Primary metric
    print(f"Layout Score: {results['layout_score']:8.3f}")
    
    # Show normalized score for comparison
    normalized = results.get('normalized_score', results['layout_score'])
    theoretical_max = results.get('theoretical_maximum', 1.0)
    print(f"Normalized Score: {normalized:8.3f} (out of 1.0)")
    print(f"Theoretical Maximum: {theoretical_max:8.3f}")
    
    if scoring_mode == 'weighted':
        print(f"Average Weighted Score: {results['average_weighted_score']:8.3f}")
        if weights_type:
            print(f"Weights Type: {weights_type}")
    else:
        print(f"Average Individual Score: {results['average_weighted_score']:8.3f}")
    print(f"Bigrams Analyzed: {results['bigram_count']:8d}")
    print(f"Scoring Mode: {scoring_mode}")
    
    # Individual criteria breakdown
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
    
    # Combination breakdown (only for weighted scoring)
    if scoring_mode == 'weighted' and results.get('combination_breakdown'):
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

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate Dvorak-9 layout scores using empirical combination weighting or unweighted individual criteria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Weighted scoring with speed-based weights
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --weights-csv "weights/combinations_weights_from_speed_significant.csv"
  
  # Weighted scoring with comfort-based weights
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --weights-csv "weights/combinations_weights_from_comfort_significant.csv"
  
  # Unweighted scoring (0-1 individual criteria only)
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --no-weights
  
  # Score layout with details (unweighted)
  python dvorak9_scorer.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details --no-weights
  
  # Use items as text if no text provided
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --no-weights
  
  # Use text file
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --text-file "sample_text.txt" --no-weights

  # Return just the 10 scores (total + 9 individual scores)
  python dvorak9_scorer.py --items "abc" --positions "FDJ" --ten-scores --no-weights
  
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
    parser.add_argument("--no-weights", action="store_true",
                       help="Use unweighted scoring (individual 0-1 criteria only)")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed breakdown with examples")
    parser.add_argument("--csv", action="store_true",
                       help="Output in CSV format")
    parser.add_argument("--ten-scores", action="store_true",
                       help="Output only 10 scores: normalized score followed by 9 individual scores")
    parser.add_argument("--compare", action="store_true",
                       help="Show comparison table with normalized scores for better comparison")

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

        # Determine weights file
        if args.no_weights:
            weights_csv = None
        else:
            weights_csv = args.weights_csv

        # Calculate scores
        scorer = Dvorak9Scorer(layout_mapping, text, weights_csv)
        results = scorer.calculate_scores()

        if args.ten_scores:
            # Output 10 scores: layout score + 9 individual scores
            individual_scores = results.get('individual_scores', {})
            
            # Use normalized_score as the primary score for comparability
            scores = [results.get('normalized_score', results['layout_score'])]
            
            # Add individual scores in consistent order
            for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                            'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
                scores.append(individual_scores.get(criterion, 0.0))
            
            print(' '.join(f"{score:.6f}" for score in scores))

        elif args.csv:
            # CSV output
            print("metric,value")
            
            # Primary score
            scoring_mode = results.get('scoring_mode', 'unknown')
            weights_type = results.get('weights_type', '')
            print(f"layout_score,{results['layout_score']:.6f}")
            print(f"normalized_score,{results.get('normalized_score', results['layout_score']):.6f}")
            print(f"theoretical_maximum,{results.get('theoretical_maximum', 1.0):.6f}")
            print(f"scoring_mode,{scoring_mode}")
            if weights_type:
                print(f"weights_type,{weights_type}")
            print(f"average_score,{results['average_weighted_score']:.6f}")
            print(f"total_score,{results['total_weighted_score']:.6f}")
            print(f"bigram_count,{results['bigram_count']}")
            
            # Individual unweighted scores
            individual_scores = results.get('individual_scores', {})
            for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                             'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']:
                score = individual_scores.get(criterion, 0.0)
                print(f"individual_{criterion},{score:.6f}")
            
            # Top combinations (only for weighted scoring)
            if scoring_mode == 'weighted' and results.get('combination_breakdown'):
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
                        bigram_data = breakdown[key]
                        print(f"\n{name}:")
                        
                        # Show all bigrams with scores, sorted by score (highest first)
                        for item in bigram_data:
                            print(f"  {item['positions']}: {item['score']:.1f}", end="")
                            if len(bigram_data) <= 10:  # Show bigram letters if not too many
                                print(f" ({item['bigram']})")
                            else:
                                print()
                        
                        if len(bigram_data) == 0:
                            print("  (no bigrams found)")
        
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
        print("Dvorak-9 Scoring Example")
        print("="*50)
        
        layout_mapping = {'a': 'F', 'b': 'D', 'c': 'J'}
        text = "abacaba"
        
        try:
            # Try weighted scoring first
            scorer = Dvorak9Scorer(layout_mapping, text, "weights/combinations_weights_from_speed_significant.csv")
            results = scorer.calculate_scores()
            print("Weighted scoring:")
            print_combination_results(results)
        except FileNotFoundError:
            print("Weights file not found, using unweighted scoring:")
            # Fall back to unweighted scoring
            scorer = Dvorak9Scorer(layout_mapping, text, weights_csv=None)
            results = scorer.calculate_scores()
            print_combination_results(results)
        except Exception as e:
            print(f"Error: {e}")