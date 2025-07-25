# validate_dvorak9_all_bigrams.py
"""
Comprehensive validation test of Dvorak-9 scoring criteria.

This script systematically tests every possible bigram combination of a QWERTY layout
to validate that the 9 Dvorak criteria are implemented correctly and produce 
expected score patterns.

This version uses the canonical scoring implementation from dvorak9_scorer.py to
ensure consistency across the codebase.

Outputs:
1. 'dvorak9_scores_all_bigrams.csv' - All possible bigrams with individual scores and metadata
2. 'dvorak9_scores_unique_all_bigrams.csv' - Unique score patterns with counts and examples

Features:
- Tests all 841 possible bigrams (29×29 QWERTY keys)
- Validates specific test cases for each criterion
- Analyzes score distributions and patterns
- Identifies criteria with constant or low-variation scores
- Provides comprehensive debugging output for criterion implementation

Usage:
python validate_dvorak9_all_bigrams.py

Outputs:
dvorak9_scores_all_bigrams.csv
dvorak9_scores_unique_all_bigrams.csv
"""

import sys
import os
import csv
from collections import defaultdict

# Import the canonical scoring function
from dvorak9_scorer import score_bigram_dvorak9, get_key_info, QWERTY_LAYOUT

def get_all_qwerty_keys():
    """Get all standard QWERTY keys for testing."""
    # Use the main letter and punctuation keys
    return list("QWERTYUIOPASDFGHJKL;ZXCVBNM,./")

def test_all_bigrams(all_bigrams_file="dvorak9_scores_all_bigrams.csv", 
                    unique_scores_file="dvorak9_scores_unique_all_bigrams.csv"):
    """Test every possible bigram combination and output to two CSV files."""
    
    keys = get_all_qwerty_keys()
    print(f"Testing {len(keys)} keys = {len(keys)**2} possible bigrams")
    
    results = []
    score_patterns = defaultdict(list)  # Track bigrams for each unique score pattern
    
    # Test every possible bigram
    for key1 in keys:
        for key2 in keys:
            bigram = key1 + key2
            
            # Use the canonical scoring function
            bigram_scores = score_bigram_dvorak9(bigram)
            
            # Get key info for analysis
            row1, finger1, hand1 = get_key_info(key1)
            row2, finger2, hand2 = get_key_info(key2)
            
            # Create result row
            result = {
                'bigram': bigram,
                'pos1': key1,
                'pos2': key2,
                'hand1': hand1,
                'hand2': hand2,
                'finger1': finger1,
                'finger2': finger2,
                'row1': row1,
                'row2': row2,
                'hands': bigram_scores['hands'],
                'fingers': bigram_scores['fingers'],
                'skip_fingers': bigram_scores['skip_fingers'],
                'dont_cross_home': bigram_scores['dont_cross_home'],
                'same_row': bigram_scores['same_row'],
                'home_row': bigram_scores['home_row'],
                'columns': bigram_scores['columns'],
                'strum': bigram_scores['strum'],
                'strong_fingers': bigram_scores['strong_fingers'],
                'total': sum(bigram_scores.values())
            }
            
            results.append(result)
            
            # Track unique score combinations
            score_tuple = tuple(bigram_scores[c] for c in ['hands', 'fingers', 'skip_fingers', 
                                                          'dont_cross_home', 'same_row', 'home_row', 
                                                          'columns', 'strum', 'strong_fingers'])
            score_patterns[score_tuple].append(bigram)
    
    # Sort all results by total score (descending), then by bigram name
    results.sort(key=lambda x: (-x['total'], x['bigram']))
    
    # Write CSV with all bigrams
    fieldnames = ['bigram', 'total', 'pos1', 'pos2', 'hand1', 'hand2', 'finger1', 'finger2', 'row1', 'row2',
                  'hands', 'fingers', 'skip_fingers', 'dont_cross_home', 'same_row', 
                  'home_row', 'columns', 'strum', 'strong_fingers']
    
    with open(all_bigrams_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Write CSV with unique score patterns
    unique_fieldnames = ['total', 'count', 'hands', 'fingers', 'skip_fingers', 'dont_cross_home', 'same_row', 
                        'home_row', 'columns', 'strum', 'strong_fingers', 'example_bigrams']
    
    unique_results = []
    for score_tuple, bigram_list in score_patterns.items():
        total_score = sum(score_tuple)
        
        # Limit examples to first 10 bigrams to keep CSV readable
        example_bigrams = ', '.join(bigram_list[:10])
        if len(bigram_list) > 10:
            example_bigrams += f' ... (+{len(bigram_list)-10} more)'
        
        unique_result = {
            'hands': score_tuple[0],
            'fingers': score_tuple[1], 
            'skip_fingers': score_tuple[2],
            'dont_cross_home': score_tuple[3],
            'same_row': score_tuple[4],
            'home_row': score_tuple[5],
            'columns': score_tuple[6],
            'strum': score_tuple[7],
            'strong_fingers': score_tuple[8],
            'total': total_score,
            'count': len(bigram_list),
            'example_bigrams': example_bigrams
        }
        unique_results.append(unique_result)
    
    # Sort by total score descending, then by count descending
    unique_results.sort(key=lambda x: (-x['total'], -x['count']))
    
    with open(unique_scores_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=unique_fieldnames)
        writer.writeheader()
        writer.writerows(unique_results)
    
    print(f"All bigrams written to {all_bigrams_file}")
    print(f"Unique score patterns written to {unique_scores_file}")
    print(f"Total bigrams: {len(results)}")
    print(f"Unique score combinations: {len(score_patterns)}")
    
    return results, score_patterns

def analyze_score_patterns(score_patterns):
    """Analyze patterns in the scoring results."""
    
    print("\n=== SCORE PATTERN ANALYSIS ===")
    
    print(f"Found {len(score_patterns)} unique score patterns:")
    
    # Show most common patterns
    sorted_patterns = sorted(score_patterns.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (pattern, bigrams) in enumerate(sorted_patterns[:10]):
        print(f"\nPattern {i+1} ({len(bigrams)} bigrams): {pattern}")
        print(f"  Total score: {sum(pattern):.1f}")
        print(f"  Examples: {', '.join(bigrams[:8])}{'...' if len(bigrams) > 8 else ''}")

def analyze_criterion_distributions(results):
    """Show distribution of scores for each criterion."""
    
    print("\n=== CRITERION SCORE DISTRIBUTIONS ===")
    
    criteria = ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 'same_row', 
                'home_row', 'columns', 'strum', 'strong_fingers']
    
    for criterion in criteria:
        scores = [result[criterion] for result in results]
        score_counts = defaultdict(int)
        for score in scores:
            score_counts[score] += 1
        
        print(f"\n{criterion.upper()}:")
        for score in sorted(score_counts.keys()):
            count = score_counts[score]
            pct = (count / len(results)) * 100
            print(f"  {score}: {count:4d} bigrams ({pct:5.1f}%)")

def test_specific_cases():
    """Test specific cases to verify key criteria are working."""
    
    print("\n=== SPECIFIC TEST CASES ===")
    
    # Test cases: (description, bigram, expected_criterion_scores)
    test_cases = [
        # Strum tests
        ("Same finger strum", "FF", {"strum": 0.0}),
        ("Different hands", "FJ", {"strum": 1.0, "hands": 1.0}),
        ("Inward roll (L)", "AF", {"strum": 1.0}),  # pinky to index
        ("Outward roll (L)", "FA", {"strum": 0.0}), # index to pinky
        
        # Skip fingers tests
        ("Same finger", "FF", {"skip_fingers": 0.0}),
        ("Adjacent fingers", "FD", {"skip_fingers": 0.3}),  # index to middle
        ("Skip 1 finger", "FS", {"skip_fingers": 0.7}),    # index to ring
        ("Skip 2 fingers", "FA", {"skip_fingers": 1.0}),   # index to pinky
        ("Different hands", "FJ", {"skip_fingers": 1.0}),
        
        # Don't cross home tests
        ("Same hand hurdling", "QZ", {"dont_cross_home": 0.0}),  # top to bottom, same hand
        ("Different hands hurdling", "QM", {"dont_cross_home": 1.0}), # different hands always 1
        ("Same hand no hurdling", "QA", {"dont_cross_home": 1.0}),   # top to home, ok
        
        # Home row tests
        ("Both home row", "FJ", {"home_row": 1.0}),
        ("One home row", "FR", {"home_row": 0.5}),
        ("Neither home row", "QZ", {"home_row": 0.0}),
    ]
    
    for description, bigram, expected in test_cases:
        # Use the canonical scoring function
        bigram_scores = score_bigram_dvorak9(bigram)
        
        print(f"\n{description}: {bigram}")
        
        all_correct = True
        for criterion, expected_score in expected.items():
            actual_score = bigram_scores[criterion]
            status = "✓" if abs(actual_score - expected_score) < 0.01 else "✗"
            print(f"  {status} {criterion}: expected {expected_score}, got {actual_score}")
            if abs(actual_score - expected_score) >= 0.01:
                all_correct = False
        
        if all_correct:
            print(f"  ✓ All checks passed")
        else:
            print(f"  ✗ Some checks failed")

def validate_canonical_implementation():
    """Validate that the canonical implementation works as expected."""
    
    print("\n=== CANONICAL IMPLEMENTATION VALIDATION ===")
    
    # Test a few sample bigrams to ensure the canonical function works
    test_bigrams = ["TH", "ER", "AN", "IN", "ON"]
    
    print("Testing canonical scoring function:")
    for bigram in test_bigrams:
        try:
            scores = score_bigram_dvorak9(bigram)
            print(f"  {bigram}: ✓ (total score: {sum(scores.values()):.2f})")
            
            # Validate score structure
            expected_keys = {'hands', 'fingers', 'skip_fingers', 'dont_cross_home', 
                           'same_row', 'home_row', 'columns', 'strum', 'strong_fingers'}
            actual_keys = set(scores.keys())
            
            if expected_keys != actual_keys:
                print(f"    ⚠️  Score keys mismatch!")
                print(f"      Expected: {expected_keys}")
                print(f"      Actual: {actual_keys}")
            
            # Validate score ranges (should be 0-1)
            for criterion, score in scores.items():
                if not (0 <= score <= 1):
                    print(f"    ⚠️  {criterion} score out of range: {score}")
                    
        except Exception as e:
            print(f"  {bigram}: ✗ Error: {e}")
    
    print("\n✓ Canonical implementation validation complete")

def main():
    """Run comprehensive test."""
    
    print("Dvorak-9 Comprehensive Scoring Test")
    print("=" * 50)
    print("Using canonical scoring implementation from dvorak9_scorer.py")
    print("=" * 50)
    
    # Validate the canonical implementation first
    validate_canonical_implementation()
    
    # Test specific cases
    test_specific_cases()
    
    # Test all possible bigrams
    print(f"\n{'='*50}")
    print("Testing all possible bigrams...")
    results, score_patterns = test_all_bigrams()
    
    # Analyze patterns
    analyze_score_patterns(score_patterns)
    analyze_criterion_distributions(results)
    
    print(f"\n{'='*50}")
    print("Test complete!")
    print("✅ SUCCESS: Using canonical implementation from dvorak9_scorer.py")
    print("Check these output files:")
    print("  - 'dvorak9_scores_all_bigrams.csv' for all possible bigrams")
    print("  - 'dvorak9_scores_unique_all_bigrams.csv' for unique score patterns")

if __name__ == "__main__":
    main()