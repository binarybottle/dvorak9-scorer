#!/usr/bin/env python3
"""
Test correlations between Dvorak-10 criteria and typing speed.

This script analyzes:
1. Sentence typing times (criteria 1-2: two-hand criteria)
2. Bigram interkey intervals (criteria 3-10: same-hand criteria)

Updates:
- Split analysis by middle column inclusion (T,G,B,Y,H,N)
- Each criterion calculated separately for sequences with/without middle columns
- Confirmed hand balance/alternation measured on words for speed correlation

Usage:
    python test_dvorak10_correlations.py
"""

import csv
import sys
import os
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.stats.multitest import multipletests

# Machine learning imports for interaction analysis
try:
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.metrics import r2_score
    sklearn_available = True
except ImportError:
    print("Warning: scikit-learn not available. Interaction analysis will be limited.")
    sklearn_available = False

# Import the Dvorak10Scorer from the provided script
try:
    from score_dvorak10 import Dvorak10Scorer
except ImportError:
    print("Error: Could not import score_layout_dvorak10.py")
    print("Make sure the file is in the same directory as this script.")
    sys.exit(1)

# Standard QWERTY layout mapping for testing
QWERTY_ITEMS = "abcdefghijklmnopqrstuvwxyz;,./"
QWERTY_POSITIONS = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./"

# Define middle column keys (index finger stretch keys)
MIDDLE_COLUMN_KEYS = {'t', 'g', 'b', 'y', 'h', 'n'}

# Global file handle for text output
output_file = None

def print_and_log(*args, **kwargs):
    """Print to console and write to log file."""
    print(*args, **kwargs)
    if output_file:
        print(*args, **kwargs, file=output_file)
        output_file.flush()

def create_qwerty_mapping():
    """Create standard QWERTY layout mapping."""
    return dict(zip(QWERTY_ITEMS.lower(), QWERTY_POSITIONS.upper()))

def contains_middle_column_key(sequence, layout_mapping):
    """Check if sequence contains any middle column keys (T,G,B,Y,H,N)."""
    for char in sequence.lower():
        if char in layout_mapping:
            pos = layout_mapping[char].lower()
            if pos in MIDDLE_COLUMN_KEYS:
                return True
    return False

def split_data_by_middle_columns(sequences, times, layout_mapping):
    """Split sequences and times into two groups: with and without middle column keys."""
    with_middle = {'sequences': [], 'times': []}
    without_middle = {'sequences': [], 'times': []}
    
    for seq, time in zip(sequences, times):
        if contains_middle_column_key(seq, layout_mapping):
            with_middle['sequences'].append(seq)
            with_middle['times'].append(time)
        else:
            without_middle['sequences'].append(seq)
            without_middle['times'].append(time)
    
    return with_middle, without_middle

def read_bigram_times(filename, min_threshold=50, max_threshold=2000, use_percentile_filter=False):
    """Read bigram times from CSV file with optional filtering."""
    bigrams = []
    times = []
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                bigram = row['bigram'].lower().strip()
                time = float(row['interkey_interval'])
                
                # Only include bigrams with characters in our layout
                if len(bigram) == 2 and all(c in QWERTY_ITEMS.lower() for c in bigram):
                    bigrams.append(bigram)
                    times.append(time)
    
    except FileNotFoundError:
        print_and_log(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print_and_log(f"Error reading {filename}: {e}")
        return [], []
    
    if not times:
        return bigrams, times
    
    # Apply filtering
    original_count = len(times)
    
    if use_percentile_filter:
        p5 = np.percentile(times, 5)
        p95 = np.percentile(times, 95)
        filtered_indices = [i for i, t in enumerate(times) if p5 <= t <= p95]
        filter_method = f"5th-95th percentile ({p5:.1f}-{p95:.1f}ms)"
    else:
        filtered_indices = [i for i, t in enumerate(times) if min_threshold <= t <= max_threshold]
        filter_method = f"absolute thresholds ({min_threshold}-{max_threshold}ms)"
    
    # Apply filtering
    filtered_bigrams = [bigrams[i] for i in filtered_indices]
    filtered_times = [times[i] for i in filtered_indices]
    
    removed_count = original_count - len(filtered_times)
    if removed_count > 0:
        print_and_log(f"Filtered {removed_count}/{original_count} bigrams using {filter_method}")
        print_and_log(f"  Kept {len(filtered_times)} bigrams ({len(filtered_times)/original_count*100:.1f}%)")
        print_and_log(f"  Time range: {min(filtered_times):.1f} - {max(filtered_times):.1f}ms")
    
    return filtered_bigrams, filtered_times

def read_word_times(filename, max_threshold=None, use_percentile_filter=False):
    """Read word times from CSV file with optional filtering."""
    words = []
    times = []
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                word = row['word'].strip()
                time = float(row['time'])
                
                words.append(word)
                times.append(time)
    
    except FileNotFoundError:
        print_and_log(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print_and_log(f"Error reading {filename}: {e}")
        return [], []
    
    if not times or max_threshold is None:
        return words, times
    
    # Apply filtering 
    original_count = len(times)
    
    if use_percentile_filter:
        p95 = np.percentile(times, 95)
        filtered_indices = [i for i, t in enumerate(times) if t <= p95]
        filter_method = f"95th percentile ({p95:.1f}ms)"
    else:
        filtered_indices = [i for i, t in enumerate(times) if t <= max_threshold]
        filter_method = f"absolute threshold ({max_threshold}ms)"
    
    # Apply filtering
    filtered_words = [words[i] for i in filtered_indices]
    filtered_times = [times[i] for i in filtered_indices]
    
    removed_count = original_count - len(filtered_times)
    if removed_count > 0:
        print_and_log(f"Filtered {removed_count}/{original_count} words using {filter_method}")
        print_and_log(f"  Kept {len(filtered_times)} words ({len(filtered_times)/original_count*100:.1f}%)")
        print_and_log(f"  Time range: {min(filtered_times):.1f} - {max(filtered_times):.1f}ms")
    
    return filtered_words, filtered_times

def analyze_criteria_for_group(sequences, times, layout_mapping, criteria_names, group_name):
    """Analyze correlations for a specific group of sequences."""
    print_and_log(f"\n--- Analyzing {group_name} ---")
    print_and_log(f"Sequences: {len(sequences)}")
    
    if len(sequences) < 10:
        print_and_log(f"Too few sequences for reliable analysis ({len(sequences)})")
        return {}
    
    # Show some examples
    examples = sequences[:5] if len(sequences) >= 5 else sequences
    print_and_log(f"Examples: {', '.join(examples)}")
    
    results = {}
    
    # Collect scores for each criterion
    criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    valid_times = []
    valid_sequences = []
    
    for seq, time in zip(sequences, times):
        try:
            scorer = Dvorak10Scorer(layout_mapping, seq)
            scores = scorer.calculate_all_scores()
            
            # Check if we have relevant data for the criteria we're analyzing
            if group_name.startswith("Same-Hand"):
                # For same-hand criteria, need same-hand digraphs
                has_same_hand = any(
                    scores[criterion]['details'].get('total_same_hand', 0) > 0
                    for criterion in criteria_names.keys()
                )
                if not has_same_hand:
                    continue
            
            valid_sequences.append(seq)
            valid_times.append(time)
            
            for criterion in criteria_names.keys():
                score = scores[criterion]['score']
                criterion_scores[criterion].append(score)
        
        except Exception as e:
            print_and_log(f"Error processing sequence '{seq}': {e}")
            continue
    
    print_and_log(f"Valid sequences for analysis: {len(valid_sequences)}")
    
    # Calculate correlations
    for criterion, scores in criterion_scores.items():
        if len(scores) >= 3:  # Need at least 3 points for correlation
            try:
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(scores, valid_times)
                
                # Spearman correlation (rank-based, more robust)
                spearman_r, spearman_p = spearmanr(scores, valid_times)
                
                results[criterion] = {
                    'name': criteria_names[criterion],
                    'group': group_name,
                    'n_samples': len(scores),
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'scores': scores.copy(),
                    'times': valid_times.copy()
                }
                
            except Exception as e:
                print_and_log(f"Error calculating correlation for {criterion}: {e}")
                continue
    
    return results

def analyze_bigram_correlations(bigrams, times):
    """Analyze correlations between same-hand criteria (3-10) and bigram times, split by middle column inclusion."""
    layout_mapping = create_qwerty_mapping()
    
    # Criteria 3-10 (same-hand criteria)
    criteria_names = {
        '3_different_fingers': 'Two Fingers',
        '4_non_adjacent_fingers': 'Skip Fingers', 
        '5_home_block': 'Home Block',
        '6_dont_skip_home': "Don't Skip Home",
        '7_same_row': 'Same Row',
        '8_include_home': 'Include Home',
        '9_roll_inward': 'Roll Inward',
        '10_strong_fingers': 'Strong Fingers'
    }
    
    print_and_log("Analyzing bigram correlations...")
    print_and_log(f"Total bigrams: {len(bigrams)}")
    print_and_log(f"Middle column keys: {', '.join(sorted(MIDDLE_COLUMN_KEYS))}")
    
    # Split data by middle column inclusion
    with_middle, without_middle = split_data_by_middle_columns(bigrams, times, layout_mapping)
    
    print_and_log(f"\nData split:")
    print_and_log(f"  With middle columns: {len(with_middle['sequences'])} bigrams")
    print_and_log(f"  Without middle columns: {len(without_middle['sequences'])} bigrams")
    
    # Analyze each group separately
    results = {}
    
    # Group 1: Bigrams WITHOUT middle column keys
    if without_middle['sequences']:
        group_results = analyze_criteria_for_group(
            without_middle['sequences'], 
            without_middle['times'], 
            layout_mapping, 
            criteria_names, 
            "Same-Hand Bigrams (No Middle Columns)"
        )
        for criterion, data in group_results.items():
            results[f"{criterion}_no_middle"] = data
    
    # Group 2: Bigrams WITH middle column keys
    if with_middle['sequences']:
        group_results = analyze_criteria_for_group(
            with_middle['sequences'], 
            with_middle['times'], 
            layout_mapping, 
            criteria_names, 
            "Same-Hand Bigrams (With Middle Columns)"
        )
        for criterion, data in group_results.items():
            results[f"{criterion}_with_middle"] = data
    
    return results

def analyze_word_correlations(words, times):
    """Analyze correlations between two-hand criteria (1-2) and word times, split by middle column inclusion."""
    layout_mapping = create_qwerty_mapping()
    
    # Criteria 1-2 (two-hand criteria) - analyzed at WORD level
    criteria_names = {
        '1_hand_balance': 'Hand Balance',
        '2_hand_alternation': 'Hand Alternation'
    }
    
    print_and_log("Analyzing word correlations...")
    print_and_log("NOTE: Hand Balance and Hand Alternation are measured at WORD level")
    print_and_log("for correlation with word typing speed (not bigram speed).")
    print_and_log(f"Total words: {len(words)}")
    
    # Split data by middle column inclusion
    with_middle, without_middle = split_data_by_middle_columns(words, times, layout_mapping)
    
    print_and_log(f"\nData split:")
    print_and_log(f"  With middle columns: {len(with_middle['sequences'])} words")
    print_and_log(f"  Without middle columns: {len(without_middle['sequences'])} words")
    
    # Analyze each group separately
    results = {}
    
    # Group 1: Words WITHOUT middle column keys
    if without_middle['sequences']:
        group_results = analyze_criteria_for_group(
            without_middle['sequences'], 
            without_middle['times'], 
            layout_mapping, 
            criteria_names, 
            "Two-Hand Words (No Middle Columns)"
        )
        for criterion, data in group_results.items():
            results[f"{criterion}_no_middle"] = data
    
    # Group 2: Words WITH middle column keys
    if with_middle['sequences']:
        group_results = analyze_criteria_for_group(
            with_middle['sequences'], 
            with_middle['times'], 
            layout_mapping, 
            criteria_names, 
            "Two-Hand Words (With Middle Columns)"
        )
        for criterion, data in group_results.items():
            results[f"{criterion}_with_middle"] = data
    
    return results

def apply_multiple_comparisons_correction(results, alpha=0.05):
    """Apply multiple comparisons correction to p-values."""
    if not results:
        return results
        
    print_and_log(f"\nMultiple Comparisons Correction")
    print_and_log("=" * 50)
    
    # Extract p-values
    p_values = []
    result_keys = []
    
    for key, data in results.items():
        if 'spearman_p' in data:
            p_values.append(data['spearman_p'])
            result_keys.append(key)
    
    if not p_values:
        print_and_log("No p-values found for correction.")
        return results
    
    # Apply FDR correction
    from statsmodels.stats.multitest import multipletests
    rejected, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    print_and_log(f"Original α = {alpha}")
    print_and_log(f"Number of tests: {len(p_values)}")
    print_and_log(f"FDR-corrected significant results:")
    
    # Update results with corrected p-values
    corrected_results = results.copy()
    any_significant = False
    
    for i, key in enumerate(result_keys):
        corrected_results[key]['spearman_p_corrected'] = p_adjusted[i]
        corrected_results[key]['significant_corrected'] = rejected[i]
        
        if rejected[i]:
            any_significant = True
            data = corrected_results[key]
            r = data['spearman_r']
            name = data['name']
            group = data['group']
            direction = "Better criterion → Faster typing" if r < 0 else "Better criterion → Slower typing"
            print_and_log(f"  ✓ {name} ({group}): r={r:.3f}, p_adj={p_adjusted[i]:.3f} ({direction})")
    
    if not any_significant:
        print_and_log("  None remain significant after correction")
    
    return corrected_results

def interpret_effect_sizes(results):
    """Interpret practical significance of correlations."""
    print_and_log("\nPractical Significance Interpretation")
    print_and_log("=" * 50)
    print_and_log("Cohen's guidelines for correlation effect sizes:")
    print_and_log("  Small effect:    |r| = 0.10-0.30")
    print_and_log("  Medium effect:   |r| = 0.30-0.50") 
    print_and_log("  Large effect:    |r| = 0.50+")
    print_and_log("\nWith your large sample size, focus on effect size, not p-values!")
    print_and_log("-" * 50)
    
    if not results:
        return
    
    # Categorize effects by group
    groups = {}
    for criterion, data in results.items():
        group = data['group']
        if group not in groups:
            groups[group] = {'small': [], 'medium': [], 'large': []}
        
        r = abs(data['spearman_r'])
        name = data['name']
        direction = "↓" if data['spearman_r'] < 0 else "↑"
        
        if r >= 0.50:
            groups[group]['large'].append((name, data['spearman_r'], direction))
        elif r >= 0.30:
            groups[group]['medium'].append((name, data['spearman_r'], direction))
        elif r >= 0.10:
            groups[group]['small'].append((name, data['spearman_r'], direction))
    
    # Print categorized results by group
    for group_name, effects in groups.items():
        print_and_log(f"\n{group_name}:")
        
        if effects['large']:
            print_and_log("  LARGE EFFECTS (|r| ≥ 0.50):")
            for name, r, direction in sorted(effects['large'], key=lambda x: abs(x[1]), reverse=True):
                print_and_log(f"    {direction} {name}: r = {r:.3f}")
        
        if effects['medium']:
            print_and_log("  MEDIUM EFFECTS (0.30 ≤ |r| < 0.50):")
            for name, r, direction in sorted(effects['medium'], key=lambda x: abs(x[1]), reverse=True):
                print_and_log(f"    {direction} {name}: r = {r:.3f}")
        
        if effects['small']:
            print_and_log("  SMALL EFFECTS (0.10 ≤ |r| < 0.30):")
            for name, r, direction in sorted(effects['small'], key=lambda x: abs(x[1]), reverse=True):
                print_and_log(f"    {direction} {name}: r = {r:.3f}")
        
        if not any(effects.values()):
            print_and_log("  No meaningful effects (all |r| < 0.10)")

def print_correlation_results(results, title):
    """Print correlation results in a formatted table."""
    print_and_log(f"\n{title}")
    print_and_log("=" * 100)
    print_and_log("Note: Negative correlation = higher score → faster typing (good)")
    print_and_log("      Positive correlation = higher score → slower typing (bad)")
    print_and_log("-" * 100)
    
    if not results:
        print_and_log("No valid correlations found.")
        return
    
    # Group results by analysis group
    groups = {}
    for criterion, data in results.items():
        group = data['group']
        if group not in groups:
            groups[group] = []
        groups[group].append((criterion, data))
    
    # Print each group
    for group_name, group_data in groups.items():
        print_and_log(f"\n{group_name}:")
        print_and_log(f"{'Criterion':<15} {'N':<6} {'Pearson r':<10} {'p-val':<8} {'Spearman r':<11} {'p-val':<8} {'Mean±SD':<12}")
        print_and_log("-" * 80)
        
        # Sort by Spearman correlation (more robust)
        sorted_data = sorted(group_data, key=lambda x: abs(x[1]['spearman_r']), reverse=True)
        
        for criterion, data in sorted_data:
            name = data['name']
            n = data['n_samples']
            pr = data['pearson_r']
            pp = data['pearson_p']
            sr = data['spearman_r']
            sp = data['spearman_p']
            mean = data['mean_score']
            std = data['std_score']
            
            # Significance indicators
            p_sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
            s_sig = "***" if sp < 0.001 else "**" if sp < 0.01 else "*" if sp < 0.05 else ""
            
            print_and_log(f"{name:<15} {n:<6} {pr:>7.3f}{p_sig:<3} {pp:<8.3f} {sr:>7.3f}{s_sig:<4} {sp:<8.3f} {mean:.2f}±{std:.2f}")

def create_comparison_plots(bigram_results, word_results, output_dir="plots"):
    """Create comparison plots showing differences between middle column groups."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Combine all results
    all_results = {}
    all_results.update(bigram_results)
    all_results.update(word_results)
    
    if not all_results:
        return
    
    # Create correlation comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Separate data by middle column inclusion
    no_middle_criteria = []
    no_middle_correlations = []
    with_middle_criteria = []
    with_middle_correlations = []
    
    for criterion, data in all_results.items():
        if '_no_middle' in criterion:
            base_name = data['name']
            no_middle_criteria.append(base_name)
            no_middle_correlations.append(data['spearman_r'])
        elif '_with_middle' in criterion:
            base_name = data['name']
            with_middle_criteria.append(base_name)
            with_middle_correlations.append(data['spearman_r'])
    
    # Plot 1: No middle columns
    if no_middle_criteria:
        ax1.barh(range(len(no_middle_criteria)), no_middle_correlations, 
                color=['red' if r > 0 else 'blue' for r in no_middle_correlations], alpha=0.7)
        ax1.set_yticks(range(len(no_middle_criteria)))
        ax1.set_yticklabels(no_middle_criteria)
        ax1.set_xlabel('Spearman Correlation')
        ax1.set_title('Correlations: No Middle Columns\n(Blue=Good, Red=Bad)')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: With middle columns
    if with_middle_criteria:
        ax2.barh(range(len(with_middle_criteria)), with_middle_correlations,
                color=['red' if r > 0 else 'blue' for r in with_middle_correlations], alpha=0.7)
        ax2.set_yticks(range(len(with_middle_criteria)))
        ax2.set_yticklabels(with_middle_criteria)
        ax2.set_xlabel('Spearman Correlation')
        ax2.set_title('Correlations: With Middle Columns\n(Blue=Good, Red=Bad)')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/middle_column_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return "middle_column_comparison.png"

def print_summary(results):
    """Print summary of findings comparing middle column groups."""
    print_and_log("\n" + "=" * 80)
    print_and_log("SUMMARY OF FINDINGS")
    print_and_log("=" * 80)
    
    print_and_log("\nSignificant correlations (p < 0.05) by group:")
    
    # Group results
    groups = {}
    for criterion, data in results.items():
        group = data['group']
        if group not in groups:
            groups[group] = []
        
        if data['spearman_p'] < 0.05:
            direction = "GOOD" if data['spearman_r'] < 0 else "BAD"
            groups[group].append({
                'name': data['name'],
                'r': data['spearman_r'],
                'p': data['spearman_p'],
                'direction': direction
            })
    
    for group_name, group_results in groups.items():
        print_and_log(f"\n{group_name}:")
        if group_results:
            for result in group_results:
                print_and_log(f"  {result['name']}: r={result['r']:.3f}, p={result['p']:.3f} [{result['direction']}]")
        else:
            print_and_log("  No statistically significant correlations found.")
    
    print_and_log("\nInterpretation:")
    print_and_log("- GOOD: Higher criterion score → faster typing (validates Dvorak's principle)")
    print_and_log("- BAD: Higher criterion score → slower typing (contradicts Dvorak's principle)")
    print_and_log("- Compare results between groups to see if middle columns affect criterion validity")

def main():
    """Main analysis function."""
    global output_file
    
    # Open output file
    output_file = open('dvorak_analysis_results_split.txt', 'w', encoding='utf-8')
    
    try:
        print_and_log("Dvorak-10 Criteria Correlation Analysis (Split by Middle Columns)")
        print_and_log("=" * 70)
        print_and_log(f"Middle column keys: {', '.join(sorted(MIDDLE_COLUMN_KEYS))}")
        print_and_log("Analysis splits all sequences into two groups:")
        print_and_log("  1. Sequences WITHOUT any middle column keys")
        print_and_log("  2. Sequences WITH at least one middle column key")
        print_and_log()
        
        # Read data files
        print_and_log("Reading data files...")
        bigram_times_file = '../process_3.5M_keystrokes/output/bigram_times.csv'
        word_times_file = '../process_3.5M_keystrokes/output/word_times.csv'
        
        # FILTERING PARAMETERS
        MIN_INTERVAL = 50
        MAX_INTERVAL = 2000
        USE_PERCENTILE_BIGRAMS = False
        
        MAX_WORD_TIME = None
        USE_PERCENTILE_WORDS = False
        
        # Read the data files
        bigrams, bigram_times = read_bigram_times(
            bigram_times_file, 
            min_threshold=MIN_INTERVAL,
            max_threshold=MAX_INTERVAL, 
            use_percentile_filter=USE_PERCENTILE_BIGRAMS
        )
        
        words, word_times = read_word_times(
            word_times_file,
            max_threshold=MAX_WORD_TIME,
            use_percentile_filter=USE_PERCENTILE_WORDS
        )
        
        if not bigrams and not words:
            print_and_log("Error: No valid data found in CSV files.")
            sys.exit(1)
        
        # Create output directory for plots
        Path("plots").mkdir(exist_ok=True)
        print_and_log("Creating plots in 'plots/' directory...")
        
        # Analyze correlations
        bigram_results = {}
        word_results = {}
        
        if bigrams:
            bigram_results = analyze_bigram_correlations(bigrams, bigram_times)
        
        if words:
            word_results = analyze_word_correlations(words, word_times)
        
        # Combine all results
        all_results = {}
        all_results.update(bigram_results)
        all_results.update(word_results)
        
        if all_results:
            # Apply multiple comparisons correction
            all_results = apply_multiple_comparisons_correction(all_results)
            
            # Interpret effect sizes
            interpret_effect_sizes(all_results)
            
            # Print correlation tables
            print_correlation_results(all_results, "CORRELATION ANALYSIS BY MIDDLE COLUMN INCLUSION")
            
            # Create comparison plots
            create_comparison_plots(bigram_results, word_results)
            
            # Print summary
            print_summary(all_results)
        
        print_and_log(f"\n" + "=" * 60)
        print_and_log("ANALYSIS COMPLETE")
        print_and_log("=" * 60)
        print_and_log(f"Key outputs saved:")
        print_and_log(f"- Text output: dvorak_analysis_results_split.txt")
        print_and_log(f"- Comparison plots: plots/middle_column_comparison.png")
        print_and_log(f"- Each criterion analyzed separately for middle column inclusion")
        
    finally:
        if output_file:
            output_file.close()

if __name__ == "__main__":
    main()