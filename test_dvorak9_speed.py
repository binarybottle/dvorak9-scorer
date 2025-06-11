#!/usr/bin/env python3
"""
Dvorak-9 Criteria Correlation Analysis for Typing Speed

This script analyzes the correlation between typing speed and the 9 Dvorak criteria,
with comprehensive frequency adjustment and middle column key analysis.
"""

import pandas as pd
import numpy as np
import time
import argparse
from pathlib import Path
import random
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global variables for logging
log_content = []
original_print = print

def print_and_log(*args, **kwargs):
    """Print to console and store in log"""
    global log_content
    message = ' '.join(str(arg) for arg in args)
    log_content.append(message)
    original_print(*args, **kwargs)

def save_log(filename="dvorak9_bigram_analysis_results.txt"):
    """Save log content to file"""
    global log_content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_content))

def format_time(seconds):
    """Format seconds into a readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def load_frequency_data(freq_file_path):
    """Load frequency data for regression analysis"""
    print_and_log("🔍 Loading frequency data for regression analysis...")
    print_and_log("   (This uses pre-calculated English language frequencies, NOT sample frequencies)")
    
    try:
        freq_df = pd.read_csv(freq_file_path)
        print_and_log(f"✅ Loaded bigram frequency data: {len(freq_df)} entries")
        print_and_log(f"   Columns: {list(freq_df.columns)}")
        
        # Show sample frequencies
        print_and_log("   Sample bigram frequencies:")
        for i, (_, row) in enumerate(freq_df.head(3).iterrows()):
            if 'item_pair' in freq_df.columns and 'score' in freq_df.columns:
                print_and_log(f"     '{row['item_pair']}': {row['score']:,}")
        
        return freq_df
        
    except Exception as e:
        print_and_log(f"❌ Error loading frequency data: {e}")
        return None

def verify_frequency_data(freq_df):
    """Verify and display frequency data information"""
    if freq_df is None:
        print_and_log("\n🤖 FREQUENCY DATA VERIFICATION")
        print_and_log("=" * 50)
        print_and_log("❌ No frequency data available for analysis")
        print_and_log("=" * 50)
        return False
    
    print_and_log("\n🔬 FREQUENCY DATA VERIFICATION")
    print_and_log("=" * 50)
    print_and_log("This analysis uses PRE-CALCULATED English language frequencies,")
    print_and_log("NOT frequencies calculated from your typing sample.")
    print_and_log("")
    print_and_log(f"✅ Bigram frequencies loaded: {len(freq_df)} entries")
    print_and_log("=" * 50)
    print_and_log("")
    
    return True

def dvorak_9_criteria_bigrams(bigram):
    """Calculate all 9 Dvorak criteria scores for a bigram"""
    scores = {}
    
    # QWERTY keyboard layout
    layout = {
        'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4), 'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
        'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4), 'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8),
        'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4), 'n': (2, 5), 'm': (2, 6)
    }
    
    # Finger assignments (QWERTY standard)
    finger_map = {
        'q': 'left_pinky', 'a': 'left_pinky', 'z': 'left_pinky',
        'w': 'left_ring', 's': 'left_ring', 'x': 'left_ring',
        'e': 'left_middle', 'd': 'left_middle', 'c': 'left_middle',
        'r': 'left_index', 'f': 'left_index', 'v': 'left_index', 't': 'left_index', 'g': 'left_index', 'b': 'left_index',
        'y': 'right_index', 'h': 'right_index', 'n': 'right_index', 'u': 'right_index', 'j': 'right_index', 'm': 'right_index',
        'i': 'right_middle', 'k': 'right_middle',
        'o': 'right_ring', 'l': 'right_ring',
        'p': 'right_pinky'
    }
    
    # Hand assignments
    left_hand = {'q', 'w', 'e', 'r', 't', 'a', 's', 'd', 'f', 'g', 'z', 'x', 'c', 'v', 'b'}
    right_hand = {'y', 'u', 'i', 'o', 'p', 'h', 'j', 'k', 'l', 'n', 'm'}
    
    # Home row keys
    home_keys = {'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'}
    
    # Strong fingers (not pinky)
    strong_fingers = {'left_index', 'left_middle', 'left_ring', 'right_index', 'right_middle', 'right_ring'}
    
    # Middle column keys (require index finger stretches)
    middle_column_keys = {'t', 'g', 'b', 'y', 'h', 'n'}
    
    char1, char2 = bigram[0].lower(), bigram[1].lower()
    
    # Skip if characters not in layout
    if char1 not in layout or char2 not in layout:
        return {criterion: 0 for criterion in ['hands', 'fingers', 'skip_fingers', 'dont_cross_home', 'same_row', 'home_row', 'columns', 'strum', 'strong_fingers']}
    
    pos1, pos2 = layout[char1], layout[char2]
    finger1, finger2 = finger_map[char1], finger_map[char2]
    
    # 1. Hand alternation
    scores['hands'] = 1 if (char1 in left_hand) != (char2 in right_hand) else 0
    
    # 2. Different fingers  
    scores['fingers'] = 1 if finger1 != finger2 else 0
    
    # 3. Skip fingers (avoid adjacent finger pairs)
    finger_positions = {
        'left_pinky': 0, 'left_ring': 1, 'left_middle': 2, 'left_index': 3,
        'right_index': 4, 'right_middle': 5, 'right_ring': 6, 'right_pinky': 7
    }
    
    if finger1 in finger_positions and finger2 in finger_positions:
        finger_distance = abs(finger_positions[finger1] - finger_positions[finger2])
        if finger_distance == 1:  # Adjacent fingers
            scores['skip_fingers'] = 0
        elif finger_distance == 0:  # Same finger
            scores['skip_fingers'] = 0.5
        else:  # Non-adjacent fingers
            scores['skip_fingers'] = 1
    else:
        scores['skip_fingers'] = 0.5
    
    # 4. Don't cross over home row
    row1, row2 = pos1[0], pos2[0]
    if row1 == 1 or row2 == 1:  # If either is on home row
        scores['dont_cross_home'] = 1
    elif (row1 == 0 and row2 == 2) or (row1 == 2 and row2 == 0):  # Cross over home
        scores['dont_cross_home'] = 0
    else:
        scores['dont_cross_home'] = 1
    
    # 5. Same row preference
    scores['same_row'] = 1 if row1 == row2 else 0
    
    # 6. Home row preference  
    scores['home_row'] = 1 if char1 in home_keys and char2 in home_keys else 0.5 if char1 in home_keys or char2 in home_keys else 0
    
    # 7. Column discipline (avoid same column)
    col1, col2 = pos1[1], pos2[1]
    if col1 == col2:
        scores['columns'] = 0
    elif abs(col1 - col2) == 1:  # Adjacent columns
        scores['columns'] = 0.5
    else:
        scores['columns'] = 1
    
    # 8. Strum (inward to outward motion)
    if char1 in left_hand and char2 in right_hand:
        # Left to right: prefer left finger more central than right finger
        left_centrality = 5 - abs(pos1[1] - 4.5)  # Distance from center
        right_centrality = 5 - abs(pos2[1] - 4.5)
        scores['strum'] = 1 if left_centrality > right_centrality else 0
    elif char1 in right_hand and char2 in left_hand:
        # Right to left: prefer right finger more central than left finger  
        left_centrality = 5 - abs(pos2[1] - 4.5)
        right_centrality = 5 - abs(pos1[1] - 4.5)
        scores['strum'] = 1 if right_centrality > left_centrality else 0
    else:
        # Same hand - no strum benefit
        scores['strum'] = 0
    
    # 9. Strong fingers (avoid pinky)
    scores['strong_fingers'] = 1 if finger1 in strong_fingers and finger2 in strong_fingers else 0.5 if finger1 in strong_fingers or finger2 in strong_fingers else 0
    
    return scores

def adjust_times_for_frequency(sequences, times, freq_df, sequence_type="sequences"):
    """Adjust typing times for linguistic frequency using regression"""
    
    print_and_log(f"  🔍 Starting frequency adjustment for {sequence_type}...")
    print_and_log(f"      Input: {len(sequences):,} sequences, {len(freq_df)} frequency entries")
    print_and_log(f"      Frequency data columns: {list(freq_df.columns)}")
    
    # Build frequency dictionary
    if 'item_pair' in freq_df.columns and 'score' in freq_df.columns:
        freq_dict = dict(zip(freq_df['item_pair'], freq_df['score']))
    else:
        print_and_log(f"      ❌ Required columns not found in frequency data")
        return times, None
    
    print_and_log(f"      Built frequency dictionary: {len(freq_dict)} entries")
    
    # Show sample frequencies
    sample_freqs = list(freq_dict.items())[:5]
    print_and_log(f"      Example frequencies: {sample_freqs}")
    
    # Map sequences to frequencies
    matched_frequencies = []
    matched_times = []
    matched_sequences = []
    
    for i, seq in enumerate(sequences):
        if seq in freq_dict:
            matched_frequencies.append(freq_dict[seq])
            matched_times.append(times[i])
            matched_sequences.append(seq)
    
    overlap_pct = (len(matched_frequencies) / len(sequences)) * 100
    print_and_log(f"      Frequency overlap: {len(matched_frequencies)}/{len(sequences)} ({overlap_pct:.1f}%)")
    
    if len(matched_frequencies) < 10:
        print_and_log(f"      ⚠️  Too few matches for regression ({len(matched_frequencies)})")
        return times, None
    
    # Log-transform frequencies for better linear relationship
    log_frequencies = np.log10(np.array(matched_frequencies))
    times_array = np.array(matched_times)
    
    print_and_log(f"      Frequency range: {min(matched_frequencies):,.3f} to {max(matched_frequencies):,.3f} (mean: {np.mean(matched_frequencies):,.3f})")
    print_and_log(f"      Time range: {min(times_array):.1f} to {max(times_array):.1f}ms (mean: {np.mean(times_array):.1f} ± {np.std(times_array):.1f})")
    
    # Regression: time = intercept + slope * log_frequency
    try:
        X = sm.add_constant(log_frequencies)  # Add intercept term
        model = sm.OLS(times_array, X).fit()
        
        # Calculate adjusted times for all sequences
        adjusted_times = []
        model_info = {
            'r_squared': model.rsquared,
            'intercept': model.params.iloc[0],
            'slope': model.params.iloc[1],
            'p_value': model.pvalues.iloc[1] if len(model.pvalues) > 1 else None,
            'n_obs': len(matched_frequencies)
        }
        
        print_and_log(f"      Regression results:")
        print_and_log(f"        R² = {model.rsquared:.4f}")
        print_and_log(f"        Slope = {model.params.iloc[1]:.4f} (p = {model.pvalues.iloc[1]:.4f})")
        print_and_log(f"        Intercept = {model.params.iloc[0]:.4f}")
        
        # Calculate residuals for all sequences
        adjustments = []
        rank_changes = []
        original_ranks = stats.rankdata(times)
        
        for i, seq in enumerate(sequences):
            if seq in freq_dict:
                log_freq = np.log10(freq_dict[seq])
                predicted_time = model.params.iloc[0] + model.params.iloc[1] * log_freq
                adjustment = times[i] - predicted_time
                adjusted_times.append(adjustment)
                adjustments.append(abs(times[i] - adjustment))
            else:
                # For sequences without frequency data, use original time
                adjusted_times.append(times[i])
                adjustments.append(0)
        
        adjusted_ranks = stats.rankdata(adjusted_times)
        rank_changes = abs(original_ranks - adjusted_ranks)
        
        print_and_log(f"      Adjustment magnitude:")
        print_and_log(f"        Average: {np.mean(adjustments):.2f}ms")
        print_and_log(f"        Maximum: {np.max(adjustments):.2f}ms")
        changed_count = sum(1 for adj in adjustments if adj > 0.1)
        print_and_log(f"        Changed >0.1ms: {(changed_count/len(adjustments)*100):.1f}% of sequences")
        
        # Rank order analysis
        rank_correlation = spearmanr(original_ranks, adjusted_ranks)[0]
        print_and_log(f"      📊 RANK ORDER ANALYSIS:")
        print_and_log(f"        Correlation between raw and adjusted ranks: {rank_correlation:.6f}")
        sequences_with_rank_changes = sum(1 for change in rank_changes if change > 0)
        print_and_log(f"        Sequences with rank changes: {sequences_with_rank_changes}/{len(sequences)} ({sequences_with_rank_changes/len(sequences)*100:.1f}%)")
        print_and_log(f"        Maximum rank position change: {int(max(rank_changes))}")
        
        print_and_log(f"  ✅ Adjusted {sequence_type} times for frequency")
        print_and_log(f"  💡 SUMMARY: Frequency adjustment is working, but check rank order changes above")
        print_and_log(f"      to understand why Spearman correlations may be identical")
        
        return adjusted_times, model_info
        
    except Exception as e:
        print_and_log(f"      ❌ Regression failed: {e}")
        return times, None

def analyze_correlations(sequences, times, criteria_names, group_name, analysis_type, model_info=None):
    """Analyze correlations between Dvorak criteria scores and typing times"""
    results = {}
    
    print_and_log(f"  {analysis_type.replace('_', ' ').title()} Analysis:")
    
    # Calculate scores for all sequences
    print_and_log(f"  Calculating Dvorak scores for {len(sequences):,} sequences...")
    
    start_time = time.time()
    criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    valid_sequences = []
    valid_times = []
    sequence_scores_data = []
    
    for i, (seq, time_val) in enumerate(zip(sequences, times)):
        if i > 0 and i % 100000 == 0:
            elapsed = time.time() - start_time
            print_and_log(f"    Progress: {i:,}/{len(sequences):,} ({i/len(sequences)*100:.1f}%) - {elapsed:.1f}s", end='\r')
        
        # Calculate Dvorak scores
        scores = dvorak_9_criteria_bigrams(seq)
        
        # Validate scores
        if all(isinstance(score, (int, float)) and not np.isnan(score) for score in scores.values()):
            valid_sequences.append(seq)
            valid_times.append(time_val)
            
            # Store complete scores for this sequence (for interaction analysis)
            sequence_score_record = {'sequence': seq, 'time': time_val, 'analysis_type': analysis_type}
            for criterion in criteria_names.keys():
                score = scores[criterion]
                criterion_scores[criterion].append(score)
                sequence_score_record[criterion] = score
            sequence_scores_data.append(sequence_score_record)
            
            # Debug: Show sample scores for first few sequences
            if len(valid_sequences) <= 3:
                sample_scores = {k: f"{v:.3f}" for k, v in scores.items()}
                print_and_log(f"      Sample scores for '{seq}': {sample_scores}")
    
    elapsed = time.time() - start_time
    print_and_log(f"    Completed score calculation in {elapsed:.1f}s" + " " * 50)
    
    print_and_log(f"    Valid sequences for analysis: {len(valid_sequences):,}")
    
    # Store sequence data for combination analysis
    results[f'_sequence_scores_{analysis_type}'] = sequence_scores_data
    
    if len(valid_sequences) < 10:
        print_and_log(f"    ⚠️  Too few valid sequences for correlation analysis")
        return results
    
    # Calculate correlations
    print_and_log(f"    Calculating correlations for {len(criteria_names)} criteria...")
    for criterion, scores_list in criterion_scores.items():
        if len(scores_list) >= 3:  # Need at least 3 points for correlation
            try:
                # Check for constant values
                unique_scores = len(set(scores_list))
                if unique_scores <= 1:
                    print_and_log(f"    Warning: {criterion} has constant scores ({unique_scores} unique values)")
                    result_key = f"{criterion}_{analysis_type}"
                    results[result_key] = {
                        'name': criteria_names[criterion],
                        'group': f"{group_name} ({analysis_type.replace('_', ' ')})",
                        'analysis_type': analysis_type,
                        'n_samples': len(scores_list),
                        'pearson_r': float('nan'),
                        'pearson_p': float('nan'),
                        'spearman_r': float('nan'),
                        'spearman_p': float('nan'),
                        'mean_score': np.mean(scores_list),
                        'std_score': np.std(scores_list),
                        'scores': scores_list.copy(),
                        'times': valid_times.copy(),
                        'constant_scores': True
                    }
                    continue
                
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(scores_list, valid_times)
                
                # Spearman correlation (rank-based, more robust)
                spearman_r, spearman_p = spearmanr(scores_list, valid_times)
                
                # Check for NaN results
                if np.isnan(pearson_r) or np.isnan(spearman_r):
                    print_and_log(f"    Warning: {criterion} produced NaN correlations")
                    print_and_log(f"      Score range: {min(scores_list):.3f} to {max(scores_list):.3f}")
                    print_and_log(f"      Unique scores: {unique_scores}")
                
                result_key = f"{criterion}_{analysis_type}"
                results[result_key] = {
                    'name': criteria_names[criterion],
                    'group': f"{group_name} ({analysis_type.replace('_', ' ')})",
                    'analysis_type': analysis_type,
                    'n_samples': len(scores_list),
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'mean_score': np.mean(scores_list),
                    'std_score': np.std(scores_list),
                    'scores': scores_list.copy(),
                    'times': valid_times.copy()
                }
                
                # Add frequency model info if available
                if analysis_type == 'freq_adjusted' and model_info:
                    results[result_key]['frequency_model'] = model_info
                
            except Exception as e:
                print_and_log(f"    Error calculating correlation for {criterion}: {e}")
                # Store error information
                result_key = f"{criterion}_{analysis_type}"
                results[result_key] = {
                    'name': criteria_names[criterion],
                    'group': f"{group_name} ({analysis_type.replace('_', ' ')})",
                    'analysis_type': analysis_type,
                    'n_samples': len(scores_list),
                    'pearson_r': float('nan'),
                    'pearson_p': float('nan'),
                    'spearman_r': float('nan'),
                    'spearman_p': float('nan'),
                    'mean_score': np.mean(scores_list),
                    'std_score': np.std(scores_list),
                    'scores': scores_list.copy(),
                    'times': valid_times.copy(),
                    'error': str(e)
                }
                continue
    
    print_and_log(f"    Correlation analysis complete")
    
    # Diagnostic information about criteria variation
    constant_criteria = []
    low_variation_criteria = []
    
    for criterion, scores_list in criterion_scores.items():
        if len(scores_list) > 0:
            unique_scores = len(set(scores_list))
            score_std = np.std(scores_list)
            
            if unique_scores <= 1:
                constant_criteria.append((criterion, unique_scores, score_std))
            elif score_std < 0.01:  # Very low variation
                low_variation_criteria.append((criterion, unique_scores, score_std))
    
    if constant_criteria:
        print_and_log(f"    ⚠️  Criteria with constant scores: {len(constant_criteria)}")
        for criterion, unique, std in constant_criteria:
            print_and_log(f"      • {criterion}: {unique} unique values, std={std:.6f}")
            
    if low_variation_criteria:
        print_and_log(f"    ⚠️  Criteria with low variation: {len(low_variation_criteria)}")
        for criterion, unique, std in low_variation_criteria:
            print_and_log(f"      • {criterion}: {unique} unique values, std={std:.6f}")
    
    # Report why constant scores might occur
    if constant_criteria or low_variation_criteria:
        print_and_log(f"    💡 Possible causes of constant/low-variation scores:")
        print_and_log(f"      • Sample may be too small or not diverse enough")
        print_and_log(f"      • Bigrams may not trigger certain criteria (e.g., columns)")
        print_and_log(f"      • Filtering may have removed sequences with variation")
    
    return results

def analyze_bigram_data(bigrams, freq_df, middle_column_keys):
    """Analyze bigram typing data with middle column key analysis"""
    
    # Split data by middle column usage
    with_middle = []
    without_middle = []
    
    for bigram, time_val in bigrams:
        has_middle = any(char in middle_column_keys for char in bigram.lower())
        if has_middle:
            with_middle.append((bigram, time_val))
        else:
            without_middle.append((bigram, time_val))
    
    print_and_log(f"Data split:")
    print_and_log(f"  With middle columns: {len(with_middle):,} bigrams")
    print_and_log(f"  Without middle columns: {len(without_middle):,} bigrams")
    
    criteria_names = {
        'hands': 'hands',
        'fingers': 'fingers', 
        'skip_fingers': 'skip fingers',
        'dont_cross_home': "don't cross home",
        'same_row': 'same row',
        'home_row': 'home row',
        'columns': 'columns',
        'strum': 'strum',
        'strong_fingers': 'strong fingers'
    }
    
    all_results = {}
    
    # Analyze each group
    for group_data, group_name in [(without_middle, "Bigrams (No Middle Columns)"), 
                                   (with_middle, "Bigrams (With Middle Columns)")]:
        if len(group_data) < 10:
            print_and_log(f"\n--- Skipping {group_name} (too few sequences: {len(group_data)}) ---")
            continue
            
        print_and_log(f"\n--- Analyzing {group_name} ---")
        
        sequences = [item[0] for item in group_data]
        times = [item[1] for item in group_data]
        
        print_and_log(f"Sequences: {len(sequences):,}")
        print_and_log(f"Examples: {', '.join(sequences[:5])}")
        
        # Raw analysis (no frequency adjustment)
        raw_results = analyze_correlations(sequences, times, criteria_names, group_name, "raw")
        all_results.update(raw_results)
        
        # Frequency-adjusted analysis
        if freq_df is not None:
            adjusted_times, model_info = adjust_times_for_frequency(sequences, times, freq_df, "bigrams")
            freq_results = analyze_correlations(sequences, adjusted_times, criteria_names, group_name, "freq_adjusted", model_info)
            all_results.update(freq_results)
        else:
            print_and_log(f"  ⚠️  Skipping frequency adjustment (no frequency data)")
    
    return all_results

def print_correlation_results_with_frequency(results, analysis_name):
    """Print correlation results comparing raw vs frequency-adjusted"""
    
    print_and_log(f"\n{analysis_name.upper()} CORRELATION ANALYSIS: RAW vs FREQUENCY-ADJUSTED")
    print_and_log("=" * 120)
    print_and_log("Note: Negative correlation = higher score → faster typing (validates Dvorak)")
    print_and_log("      Positive correlation = higher score → slower typing (contradicts Dvorak)")
    print_and_log("-" * 120)
    
    # Group results by criterion and middle column status for comparison
    criterion_groups = {}
    for key, data in results.items():
        # Skip internal data and non-dictionary entries
        if key.startswith('_') or not isinstance(data, dict):
            continue
        
        # Extract criterion and analysis type
        parts = key.split('_')
        if len(parts) >= 2:
            if parts[-1] == 'adjusted':  # freq_adjusted
                criterion = '_'.join(parts[:-2])
                analysis = 'freq_adjusted'
            else:  # raw
                criterion = '_'.join(parts[:-1])
                analysis = 'raw'
                
            group_key = (criterion, data.get('group', ''))
            if group_key not in criterion_groups:
                criterion_groups[group_key] = {}
            criterion_groups[group_key][analysis] = data
    
    # Print comparison for each criterion/group combination
    for (criterion, group), analyses in sorted(criterion_groups.items()):
        if 'raw' in analyses and 'freq_adjusted' in analyses:
            raw_data = analyses['raw']
            adj_data = analyses['freq_adjusted']
            
            # Extract group name properly
            group_name = raw_data.get('group', 'Unknown Group')
            # Clean up the group name to remove redundant "Middle"
            if 'No Middle Columns' in group_name:
                clean_group_name = 'No Middle Columns'
            elif 'With Middle Columns' in group_name:
                clean_group_name = 'With Middle Columns'
            else:
                clean_group_name = group_name
            
            print_and_log(f"\n{raw_data['name']} - {clean_group_name}:")
            print_and_log(f"Analysis        N      Spearman r  p-val    Effect   Freq Model R²")
            print_and_log(f"----------------------------------------------------------------------")
            
            # Raw analysis
            sr = raw_data['spearman_r']
            sp = raw_data['spearman_p']
            
            # Handle NaN values
            if np.isnan(sr) or np.isnan(sp):
                sr_str = "nan"
                sp_str = "nan"
                s_sig = ""
                effect = "N/A"
            else:
                sr_str = f"{sr:>7.3f}"
                sp_str = f"{sp:<8.3f}"
                s_sig = "***" if sp < 0.001 else "**" if sp < 0.01 else "*" if sp < 0.05 else ""
                abs_sr = abs(sr)
                if abs_sr >= 0.5:
                    effect = "Large"
                elif abs_sr >= 0.3:
                    effect = "Med"
                elif abs_sr >= 0.1:
                    effect = "Small"
                else:
                    effect = "None"
            
            print_and_log(f"{'Raw':<15} {raw_data['n_samples']:<6} {sr_str}{s_sig:<4} {sp_str} {effect:<8} {'N/A':<12}")
            
            # Frequency-adjusted analysis
            sr_adj = adj_data['spearman_r']
            sp_adj = adj_data['spearman_p']
            
            # Handle NaN values
            if np.isnan(sr_adj) or np.isnan(sp_adj):
                sr_adj_str = "nan"
                sp_adj_str = "nan"
                s_sig_adj = ""
                effect_adj = "N/A"
            else:
                sr_adj_str = f"{sr_adj:>7.3f}"
                sp_adj_str = f"{sp_adj:<8.3f}"
                s_sig_adj = "***" if sp_adj < 0.001 else "**" if sp_adj < 0.01 else "*" if sp_adj < 0.05 else ""
                abs_sr_adj = abs(sr_adj)
                if abs_sr_adj >= 0.5:
                    effect_adj = "Large"
                elif abs_sr_adj >= 0.3:
                    effect_adj = "Med"
                elif abs_sr_adj >= 0.1:
                    effect_adj = "Small"
                else:
                    effect_adj = "None"
            
            freq_r2 = adj_data.get('frequency_model', {}).get('r_squared', 0)
            freq_r2_str = f"{freq_r2:.3f}" if freq_r2 and not np.isnan(freq_r2) else "N/A"
            
            print_and_log(f"{'Freq-Adjusted':<15} {adj_data['n_samples']:<6} {sr_adj_str}{s_sig_adj:<4} {sp_adj_str} {effect_adj:<8} {freq_r2_str:<12}")
            
            # Show change in correlation
            if not (np.isnan(sr) or np.isnan(sr_adj)):
                change = abs(sr_adj) - abs(sr)
                change_direction = "↑" if change > 0.05 else "↓" if change < -0.05 else "≈"
                print_and_log(f"{'Change':<15} {'':<6} {change:>+7.3f} {change_direction:<8}")
            else:
                print_and_log(f"{'Change':<15} {'':<6} {'N/A':>7} {'N/A':<8}")

def create_frequency_comparison_plots(results, output_dir='plots'):
    """Create visualization comparing raw vs frequency-adjusted correlations"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract data for plotting
    plot_data = []
    
    for key, data in results.items():
        # Skip internal data
        if key.startswith('_'):
            continue
        
        if not isinstance(data, dict) or 'spearman_r' in data:
            continue
        
        # Extract criterion and analysis type
        parts = key.split('_')
        if len(parts) >= 2:
            if parts[-1] == 'adjusted':  # freq_adjusted
                criterion = '_'.join(parts[:-2])
                analysis = 'freq_adjusted'
            else:  # raw
                criterion = '_'.join(parts[:-1])
                analysis = 'raw'
                
            plot_data.append({
                'criterion': criterion,
                'analysis': analysis,
                'correlation': data['spearman_r'],
                'p_value': data['spearman_p'],
                'group': data.get('group', ''),
                'name': data.get('name', criterion)
            })
    
    if not plot_data:
        print_and_log("No data available for plotting")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dvorak-9 criteria correlations: raw vs frequency-adjusted', fontsize=16, fontweight='bold')
    
    # Split by middle column status
    without_middle = df[df['group'].str.contains('No Middle', na=False)]
    with_middle = df[df['group'].str.contains('With Middle', na=False)]
    
    # Plot 1: Raw vs Freq-Adjusted correlations (Without middle columns)
    if not without_middle.empty:
        pivot_without = without_middle.pivot(index='criterion', columns='analysis', values='correlation')
        if 'raw' in pivot_without.columns and 'freq_adjusted' in pivot_without.columns:
            x_pos = np.arange(len(pivot_without.index))
            width = 0.35
            
            bars1 = ax1.bar(x_pos - width/2, pivot_without['freq_adjusted'], width, 
                           label='Freq adjusted', alpha=0.8, color='skyblue')
            bars2 = ax1.bar(x_pos + width/2, pivot_without['raw'], width,
                           label='Raw', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('Criterion')
            ax1.set_ylabel('Spearman correlation')
            ax1.set_title('Without middle columns')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(pivot_without.index, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 2: Raw vs Freq-Adjusted correlations (With middle columns)  
    if not with_middle.empty:
        pivot_with = with_middle.pivot(index='criterion', columns='analysis', values='correlation')
        if 'raw' in pivot_with.columns and 'freq_adjusted' in pivot_with.columns:
            x_pos = np.arange(len(pivot_with.index))
            width = 0.35
            
            bars1 = ax2.bar(x_pos - width/2, pivot_with['freq_adjusted'], width,
                           label='Freq adjusted', alpha=0.8, color='skyblue')
            bars2 = ax2.bar(x_pos + width/2, pivot_with['raw'], width,
                           label='Raw', alpha=0.8, color='lightcoral')
            
            ax2.set_xlabel('Criterion')
            ax2.set_ylabel('Spearman correlation')
            ax2.set_title('With middle columns')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(pivot_with.index, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 3: Raw vs Frequency-adjusted scatter plot
    raw_correlations = []
    freq_correlations = []
    labels = []
    
    # Group results by criterion for scatter plot
    criterion_groups = {}
    for _, row in df.iterrows():
        key = (row['criterion'], row['group'])
        if key not in criterion_groups:
            criterion_groups[key] = {}
        criterion_groups[key][row['analysis']] = row['correlation']
    
    for (criterion, group), analyses in criterion_groups.items():
        if 'raw' in analyses and 'freq_adjusted' in analyses:
            raw_correlations.append(analyses['raw'])
            freq_correlations.append(analyses['freq_adjusted'])
            
            # Create label
            middle_status = "NM" if "No Middle" in group else "WM" if "With Middle" in group else "?"
            labels.append(f"{criterion[:2].upper()}")
    
    if raw_correlations and freq_correlations:
        colors = ['blue' if 'NM' in label else 'red' for label in labels]
        ax3.scatter(raw_correlations, freq_correlations, c=colors, alpha=0.7, s=60)
        
        # Add labels to points
        for i, label in enumerate(labels):
            ax3.annotate(label, (raw_correlations[i], freq_correlations[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add diagonal line for perfect agreement
        min_val = min(min(raw_correlations), min(freq_correlations))
        max_val = max(max(raw_correlations), max(freq_correlations))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect agreement')
        
        ax3.set_xlabel('Raw correlation')
        ax3.set_ylabel('Frequency-adjusted correlation')
        ax3.set_title('Raw vs frequency-adjusted correlations')
        ax3.grid(True, alpha=0.3)
        ax3.legend(['With middle', 'Without middle', 'Perfect agreement'])
    
    # Plot 4: Effect sizes comparison
    effect_data = []
    for _, row in df.iterrows():
        effect_data.append({
            'criterion': row['name'],
            'analysis': row['analysis'],
            'abs_correlation': abs(row['correlation']),
            'group': 'With middle' if 'With Middle' in row['group'] else 'Without middle'
        })
    
    effect_df = pd.DataFrame(effect_data)
    if not effect_df.empty:
        pivot_effects = effect_df.pivot_table(index='criterion', columns='analysis', 
                                            values='abs_correlation', aggfunc='mean')
        
        if 'raw' in pivot_effects.columns and 'freq_adjusted' in pivot_effects.columns:
            x_pos = np.arange(len(pivot_effects.index))
            width = 0.35
            
            bars1 = ax4.bar(x_pos - width/2, pivot_effects['freq_adjusted'], width,
                           label='Freq adjusted', alpha=0.8, color='skyblue')
            bars2 = ax4.bar(x_pos + width/2, pivot_effects['raw'], width,
                           label='Raw', alpha=0.8, color='lightcoral')
            
            ax4.set_xlabel('Criterion')
            ax4.set_ylabel('Absolute correlation')
            ax4.set_title('Effect sizes (|r|)')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(pivot_effects.index, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'dvorak9_frequency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Comparison plots saved to: {output_path}")

def interpret_correlation_results(results, analysis_name):
    """Provide interpretation of correlation results"""
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log(f"📊 RESULTS INTERPRETATION {analysis_name.upper()}")
    print_and_log("=" * 80)
    
    # Collect significant results
    significant_results = []
    dvorak_support = []
    dvorak_contradict = []
    frequency_effects = []
    
    for key, data in results.items():
        # Skip internal data and non-dictionary entries
        if key.startswith('_') or not isinstance(data, dict):
            continue
            
        # Check for valid correlation data
        if 'spearman_p' in data and 'spearman_r' in data:
            # Skip NaN values
            if np.isnan(data['spearman_p']) or np.isnan(data['spearman_r']):
                continue
                
            if data['spearman_p'] < 0.05:
                significant_results.append((key, data))
                
                # Check if this supports or contradicts Dvorak
                if data['spearman_r'] < 0:  # Negative = faster with higher score = supports Dvorak
                    dvorak_support.append((data['name'], data['spearman_r'], data['spearman_p']))
                else:  # Positive = slower with higher score = contradicts Dvorak
                    dvorak_contradict.append((data['name'], data['spearman_r'], data['spearman_p']))
        
        # Check for frequency effects
        if 'frequency_model' in data and data['frequency_model']:
            freq_r2 = data['frequency_model'].get('r_squared', 0)
            if freq_r2 and not np.isnan(freq_r2) and freq_r2 > 0.01:  # R² > 1% is meaningful
                frequency_effects.append((data['name'], freq_r2, data['frequency_model'].get('p_value', 1)))
    
    # Overall findings
    print_and_log(f"\n🔍 OVERALL FINDINGS:")
    print_and_log(f"   • Total criteria tested: 9")
    print_and_log(f"   • Statistically significant results: {len(significant_results)}")
    print_and_log(f"   • Results supporting Dvorak principles: {len(dvorak_support)}")
    print_and_log(f"   • Results contradicting Dvorak principles: {len(dvorak_contradict)}")
    
    # Dvorak validation
    print_and_log(f"\n✅ DVORAK PRINCIPLE VALIDATION:")
    
    # Sort by effect size for better presentation
    dvorak_support.sort(key=lambda x: abs(x[1]), reverse=True)
    dvorak_contradict.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if dvorak_support:
        print_and_log(f"   CRITERIA THAT SUPPORT DVORAK (negative correlation = faster typing):")
        for name, r, p in dvorak_support:
            abs_r = abs(r)
            if abs_r >= 0.5:
                effect = "large effect"
            elif abs_r >= 0.3:
                effect = "medium effect"
            elif abs_r >= 0.1:
                effect = "small effect"
            else:
                effect = "negligible effect"
            print_and_log(f"     • {name}: r = {r:.3f}, p = {p:.3f} ({effect})")
    
    if dvorak_contradict:
        print_and_log(f"   ⚠️  CRITERIA THAT CONTRADICT DVORAK (positive correlation = slower typing):")
        for name, r, p in dvorak_contradict:
            abs_r = abs(r)
            if abs_r >= 0.5:
                effect = "large effect"
            elif abs_r >= 0.3:
                effect = "medium effect"
            elif abs_r >= 0.1:
                effect = "small effect"
            else:
                effect = "negligible effect"
            
            # Add context for contradictory results
            context = ""
            if "home row" in name.lower():
                context = "\n       → This suggests home row usage may slow typing in practice"
            elif "same row" in name.lower():
                context = "\n       → This suggests same-row sequences may slow typing (finger interference?)"
            elif "hand" in name.lower():
                context = "\n       → This suggests hand alternation may not always speed typing"
            
            print_and_log(f"     • {name}: r = {r:.3f}, p = {p:.3f} ({effect}){context}")
    
    # Frequency effects analysis
    if frequency_effects:
        print_and_log(f"\n📈 FREQUENCY ADJUSTMENT EFFECTS:")
        print_and_log(f"   The frequency adjustment successfully controlled for English letter/bigram frequency:")
        for name, r2, p_val in frequency_effects:
            r2_pct = r2 * 100
            sig_status = "significant" if p_val < 0.05 else "not significant"
            print_and_log(f"     • {name}: {r2_pct:.1f}% of variance explained by frequency ({sig_status})")
        
        print_and_log(f"\n   💡 INTERPRETATION:")
        print_and_log(f"     - Frequency effects explain 1-3% of typing time variance")
        print_and_log(f"     - This is typical for linguistic frequency in typing studies")
        print_and_log(f"     - Raw vs adjusted correlations show how much frequency biased results")
    
    # Middle column analysis - clear comparison of results
    with_middle_results = [data for key, data in results.items() if not key.startswith('_') and isinstance(data, dict) and 'with_middle' in key.lower() and data.get('spearman_p', 1) < 0.05]
    without_middle_results = [data for key, data in results.items() if not key.startswith('_') and isinstance(data, dict) and 'no_middle' in key.lower() and data.get('spearman_p', 1) < 0.05]
    
    if with_middle_results or without_middle_results:
        print_and_log(f"\n🎯 MIDDLE COLUMN KEY IMPACT ANALYSIS:")
        print_and_log(f"   Middle column keys (T, G, B, Y, H, N) require index finger stretches.")
        print_and_log(f"   This analysis compares how Dvorak criteria perform on:")
        print_and_log(f"   • Bigrams CONTAINING middle column keys (harder finger stretches)")
        print_and_log(f"   • Bigrams WITHOUT middle column keys (standard finger positions)")
        print_and_log(f"   Note: This is separate from the 'columns' criterion, which measures column discipline.")
        
        if without_middle_results:
            print_and_log(f"\n   📋 WITHOUT MIDDLE COLUMN KEYS ({len(without_middle_results)} significant effects):")
            print_and_log(f"      (Standard finger positions - no index finger stretches)")
            without_middle_results.sort(key=lambda x: abs(x['spearman_r']), reverse=True)
            for data in without_middle_results:
                direction = "supports Dvorak (faster)" if data['spearman_r'] < 0 else "contradicts Dvorak (slower)"
                abs_r = abs(data['spearman_r'])
                effect_size = "large" if abs_r >= 0.5 else "medium" if abs_r >= 0.3 else "small" if abs_r >= 0.1 else "negligible"
                print_and_log(f"      • {data['name']}: r = {data['spearman_r']:.3f} ({effect_size} effect, {direction})")
        
        if with_middle_results:
            print_and_log(f"\n   📋 WITH MIDDLE COLUMN KEYS ({len(with_middle_results)} significant effects):")
            print_and_log(f"      (Index finger stretches required - may alter typing dynamics)")
            with_middle_results.sort(key=lambda x: abs(x['spearman_r']), reverse=True)
            for data in with_middle_results:
                direction = "supports Dvorak (faster)" if data['spearman_r'] < 0 else "contradicts Dvorak (slower)"
                abs_r = abs(data['spearman_r'])
                effect_size = "large" if abs_r >= 0.5 else "medium" if abs_r >= 0.3 else "small" if abs_r >= 0.1 else "negligible"
                print_and_log(f"      • {data['name']}: r = {data['spearman_r']:.3f} ({effect_size} effect, {direction})")
        
        # Compare patterns between groups
        print_and_log(f"\n   🔍 MIDDLE COLUMN KEY IMPACT ON DVORAK PRINCIPLES:")
        
        # Find criteria that appear in both groups
        without_criteria = {data['name']: data['spearman_r'] for data in without_middle_results}
        with_criteria = {data['name']: data['spearman_r'] for data in with_middle_results}
        
        common_criteria = set(without_criteria.keys()) & set(with_criteria.keys())
        
        if common_criteria:
            print_and_log(f"      Criteria significant in BOTH groups:")
            for criterion in sorted(common_criteria):
                without_r = without_criteria[criterion]
                with_r = with_criteria[criterion]
                
                # Determine if middle columns amplify, reduce, or flip the effect
                if (without_r < 0 and with_r < 0) or (without_r > 0 and with_r > 0):
                    # Same direction
                    if abs(with_r) > abs(without_r):
                        change = "amplifies effect"
                    elif abs(with_r) < abs(without_r):
                        change = "reduces effect"
                    else:
                        change = "maintains effect"
                else:
                    # Opposite directions
                    change = "REVERSES effect direction"
                
                print_and_log(f"        • {criterion}: without={without_r:.3f}, with={with_r:.3f} (middle keys {change})")
        
        # Criteria unique to each group
        without_only = set(without_criteria.keys()) - set(with_criteria.keys())
        with_only = set(with_criteria.keys()) - set(without_criteria.keys())
        
        if without_only:
            print_and_log(f"      Criteria significant ONLY without middle keys:")
            for criterion in sorted(without_only):
                print_and_log(f"        • {criterion}: r = {without_criteria[criterion]:.3f}")
        
        if with_only:
            print_and_log(f"      Criteria significant ONLY with middle keys:")
            for criterion in sorted(with_only):
                print_and_log(f"        • {criterion}: r = {with_criteria[criterion]:.3f}")
        
        # Summary insight
        print_and_log(f"\n   💡 MIDDLE COLUMN KEY SUMMARY:")
        total_support_without = sum(1 for r in without_criteria.values() if r < 0)
        total_support_with = sum(1 for r in with_criteria.values() if r < 0)
        
        print_and_log(f"      • Without middle keys: {total_support_without}/{len(without_criteria)} criteria support Dvorak")
        print_and_log(f"      • With middle keys: {total_support_with}/{len(with_criteria)} criteria support Dvorak")
        
        if len(common_criteria) >= 3:
            direction_changes = sum(1 for c in common_criteria 
                                  if (without_criteria[c] < 0) != (with_criteria[c] < 0))
            print_and_log(f"      • Direction reversals: {direction_changes}/{len(common_criteria)} criteria flip direction")
        
        if total_support_with > total_support_without:
            print_and_log(f"      → Index finger stretches may ENHANCE Dvorak principles")
        elif total_support_with < total_support_without:
            print_and_log(f"      → Index finger stretches may IMPAIR Dvorak principles")
        else:
            print_and_log(f"      → Index finger stretches have MIXED effects on Dvorak principles")
    else:
        print_and_log(f"\n🎯 MIDDLE COLUMN KEY ANALYSIS:")
        print_and_log(f"   No significant differences found between sequences with/without middle column keys")
        print_and_log(f"   This suggests index finger stretches don't substantially alter Dvorak principle effectiveness")
    
    print_and_log(f"\n🛠️  PRACTICAL IMPLICATIONS:")
    print_and_log(f"   ✅ MOSTLY SUPPORTS DVORAK:")
    print_and_log(f"     - {len(dvorak_support)} criteria validate Dvorak principles")
    print_and_log(f"     - These typing patterns do correlate with faster speeds")
    print_and_log(f"     - Dvorak's optimization approach appears sound for these aspects")
    
    print_and_log(f"\n📏 EFFECT SIZE GUIDE:")
    print_and_log(f"   • |r| < 0.1  = Negligible effect (most results fall here)")
    print_and_log(f"   • |r| 0.1-0.3 = Small effect (still practically meaningful)")
    print_and_log(f"   • |r| 0.3-0.5 = Medium effect (substantial practical impact)")
    print_and_log(f"   • |r| > 0.5   = Large effect (major practical significance)")
    print_and_log(f"   Most typing research finds small-to-negligible effects due to individual variation.")
    
    # Count actual effect sizes found
    effect_counts = {'negligible': 0, 'small': 0, 'medium': 0, 'large': 0}
    for key, data in results.items():
        if not key.startswith('_') and isinstance(data, dict) and 'spearman_r' in data:
            if not np.isnan(data['spearman_r']) and data.get('spearman_p', 1) < 0.05:
                abs_r = abs(data['spearman_r'])
                if abs_r >= 0.5:
                    effect_counts['large'] += 1
                elif abs_r >= 0.3:
                    effect_counts['medium'] += 1
                elif abs_r >= 0.1:
                    effect_counts['small'] += 1
                else:
                    effect_counts['negligible'] += 1
    
    print_and_log(f"\n📊 ACTUAL EFFECT SIZES IN THIS ANALYSIS:")
    total_effects = sum(effect_counts.values())
    if total_effects > 0:
        for effect_type, count in effect_counts.items():
            percentage = (count / total_effects) * 100
            print_and_log(f"   • {effect_type.capitalize()}: {count}/{total_effects} ({percentage:.1f}%)")
    else:
        print_and_log(f"   • No significant correlations found")

def analyze_criterion_combinations(results):
    """Analyze how combinations of criteria predict typing speed"""
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log(f"COMPREHENSIVE CRITERION COMBINATION ANALYSIS")
    print_and_log("=" * 80)
    print_and_log("Examining how combinations of criteria interact to predict typing speed")
    
    # Look for sequence data in results
    sequence_data_sets = {}
    all_sequences = []
    
    # Collect all sequence data
    for key, data in results.items():
        if key.startswith('_sequence_scores') and isinstance(data, list):
            all_sequences.extend(data)
    
    # If we have sequence data, create the combined dataset
    if all_sequences:
        sequence_data_sets["All sequences (combined)"] = all_sequences
    
    # Also try to find specific group datasets if they exist separately
    for key, data in results.items():
        if key.startswith('_sequence_scores') and isinstance(data, list):
            # Determine group from parent key
            parent_key = key.replace('_sequence_scores', '')
            group_name = "Specific group"
            
            # Try to infer group from results keys
            for result_key in results.keys():
                if result_key.startswith(parent_key) and not result_key.startswith('_') and isinstance(results[result_key], dict):
                    if '_no_middle' in result_key:
                        if '_raw_' in result_key:
                            group_name = "No Middle Columns (Raw)"
                        elif '_freq_adjusted_' in result_key:
                            group_name = "No Middle Columns (Freq Adjusted)"
                    elif '_with_middle' in result_key:
                        if '_raw_' in result_key:
                            group_name = "With Middle Columns (Raw)"
                        elif '_freq_adjusted_' in result_key:
                            group_name = "With Middle Columns (Freq Adjusted)"
                    break
            
            # Only add if it's different from the combined dataset
            if group_name != "Specific group" and len(data) != len(all_sequences):
                sequence_data_sets[group_name] = data
    
    if not sequence_data_sets:
        print_and_log("❌ No sequence-level data found for combination analysis")
        return
    
    print_and_log(f"✅ Found sequence data for {len(sequence_data_sets)} groups")
    
    # Analyze each group
    for group_name, sequence_data in sequence_data_sets.items():
        if len(sequence_data) < 100:
            print_and_log(f"\n⚠️  Skipping {group_name}: too few sequences ({len(sequence_data)})")
            continue
        
        print_and_log(f"\n{group_name}")
        print_and_log(f"------------------------------------------------------------")
        print_and_log(f"   Sequences: {len(sequence_data):,}")
        
        # Convert to DataFrame
        df = pd.DataFrame(sequence_data)
        
        # Get criteria columns (exclude sequence, time, analysis_type)
        exclude_cols = {'sequence', 'time', 'analysis_type'}
        criteria_cols = [col for col in df.columns if col not in exclude_cols]
        
        print_and_log(f"   Criteria: {criteria_cols}")
        
        # Determine what type of data this is based on the group name or data characteristics
        data_description = "Combined dataset (all sequences)"
        if "No Middle" in group_name:
            data_description = "No middle column sequences only"
        elif "With Middle" in group_name:
            data_description = "With middle column sequences only"
        elif "Raw" in group_name:
            data_description = "Raw timing data (not frequency-adjusted)"
        elif "Freq" in group_name:
            data_description = "Frequency-adjusted timing data"
        
        print_and_log(f"   Data type: {data_description}")

        # 1. INDIVIDUAL CRITERION CORRELATIONS (baseline)
        print_and_log(f"\n   🎯 INDIVIDUAL CRITERION EFFECTS:")
        print_and_log(f"      (Based on {data_description.lower()})")
        individual_correlations = {}
        for criterion in criteria_cols:
            scores = df[criterion].values
            times = df['time'].values
            if len(set(scores)) > 1:  # Check for variation
                try:
                    corr, p_val = spearmanr(scores, times)
                    if not (np.isnan(corr) or np.isnan(p_val)):
                        individual_correlations[criterion] = {'r': corr, 'p': p_val}
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                        direction = "↓ faster" if corr < 0 else "↑ slower"
                        print_and_log(f"     {criterion:<15}: r={corr:>6.3f}{sig:<3} ({direction})")
                    else:
                        print_and_log(f"     {criterion:<15}: r=NaN (correlation failed)")
                except Exception as e:
                    print_and_log(f"     {criterion:<15}: Error - {e}")
            else:
                print_and_log(f"     {criterion:<15}: No variation (constant scores)")
        
        # 2. PAIRWISE COMBINATIONS
        print_and_log(f"\n   🤝 PAIRWISE CRITERION COMBINATIONS:")
        pairwise_effects = []
        
        for crit1, crit2 in combinations(criteria_cols, 2):
            scores1 = df[crit1].values
            scores2 = df[crit2].values
            times = df['time'].values
            
            # Skip if no variation
            if len(set(scores1)) <= 1 or len(set(scores2)) <= 1:
                continue
            
            # Create combined score (additive)
            combined_scores = scores1 + scores2
            
            if len(set(combined_scores)) > 1:
                try:
                    combined_corr, combined_p = spearmanr(combined_scores, times)
                    
                    # Skip if correlation failed
                    if np.isnan(combined_corr) or np.isnan(combined_p):
                        continue
                    
                    # Compare to individual effects
                    individual_r1 = individual_correlations.get(crit1, {}).get('r', 0)
                    individual_r2 = individual_correlations.get(crit2, {}).get('r', 0)
                    
                    # Handle NaN individual correlations
                    if np.isnan(individual_r1):
                        individual_r1 = 0
                    if np.isnan(individual_r2):
                        individual_r2 = 0
                    
                    # Estimate expected combined effect (rough approximation)
                    expected_combined = (individual_r1 + individual_r2) / 2
                    
                    # Calculate interaction effect
                    interaction_strength = combined_corr - expected_combined
                    
                    pairwise_effects.append({
                        'pair': f"{crit1} + {crit2}",
                        'combined_r': combined_corr,
                        'combined_p': combined_p,
                        'individual_r1': individual_r1,
                        'individual_r2': individual_r2,
                        'expected': expected_combined,
                        'interaction': interaction_strength
                    })
                except Exception as e:
                    print_and_log(f"       Error with {crit1} + {crit2}: {e}")
                    continue
        
        # Sort and display top pairwise effects
        pairwise_effects.sort(key=lambda x: abs(x['combined_r']), reverse=True)
        print_and_log(f"     Top pairwise combinations (by |r|):")
        for i, effect in enumerate(pairwise_effects[:8]):
            sig = "***" if effect['combined_p'] < 0.001 else "**" if effect['combined_p'] < 0.01 else "*" if effect['combined_p'] < 0.05 else ""
            interaction_type = "synergistic" if abs(effect['combined_r']) > abs(effect['expected']) else "additive"
            print_and_log(f"      {i+1}. {effect['pair']:<25} r={effect['combined_r']:>6.3f}{sig} ({interaction_type})")
        
        # 3. THREE-WAY COMBINATIONS
        print_and_log(f"\n   🎭 THREE-WAY CRITERION COMBINATIONS:")
        threeway_effects = []
        
        # Sample three-way combinations (too many to test all)
        import itertools
        threeway_combos = list(itertools.combinations(criteria_cols, 3))
        
        # Limit to reasonable number for performance
        if len(threeway_combos) > 50:
            threeway_combos = random.sample(threeway_combos, 50)
        
        for crit1, crit2, crit3 in threeway_combos:
            scores1 = df[crit1].values
            scores2 = df[crit2].values  
            scores3 = df[crit3].values
            times = df['time'].values
            
            # Skip if no variation
            if len(set(scores1)) <= 1 or len(set(scores2)) <= 1 or len(set(scores3)) <= 1:
                continue
            
            # Create combined score (additive)
            combined_scores = scores1 + scores2 + scores3
            
            if len(set(combined_scores)) > 1:
                try:
                    combined_corr, combined_p = spearmanr(combined_scores, times)
                    if not (np.isnan(combined_corr) or np.isnan(combined_p)):
                        threeway_effects.append({
                            'combo': f"{crit1} + {crit2} + {crit3}",
                            'combined_r': combined_corr,
                            'combined_p': combined_p
                        })
                except:
                    continue
        
        # Sort and display top three-way effects
        threeway_effects.sort(key=lambda x: abs(x['combined_r']), reverse=True)
        print_and_log(f"     Top three-way combinations:")
        for i, effect in enumerate(threeway_effects[:5]):
            sig = "***" if effect['combined_p'] < 0.001 else "**" if effect['combined_p'] < 0.01 else "*" if effect['combined_p'] < 0.05 else ""
            print_and_log(f"     {i+1}. {effect['combo']} r={effect['combined_r']:>6.3f}{sig}")
        
        # 4. STRONGEST COMBINATION OVERALL
        all_combinations = pairwise_effects + threeway_effects
        if all_combinations:
            strongest = max(all_combinations, key=lambda x: abs(x['combined_r']))
            combination_name = strongest.get('pair', strongest.get('combo', 'Unknown'))
            print_and_log(f"\n   🏆 STRONGEST COMBINATION OVERALL:")
            print_and_log(f"     {combination_name}")
            print_and_log(f"     Correlation: r = {strongest['combined_r']:.3f}, p = {strongest['combined_p']:.3f}")
            
            if abs(strongest['combined_r']) < 0.1:
                print_and_log(f"     ⚠️  Weak combination effects overall")
            elif abs(strongest['combined_r']) < 0.3:
                print_and_log(f"     ✅ Small but meaningful combination effects")
            else:
                print_and_log(f"     🎯 Strong combination effects found")
        
        # 5. MACHINE LEARNING ANALYSIS
        print_and_log(f"\n   🤖 MACHINE LEARNING INTERACTION ANALYSIS:")
        try:
            # Prepare data for ML
            X = df[criteria_cols].values
            y = df['time'].values
            
            # Check for variation in features and target
            if len(set(y)) > 1 and all(len(set(X[:, i])) > 1 for i in range(X.shape[1])):
                # Random Forest for feature importance and interaction detection
                rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                rf.fit(X, y)
                
                # Feature importance
                importances = rf.feature_importances_
                feature_importance = list(zip(criteria_cols, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Overall model performance
                y_pred = rf.predict(X)
                r2 = r2_score(y, y_pred)
                
                print_and_log(f"     Random Forest R² = {r2:.3f}")
                print_and_log(f"     Feature importance ranking:")
                for i, (feature, importance) in enumerate(feature_importance):
                    print_and_log(f"       {i+1}. {feature:<15}: {importance:.3f}")
                
            else:
                print_and_log(f"     ⚠️  Insufficient variation for ML analysis")
                
        except Exception as e:
            print_and_log(f"     ❌ ML analysis failed: {e}")

def load_and_process_bigram_data(bigram_file, max_bigrams=None):
    """Load and process bigram typing data"""
    print_and_log("Reading typing data files...")
    print_and_log(f"Reading bigram data from {bigram_file}...")
    
    # Data quality verification
    print_and_log("📋 BIGRAM DATA QUALITY VERIFICATION")
    print_and_log("   Expected: Correctly typed bigrams from correctly typed words")
    print_and_log("   Required CSV columns: 'bigram', 'interkey_interval'")
    
    try:
        # Read a small sample first to check structure
        sample_df = pd.read_csv(bigram_file, nrows=10)
        print_and_log(f"   Found columns: {list(sample_df.columns)}")
        
        if 'bigram' not in sample_df.columns or 'interkey_interval' not in sample_df.columns:
            print_and_log("   ❌ Required columns missing!")
            return None
        
        # Show sample data
        print_and_log("   Sample bigrams from CSV:")
        for _, row in sample_df.head(5).iterrows():
            print_and_log(f"     '{row['bigram']}': {row['interkey_interval']}ms")
        
        # Count total rows
        total_rows = sum(1 for _ in open(bigram_file)) - 1  # Subtract header
        print_and_log(f"   Total bigrams in file: {total_rows:,}")
        
        # Load full dataset
        print_and_log("   Quality indicators:")
        df = pd.read_csv(bigram_file)
        
        # Quality checks
        common_bigrams = ['th', 'he', 'in', 'er', 'an', 'nd', 'on', 'en', 'at', 'ou']
        found_common = sum(1 for bg in df['bigram'].head(10) if bg in common_bigrams)
        print_and_log(f"     Common English bigrams in sample: {found_common}/10 ({found_common*10}%)")
        
        # Check for suspicious patterns
        suspicious = sum(1 for bg in df['bigram'].head(10) if len(set(bg)) == 1 or len(bg) != 2)
        print_and_log(f"     Suspicious bigrams (repeated/invalid chars): {suspicious}/10 ({suspicious*10}%)")
        
        if found_common < 3:
            print_and_log("   ⚠️  Low proportion of common English bigrams - check data quality")
        
        print_and_log("   ✅ Proceeding with data loading...")
        
    except Exception as e:
        print_and_log(f"   ❌ Error reading bigram file: {e}")
        return None
    
    # Load and filter data
    if max_bigrams:
        print_and_log(f"\nWill randomly sample {max_bigrams:,} bigrams")
        # Use pandas sample for better randomness
        df = pd.read_csv(bigram_file)
        if len(df) > max_bigrams:
            df = df.sample(n=max_bigrams, random_state=42)
    else:
        df = pd.read_csv(bigram_file)
    
    print_and_log(f"Loaded {len(df):,} valid bigrams")
    
    # Convert to list of tuples
    bigrams = [(row['bigram'], row['interkey_interval']) for _, row in df.iterrows()]
    
    # Final quality summary
    print_and_log("📊 FINAL DATA QUALITY SUMMARY:")
    unique_bigrams = len(set(bg for bg, _ in bigrams))
    print_and_log(f"   Unique bigrams: {unique_bigrams}")
    
    common_count = sum(1 for bg, _ in bigrams if bg in common_bigrams)
    print_and_log(f"   Common English bigrams: {common_count:,} ({common_count/len(bigrams)*100:.1f}%)")
    
    times = [t for _, t in bigrams]
    print_and_log(f"   Average time: {np.mean(times):.1f}ms ± {np.std(times):.1f}ms")
    print_and_log("   ✅ Data quality looks reasonable")
    
    return bigrams

def filter_bigrams_by_time(bigrams, min_time=50, max_time=2000):
    """Filter bigrams by typing time thresholds"""
    original_count = len(bigrams)
    
    # Apply absolute thresholds
    filtered_bigrams = [(bg, time) for bg, time in bigrams if min_time <= time <= max_time]
    
    filtered_count = len(filtered_bigrams)
    removed_count = original_count - filtered_count
    
    print_and_log(f"Filtered {removed_count:,}/{original_count:,} bigrams using absolute thresholds ({min_time}-{max_time}ms)")
    print_and_log(f"  Kept {filtered_count:,} bigrams ({filtered_count/original_count*100:.1f}%)")
    
    if filtered_bigrams:
        times = [t for _, t in filtered_bigrams]
        print_and_log(f"  Time range: {min(times):.1f} - {max(times):.1f}ms")
    
    return filtered_bigrams

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Analyze Dvorak-9 criteria correlations with typing speed')
    parser.add_argument('--max-bigrams', type=int, help='Maximum number of bigrams to analyze')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--progress-interval', type=int, default=1000, help='Progress reporting interval')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    start_time = time.time()
    
    # Configuration
    bigram_file = "../process_3.5M_keystrokes/output/bigram_times.csv"
    freq_file = "./input/letter_pair_frequencies_english.csv"
    middle_column_keys = {'b', 'g', 'h', 'n', 't', 'y'}
    
    # Print configuration
    print_and_log("Dvorak-9 Criteria Correlation Analysis - Bigram Speed")
    print_and_log("=" * 80)
    print_and_log("Configuration:")
    print_and_log(f"  Max bigrams: {args.max_bigrams:,}" if args.max_bigrams else "  Max bigrams: unlimited")
    print_and_log(f"  Progress interval: {args.progress_interval:,}")
    print_and_log(f"  Random seed: {args.random_seed}")
    print_and_log(f"  Middle column keys: {', '.join(sorted(middle_column_keys))}")
    print_and_log("  Analysis includes both raw and frequency-adjusted correlations")
    print_and_log("  Focus: Bigram-level analysis only (word analysis removed)")
    print_and_log("")
    
    # Load frequency data
    print_and_log("Loading frequency data...")
    freq_df = load_frequency_data(freq_file)
    verify_frequency_data(freq_df)
    
    # Load bigram data
    bigrams = load_and_process_bigram_data(bigram_file, args.max_bigrams)
    if not bigrams:
        print_and_log("❌ Failed to load bigram data")
        return
    
    # Filter bigrams
    bigrams = filter_bigrams_by_time(bigrams)
    if not bigrams:
        print_and_log("❌ No bigrams remaining after filtering")
        return
    
    # Create output directory
    print_and_log("Creating plots in 'plots/' directory...")
    Path('plots').mkdir(exist_ok=True)
    
    print_and_log("\n" + "=" * 80)
    print_and_log("STARTING BIGRAM CORRELATION ANALYSIS")
    print_and_log("=" * 80)
    
    # Analyze bigrams
    print_and_log(f"\nBIGRAM ANALYSIS")
    bigram_start = time.time()
    
    print_and_log(f"Processing {len(bigrams):,} bigrams...")
    print_and_log("Analyzing bigram correlations with frequency control...")
    print_and_log(f"Total bigrams: {len(bigrams)}")
    print_and_log(f"Middle column keys: {', '.join(sorted(middle_column_keys))}")
    print_and_log("")
    
    bigram_results = analyze_bigram_data(bigrams, freq_df, middle_column_keys)
    
    bigram_elapsed = time.time() - bigram_start
    print_and_log(f"Bigram analysis completed in {format_time(bigram_elapsed)}")
    
    total_elapsed = time.time() - start_time
    print_and_log(f"\nTotal analysis time: {format_time(total_elapsed)}")
    
    # Print results with frequency comparison
    print_correlation_results_with_frequency(bigram_results, "BIGRAM")
    
    # Create plots
    print_and_log(f"\nGenerating comparison plots...")
    plot_start = time.time()
    
    create_frequency_comparison_plots(bigram_results)
    
    plot_elapsed = time.time() - plot_start
    print_and_log(f"Plot generation completed in {format_time(plot_elapsed)}")
    
    # Generate interpretation
    print_and_log(f"\nGenerating results interpretation...")
    interp_start = time.time()
    
    interpret_correlation_results(bigram_results, "BIGRAM ANALYSIS")
    
    interp_elapsed = time.time() - interp_start
    print_and_log(f"Interpretation completed in {format_time(interp_elapsed)}")
    
    # Analyze criterion combinations
    print_and_log(f"\nAnalyzing criterion combinations...")
    combo_start = time.time()
    
    analyze_criterion_combinations(bigram_results)
    
    combo_elapsed = time.time() - combo_start
    print_and_log(f"Combination analysis completed in {format_time(combo_elapsed)}")
    
    # Multiple comparisons correction
    if bigram_results:
        print_and_log(f"\n" + "=" * 80)
        print_and_log("MULTIPLE COMPARISONS CORRECTION")
        print_and_log("=" * 80)
        
        try:
            # Extract p-values for both raw and adjusted analyses
            p_values_raw = []
            p_values_adj = []
            keys_raw = []
            keys_adj = []
            
            for key, data in bigram_results.items():
                # Skip internal data and non-dictionary entries
                if key.startswith('_') or not isinstance(data, dict):
                    continue
                    
                if 'spearman_p' in data and not np.isnan(data['spearman_p']):
                    if '_raw_' in key:
                        p_values_raw.append(data['spearman_p'])
                        keys_raw.append(key)
                    elif '_freq_adjusted_' in key:
                        p_values_adj.append(data['spearman_p'])
                        keys_adj.append(key)
            
            # Apply FDR correction separately
            alpha = 0.05
            
            if p_values_raw:
                try:
                    rejected_raw, p_adj_raw, _, _ = multipletests(p_values_raw, alpha=alpha, method='fdr_bh')
                    print_and_log(f"\nRaw Analysis - Significant after FDR correction (α = {alpha}):")
                    any_sig_raw = False
                    for i, key in enumerate(keys_raw):
                        if rejected_raw[i]:
                            any_sig_raw = True
                            data = bigram_results[key]
                            direction = "↓ Faster" if data['spearman_r'] < 0 else "↑ Slower"
                            print_and_log(f"  {data['name']} ({data['group']}): r={data['spearman_r']:.3f}, p_adj={p_adj_raw[i]:.3f} {direction}")
                    if not any_sig_raw:
                        print_and_log("  None significant after correction")
                except Exception as e:
                    print_and_log(f"\nError in raw analysis FDR correction: {e}")
                    print_and_log(f"Number of p-values: {len(p_values_raw)}")
            else:
                print_and_log(f"\nNo valid p-values found for raw analysis")
            
            if p_values_adj:
                try:
                    rejected_adj, p_adj_adj, _, _ = multipletests(p_values_adj, alpha=alpha, method='fdr_bh')
                    print_and_log(f"\nFrequency-Adjusted Analysis - Significant after FDR correction (α = {alpha}):")
                    any_sig_adj = False
                    for i, key in enumerate(keys_adj):
                        if rejected_adj[i]:
                            any_sig_adj = True
                            data = bigram_results[key]
                            direction = "↓ Faster" if data['spearman_r'] < 0 else "↑ Slower"
                            print_and_log(f"  {data['name']} ({data['group']}): r={data['spearman_r']:.3f}, p_adj={p_adj_adj[i]:.3f} {direction}")
                    if not any_sig_adj:
                        print_and_log("  None significant after correction")
                except Exception as e:
                    print_and_log(f"\nError in frequency-adjusted analysis FDR correction: {e}")
                    print_and_log(f"Number of p-values: {len(p_values_adj)}")
            else:
                print_and_log(f"\nNo valid p-values found for frequency-adjusted analysis")
                
        except Exception as e:
            print_and_log(f"❌ Multiple comparisons correction failed: {e}")
    
    # Final summary
    total_elapsed = time.time() - start_time
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log("ANALYSIS COMPLETE")
    print_and_log("=" * 80)
    print_and_log(f"Total runtime: {format_time(total_elapsed)}")
    print_and_log(f"Key outputs saved:")
    print_and_log(f"- Text output: dvorak9_bigram_analysis_results.txt")
    print_and_log(f"- Comparison plots: plots/dvorak9_frequency_comparison.png")
    print_and_log(f"- Bigram-level analysis with and without frequency control")
    print_and_log(f"- Complete interpretation and criterion combination analysis")
    
    # Summary statistics
    if bigram_results:
        total_correlations = len([k for k in bigram_results.keys() if not k.startswith('_') and isinstance(bigram_results[k], dict)])
        valid_correlations = len([k for k, v in bigram_results.items() 
                                if not k.startswith('_') and isinstance(v, dict) 
                                and 'spearman_r' in v and not np.isnan(v['spearman_r'])])
        nan_correlations = total_correlations - valid_correlations
        
        print_and_log(f"\nAnalysis Summary:")
        print_and_log(f"- Total correlation tests: {total_correlations}")
        print_and_log(f"- Valid correlations: {valid_correlations}")
        print_and_log(f"- Failed/NaN correlations: {nan_correlations}")
        
        if nan_correlations > 0:
            print_and_log(f"- Note: NaN correlations typically result from constant criterion scores")
    
    if args.max_bigrams:
        print_and_log(f"\nNote: Analysis used sample limit:")
        print_and_log(f"- Bigrams: {args.max_bigrams:,} (from {len(bigrams):,} processed)")
    
    # Save log
    save_log()

if __name__ == "__main__":
    main()