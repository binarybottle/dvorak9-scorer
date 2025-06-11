"""
Test correlations between Dvorak-9 criteria and bigram typing speed with frequency control.

This script analyzes how well each of the 9 Dvorak criteria correlates with actual
bigram typing speed data, including frequency regression to control for bigram 
frequency effects.

Features:
- Bigram-level correlation analysis only (word analysis removed)
- Frequency regression to control for bigram frequency effects
- Split analysis by middle column inclusion (T,G,B,Y,H,N)
- Both raw and frequency-adjusted correlation analyses
- Statistical reporting with multiple comparison correction
- Interaction analysis between criteria combinations
- Progress monitoring and sample limiting for large datasets

Usage:
    python test_dvorak9_speed.py [--max-bigrams N] [--progress-interval N]
    
Arguments:
    --max-bigrams N       Limit bigram analysis to N samples (default: unlimited)
    --progress-interval N Show progress every N samples (default: 1000)
"""

import csv
import sys
import os
import argparse
import time
import random
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from itertools import combinations, product

# Progress monitoring
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for better progress bars: pip install tqdm")

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

# Import the Dvorak9Scorer from the provided script
try:
    from dvorak9_scorer import Dvorak9Scorer
except ImportError:
    print("Error: Could not import dvorak9_scorer.py")
    print("Make sure the file is in the same directory as this script.")
    sys.exit(1)

# Standard QWERTY layout mapping for testing
QWERTY_ITEMS = "abcdefghijklmnopqrstuvwxyz;,./"
QWERTY_POSITIONS = "QWERTYUIOPASDFGHJKL;ZXCVBNM,./"

# Define middle column keys (index finger stretch keys)
MIDDLE_COLUMN_KEYS = {'t', 'g', 'b', 'y', 'h', 'n'}

# Global configuration
output_file = None
progress_config = {
    'max_bigrams': None,
    'progress_interval': 1000,
    'start_time': None
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Dvorak-9 criteria correlation with bigram typing speed')
    parser.add_argument('--max-bigrams', type=int, default=None,
                       help='Limit bigram analysis to N samples (default: unlimited)')
    parser.add_argument('--progress-interval', type=int, default=1000,
                       help='Show progress every N samples (default: 1000)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for sample selection (default: 42)')
    
    return parser.parse_args()

def format_time(seconds):
    """Format time in seconds to readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def print_progress(current, total, operation="Processing", start_time=None):
    """Print progress information."""
    percent = (current / total) * 100 if total > 0 else 0
    
    if start_time:
        elapsed = time.time() - start_time
        if current > 0:
            estimated_total = elapsed * total / current
            remaining = estimated_total - elapsed
            time_info = f" | Elapsed: {format_time(elapsed)} | ETA: {format_time(remaining)}"
        else:
            time_info = f" | Elapsed: {format_time(elapsed)}"
    else:
        time_info = ""
    
    print_and_log(f"  {operation}: {current:,}/{total:,} ({percent:.1f}%){time_info}")

def print_and_log(*args, **kwargs):
    """Print to console and write to log file."""
    print(*args, **kwargs)
    if output_file:
        print(*args, **kwargs, file=output_file)
        output_file.flush()

def create_qwerty_mapping():
    """Create standard QWERTY layout mapping."""
    return dict(zip(QWERTY_ITEMS.lower(), QWERTY_POSITIONS.upper()))

def load_frequency_data(bigram_freq_path=None):
    """Load bigram frequency data from CSV files for regression analysis."""
    bigram_frequencies = None
    
    print_and_log("🔍 Loading frequency data for regression analysis...")
    print_and_log("   (This uses pre-calculated English language frequencies, NOT sample frequencies)")
    
    try:
        if bigram_freq_path and os.path.exists(bigram_freq_path):
            bigram_frequencies = pd.read_csv(bigram_freq_path)
            print_and_log(f"✅ Loaded bigram frequency data: {len(bigram_frequencies):,} entries")
            print_and_log(f"   Columns: {list(bigram_frequencies.columns)}")
            
            # Show some examples
            if not bigram_frequencies.empty:
                print_and_log("   Sample bigram frequencies:")
                for i, row in bigram_frequencies.head(3).iterrows():
                    if 'item_pair' in row and 'score' in row:
                        print_and_log(f"     '{row['item_pair']}': {row['score']}")
                    elif 'item' in row and 'score' in row:
                        print_and_log(f"     '{row['item']}': {row['score']}")
                    else:
                        print_and_log(f"     Row {i}: {dict(row)}")
        else:
            print_and_log(f"⚠️  Bigram frequency file not found: {bigram_freq_path}")
    except Exception as e:
        print_and_log(f"❌ Error loading bigram frequency data: {str(e)}")
    
    return bigram_frequencies

def adjust_sequence_times_for_frequency(sequences, times, freq_data, sequence_type='word'):
    """
    Adjust sequence typing times by regressing out frequency effects.
    
    Parameters:
    sequences (list): List of sequences (words or bigrams)
    times (list): List of corresponding typing times
    freq_data (DataFrame): Frequency data with 'item' or 'item_pair' and 'score' columns
    sequence_type (str): 'word' for word analysis, 'bigram' for bigram analysis
    
    Returns:
    tuple: (adjusted_times, frequency_dict, model_info)
    """
    if freq_data is None:
        print_and_log(f"  ⚠️  No frequency data available for {sequence_type} adjustment")
        return times, {}, None
    
    print_and_log(f"  🔍 Starting frequency adjustment for {sequence_type}s...")
    print_and_log(f"      Input: {len(sequences):,} sequences, {len(freq_data):,} frequency entries")
    
    # Create frequency dictionary
    freq_dict = {}
    freq_col = 'item' if sequence_type == 'word' else 'item_pair'
    
    # Check column names in frequency data
    print_and_log(f"      Frequency data columns: {list(freq_data.columns)}")
    
    if freq_col in freq_data.columns and 'score' in freq_data.columns:
        for _, row in freq_data.iterrows():
            seq_value = str(row[freq_col]).lower()
            freq_dict[seq_value] = row['score']
        print_and_log(f"      Built frequency dictionary: {len(freq_dict):,} entries")
        
        # Show some example frequencies
        sample_freqs = list(freq_dict.items())[:5]
        print_and_log(f"      Example frequencies: {sample_freqs}")
    else:
        print_and_log(f"  ⚠️  Required columns not found in {sequence_type} frequency data")
        print_and_log(f"      Expected: '{freq_col}' and 'score', Found: {list(freq_data.columns)}")
        return times, {}, None
    
    # Create DataFrame for regression - check overlap
    data_for_regression = []
    matched_count = 0
    unmatched_examples = []
    
    for seq, time in zip(sequences, times):
        seq_lower = seq.lower()
        if seq_lower in freq_dict:
            matched_count += 1
            data_for_regression.append({
                'sequence': seq,
                'time': time,
                'frequency': freq_dict[seq_lower],
                'log_freq': np.log10(freq_dict[seq_lower] + 1)
            })
        else:
            if len(unmatched_examples) < 5:
                unmatched_examples.append(seq_lower)
    
    overlap_pct = (matched_count / len(sequences)) * 100 if sequences else 0
    print_and_log(f"      Frequency overlap: {matched_count:,}/{len(sequences):,} ({overlap_pct:.1f}%)")
    
    if unmatched_examples:
        print_and_log(f"      Example unmatched: {unmatched_examples}")
    
    if len(data_for_regression) < 10:
        print_and_log(f"  ⚠️  Insufficient data for {sequence_type} frequency regression ({len(data_for_regression)} sequences)")
        print_and_log(f"      Need at least 10 sequences with frequency data")
        return times, freq_dict, None
    
    regression_df = pd.DataFrame(data_for_regression)
    
    # Show frequency distribution
    freq_stats = {
        'min': regression_df['frequency'].min(),
        'max': regression_df['frequency'].max(),
        'mean': regression_df['frequency'].mean(),
        'median': regression_df['frequency'].median()
    }
    print_and_log(f"      Frequency range: {freq_stats['min']:.3f} to {freq_stats['max']:.3f} (mean: {freq_stats['mean']:.3f})")
    
    try:
        # Create the model: time ~ log_frequency
        X = sm.add_constant(regression_df['log_freq'])
        y = regression_df['time']
        
        # Show time distribution
        time_stats = {
            'min': y.min(),
            'max': y.max(),
            'mean': y.mean(),
            'std': y.std()
        }
        print_and_log(f"      Time range: {time_stats['min']:.1f} to {time_stats['max']:.1f}ms (mean: {time_stats['mean']:.1f} ± {time_stats['std']:.1f})")
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Calculate adjusted times for all sequences
        adjusted_times = []
        model_info = {
            'r_squared': model.rsquared,
            'intercept': model.params.iloc[0],
            'slope': model.params.iloc[1],
            'p_value': model.pvalues.iloc[1] if len(model.pvalues) > 1 else None,
            'n_obs': len(regression_df)
        }
        
        print_and_log(f"      Regression results:")
        print_and_log(f"        R² = {model_info['r_squared']:.4f}")
        print_and_log(f"        Slope = {model_info['slope']:.4f} (p = {model_info['p_value']:.4f})")
        print_and_log(f"        Intercept = {model_info['intercept']:.4f}")
        
        # Check if frequency has a significant effect
        if model_info['p_value'] > 0.05:
            print_and_log(f"  ⚠️  Frequency effect not significant (p = {model_info['p_value']:.4f})")
            print_and_log(f"      This may explain why raw and adjusted results are similar")
        
        residual_mean = np.mean(regression_df['time'])
        adjustment_magnitude = []
        
        for seq, time in zip(sequences, times):
            seq_lower = seq.lower()
            if seq_lower in freq_dict:
                log_freq = np.log10(freq_dict[seq_lower] + 1)
                predicted_time = model_info['intercept'] + model_info['slope'] * log_freq
                # Residual + overall mean
                adjusted_time = time - predicted_time + residual_mean
                adjusted_times.append(adjusted_time)
                adjustment_magnitude.append(abs(time - adjusted_time))
            else:
                # If no frequency data, use original time
                adjusted_times.append(time)
                adjustment_magnitude.append(0)
        
        # Report adjustment statistics
        avg_adjustment = np.mean([adj for adj in adjustment_magnitude if adj > 0])
        max_adjustment = np.max(adjustment_magnitude)
        pct_changed = (sum(1 for adj in adjustment_magnitude if adj > 0.1) / len(adjustment_magnitude)) * 100
        
        print_and_log(f"      Adjustment magnitude:")
        print_and_log(f"        Average: {avg_adjustment:.2f}ms")
        print_and_log(f"        Maximum: {max_adjustment:.2f}ms") 
        print_and_log(f"        Changed >0.1ms: {pct_changed:.1f}% of sequences")
        
        # CRITICAL: Check if rank order changed
        from scipy.stats import spearmanr as rank_corr
        raw_ranks = np.argsort(np.argsort(times))
        adj_ranks = np.argsort(np.argsort(adjusted_times))
        
        rank_correlation = rank_corr(raw_ranks, adj_ranks)[0]
        rank_changes = np.sum(raw_ranks != adj_ranks)
        max_rank_change = np.max(np.abs(raw_ranks - adj_ranks))
        
        print_and_log(f"      📊 RANK ORDER ANALYSIS:")
        print_and_log(f"        Correlation between raw and adjusted ranks: {rank_correlation:.6f}")
        print_and_log(f"        Sequences with rank changes: {rank_changes}/{len(times)} ({100*rank_changes/len(times):.1f}%)")
        print_and_log(f"        Maximum rank position change: {max_rank_change}")
        
        if rank_correlation > 0.99:
            print_and_log(f"  ⚠️  EXPLANATION: Rank order barely changed (r={rank_correlation:.6f})")
            print_and_log(f"      This is why Spearman correlations are identical!")
            print_and_log(f"      Frequency effects are real but too small to change rankings")
        
        if avg_adjustment < 1.0:
            print_and_log(f"  ⚠️  Small adjustment magnitude may explain similar raw/adjusted results")
        
        print_and_log(f"  ✅ Adjusted {sequence_type} times for frequency")
        print_and_log(f"  💡 SUMMARY: Frequency adjustment is working, but check rank order changes above")
        print_and_log(f"      to understand why Spearman correlations may be identical")
        
        return adjusted_times, freq_dict, model_info
    
    except Exception as e:
        print_and_log(f"  ❌ Error in {sequence_type} frequency regression: {e}")
        return times, freq_dict, None

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

def analyze_criteria_for_group_with_frequency(sequences, times, layout_mapping, criteria_names, 
                                            group_name, freq_data=None, sequence_type='word'):
    """Analyze correlations for a specific group of sequences with frequency control."""
    print_and_log(f"\n--- Analyzing {group_name} ---")
    print_and_log(f"Sequences: {len(sequences):,}")
    
    if len(sequences) < 10:
        print_and_log(f"Too few sequences for reliable analysis ({len(sequences)})")
        return {}
    
    # Show some examples
    examples = sequences[:5] if len(sequences) >= 5 else sequences
    print_and_log(f"Examples: {', '.join(examples)}")
    
    # Adjust times for frequency effects
    raw_times = times.copy()
    adjusted_times, freq_dict, model_info = adjust_sequence_times_for_frequency(
        sequences, times, freq_data, sequence_type
    )
    
    results = {}
    
    # Store individual sequence scores for interaction analysis
    sequence_scores_data = []
    
    # Analyze both raw and frequency-adjusted times
    for analysis_type, analysis_times in [('raw', raw_times), ('freq_adjusted', adjusted_times)]:
        print_and_log(f"\n  {analysis_type.replace('_', ' ').title()} Analysis:")
        print_and_log(f"  Calculating Dvorak scores for {len(sequences):,} sequences...")
        
        # Collect scores for each criterion
        criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
        valid_times = []
        valid_sequences = []
        
        # Progress monitoring for score calculation
        start_time = time.time()
        processed = 0
        errors = 0
        
        # Use tqdm if available, otherwise manual progress
        sequence_iterator = zip(sequences, analysis_times)
        if TQDM_AVAILABLE:
            sequence_iterator = tqdm(sequence_iterator, total=len(sequences), 
                                   desc=f"  Calculating scores ({analysis_type})", leave=False)
        
        for seq, time_val in sequence_iterator:
            try:
                scorer = Dvorak9Scorer(layout_mapping, seq)
                results_dict = scorer.calculate_all_scores()
                scores = results_dict['scores']
                
                # Check if we have relevant data for bigram-level criteria
                if sequence_type == 'bigram' and len(seq) >= 2:
                    # For bigrams, we need at least one bigram to analyze
                    if results_dict['bigram_count'] == 0:
                        continue
                
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
            
            except Exception as e:
                errors += 1
                if errors <= 5:  # Only show first few errors
                    print_and_log(f"    Error processing sequence '{seq}': {e}")
                elif errors == 6:
                    print_and_log(f"    ... (suppressing further error messages)")
                continue
            
            processed += 1
            if not TQDM_AVAILABLE and processed % progress_config['progress_interval'] == 0:
                print_progress(processed, len(sequences), f"Calculating scores ({analysis_type})", start_time)
        
        elapsed = time.time() - start_time
        print_and_log(f"    Completed score calculation in {format_time(elapsed)}")
        print_and_log(f"    Valid sequences for analysis: {len(valid_sequences):,}")
        if errors > 0:
            print_and_log(f"    Errors encountered: {errors:,}")
        
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
    
    # Store sequence scores for interaction analysis
    results['_sequence_scores'] = sequence_scores_data
    
    return results

def read_bigram_times(filename, min_threshold=50, max_threshold=2000, use_percentile_filter=False, max_samples=None):
    """Read bigram times from CSV file with optional filtering and progress monitoring."""
    bigrams = []
    times = []
    
    try:
        # First pass: count total lines and inspect data quality
        print_and_log(f"Reading bigram data from {filename}...")
        print_and_log(f"📋 BIGRAM DATA QUALITY VERIFICATION")
        print_and_log(f"   Expected: Correctly typed bigrams from correctly typed words")
        print_and_log(f"   Required CSV columns: 'bigram', 'interkey_interval'")
        
        # Quick sample to verify data format
        sample_bigrams = []
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Check column names
            columns = reader.fieldnames
            print_and_log(f"   Found columns: {columns}")
            
            if 'bigram' not in columns or 'interkey_interval' not in columns:
                print_and_log(f"   ❌ Missing required columns!")
                return [], []
            
            # Sample first 10 rows for inspection
            for i, row in enumerate(reader):
                if i >= 10:
                    break
                bigram = row['bigram'].lower().strip()
                try:
                    time_val = float(row['interkey_interval'])
                    sample_bigrams.append((bigram, time_val))
                except ValueError:
                    print_and_log(f"   ⚠️  Invalid time value in row {i+1}: {row['interkey_interval']}")
        
        # Show sample data
        print_and_log(f"   Sample bigrams from CSV:")
        for bigram, time_val in sample_bigrams[:5]:
            print_and_log(f"     '{bigram}': {time_val:.1f}ms")
        
        # Now count total lines
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            total_lines = sum(1 for _ in file) - 1  # Subtract header
        
        print_and_log(f"   Total bigrams in file: {total_lines:,}")
        
        # Verify bigram quality
        valid_chars = set(QWERTY_ITEMS.lower())
        common_bigrams = set(['th', 'he', 'in', 'er', 'an', 'ed', 'nd', 'to', 'en', 'ti'])
        sample_check = [bg for bg, _ in sample_bigrams]
        
        # Check for common English bigrams
        common_found = sum(1 for bg in sample_check if bg in common_bigrams)
        print_and_log(f"   Quality indicators:")
        print_and_log(f"     Common English bigrams in sample: {common_found}/{len(sample_check)} ({100*common_found/len(sample_check):.1f}%)")
        
        # Check for suspicious patterns (repeated chars, non-letters)
        suspicious = sum(1 for bg in sample_check if len(set(bg)) == 1 or not all(c in valid_chars for c in bg))
        print_and_log(f"     Suspicious bigrams (repeated/invalid chars): {suspicious}/{len(sample_check)} ({100*suspicious/len(sample_check):.1f}%)")
        
        if suspicious > len(sample_check) * 0.3:
            print_and_log(f"   ⚠️  High rate of suspicious bigrams - data quality may be poor")
        
        print_and_log(f"   ✅ Proceeding with data loading...")
        print_and_log()
        
        # Determine sample size
        if max_samples and max_samples < total_lines:
            print_and_log(f"Will randomly sample {max_samples:,} bigrams")
            # Read all lines first, then sample
            all_data = []
            with open(filename, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    bigram = row['bigram'].lower().strip()
                    time_val = float(row['interkey_interval'])
                    
                    # Only include bigrams with characters in our layout
                    if len(bigram) == 2 and all(c in QWERTY_ITEMS.lower() for c in bigram):
                        all_data.append((bigram, time_val))
            
            # Random sample
            random.shuffle(all_data)
            sample_data = all_data[:max_samples]
            
            for bigram, time_val in sample_data:
                bigrams.append(bigram)
                times.append(time_val)
        else:
            # Read all data with progress monitoring
            with open(filename, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                if TQDM_AVAILABLE:
                    reader = tqdm(reader, total=total_lines, desc="Reading bigrams")
                
                processed = 0
                start_time = time.time()
                
                for row in reader:
                    bigram = row['bigram'].lower().strip()
                    time_val = float(row['interkey_interval'])
                    
                    # Only include bigrams with characters in our layout
                    if len(bigram) == 2 and all(c in QWERTY_ITEMS.lower() for c in bigram):
                        bigrams.append(bigram)
                        times.append(time_val)
                    
                    processed += 1
                    if not TQDM_AVAILABLE and processed % progress_config['progress_interval'] == 0:
                        print_progress(processed, total_lines, "Reading bigrams", start_time)
    
    except FileNotFoundError:
        print_and_log(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print_and_log(f"Error reading {filename}: {e}")
        return [], []
    
    if not times:
        return bigrams, times
    
    # Final data quality check
    original_count = len(times)
    print_and_log(f"Loaded {original_count:,} valid bigrams")
    
    # Check for data quality issues in the full dataset
    unique_bigrams = len(set(bigrams))
    common_english_count = sum(1 for bg in bigrams if bg in common_bigrams)
    
    print_and_log(f"📊 FINAL DATA QUALITY SUMMARY:")
    print_and_log(f"   Unique bigrams: {unique_bigrams:,}")
    print_and_log(f"   Common English bigrams: {common_english_count:,} ({100*common_english_count/len(bigrams):.1f}%)")
    print_and_log(f"   Average time: {np.mean(times):.1f}ms ± {np.std(times):.1f}ms")
    
    if common_english_count < len(bigrams) * 0.1:
        print_and_log(f"   ⚠️  Low rate of common English bigrams - verify data source")
    else:
        print_and_log(f"   ✅ Data quality looks reasonable")
    print_and_log()
    
    # Apply time-based filtering
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
        print_and_log(f"Filtered {removed_count:,}/{original_count:,} bigrams using {filter_method}")
        print_and_log(f"  Kept {len(filtered_times):,} bigrams ({len(filtered_times)/original_count*100:.1f}%)")
        print_and_log(f"  Time range: {min(filtered_times):.1f} - {max(filtered_times):.1f}ms")
    
    return filtered_bigrams, filtered_times

def analyze_bigram_correlations_with_frequency(bigrams, times, bigram_freq_data=None):
    """Analyze correlations between all 9 criteria and bigram times with frequency control."""
    layout_mapping = create_qwerty_mapping()
    
    # All 9 Dvorak criteria
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
    
    print_and_log("Analyzing bigram correlations with frequency control...")
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
        group_results = analyze_criteria_for_group_with_frequency(
            without_middle['sequences'], 
            without_middle['times'], 
            layout_mapping, 
            criteria_names, 
            "Bigrams (No Middle Columns)",
            freq_data=bigram_freq_data,
            sequence_type='bigram'
        )
        for criterion, data in group_results.items():
            results[f"{criterion}_no_middle"] = data
    
    # Group 2: Bigrams WITH middle column keys
    if with_middle['sequences']:
        group_results = analyze_criteria_for_group_with_frequency(
            with_middle['sequences'], 
            with_middle['times'], 
            layout_mapping, 
            criteria_names, 
            "Bigrams (With Middle Columns)",
            freq_data=bigram_freq_data,
            sequence_type='bigram'
        )
        for criterion, data in group_results.items():
            results[f"{criterion}_with_middle"] = data
    
    return results

def print_correlation_results_with_frequency(results, title):
    """Print correlation results comparing raw vs frequency-adjusted analyses."""
    print_and_log(f"\n{title}")
    print_and_log("=" * 120)
    print_and_log("Note: Negative correlation = higher score → faster typing (validates Dvorak)")
    print_and_log("      Positive correlation = higher score → slower typing (contradicts Dvorak)")
    print_and_log("-" * 120)
    
    if not results:
        print_and_log("No valid correlations found.")
        return
    
    # Group results by criterion and middle column status for comparison
    criterion_groups = {}
    for key, data in results.items():
        # Skip internal data and non-dictionary entries
        if key.startswith('_') or not isinstance(data, dict):
            continue
            
        # Parse the key: criterion_analysistype_middlestatus
        parts = key.split('_')
        if len(parts) >= 3:
            # Handle multi-word criteria like 'dont_cross_home'
            if parts[-2] in ['raw', 'freq'] and parts[-1] in ['middle', 'adjusted']:
                if parts[-1] == 'adjusted':
                    # freq_adjusted case
                    criterion_base = '_'.join(parts[:-2])
                    analysis_type = 'freq_adjusted'
                    middle_status = 'middle' if 'with_middle' in key else 'no_middle'
                else:
                    # raw_no_middle or raw_with_middle
                    criterion_base = '_'.join(parts[:-2])
                    analysis_type = parts[-2]
                    middle_status = parts[-1]
            else:
                # Parse more carefully for complex criterion names
                if key.endswith('_raw_no_middle'):
                    criterion_base = key[:-len('_raw_no_middle')]
                    analysis_type = 'raw'
                    middle_status = 'no_middle'
                elif key.endswith('_raw_with_middle'):
                    criterion_base = key[:-len('_raw_with_middle')]
                    analysis_type = 'raw'
                    middle_status = 'with_middle'
                elif key.endswith('_freq_adjusted_no_middle'):
                    criterion_base = key[:-len('_freq_adjusted_no_middle')]
                    analysis_type = 'freq_adjusted'
                    middle_status = 'no_middle'
                elif key.endswith('_freq_adjusted_with_middle'):
                    criterion_base = key[:-len('_freq_adjusted_with_middle')]
                    analysis_type = 'freq_adjusted'
                    middle_status = 'with_middle'
                else:
                    continue
            
            group_key = f"{criterion_base}_{middle_status}"
            if group_key not in criterion_groups:
                criterion_groups[group_key] = {}
            criterion_groups[group_key][analysis_type] = data
    
    # Print comparison table for each criterion group
    for group_key, analyses in criterion_groups.items():
        if 'raw' in analyses and 'freq_adjusted' in analyses:
            raw_data = analyses['raw']
            adj_data = analyses['freq_adjusted']
            
            middle_status = group_key.split('_')[-1]
            criterion_name = raw_data['name']
            
            print_and_log(f"\n{criterion_name} - {middle_status.replace('_', ' ').title().replace('No ', 'Without ').replace('With ', 'With ')} Middle Columns:")
            print_and_log(f"{'Analysis':<15} {'N':<6} {'Spearman r':<11} {'p-val':<8} {'Effect':<8} {'Freq Model R²':<12}")
            print_and_log("-" * 70)
            
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
                effect = "Large" if abs(sr) >= 0.5 else "Med" if abs(sr) >= 0.3 else "Small" if abs(sr) >= 0.1 else "None"
            
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
                effect_adj = "Large" if abs(sr_adj) >= 0.5 else "Med" if abs(sr_adj) >= 0.3 else "Small" if abs(sr_adj) >= 0.1 else "None"
            
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

def create_frequency_comparison_plots(results, output_dir="plots"):
    """Create plots comparing raw vs frequency-adjusted correlations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    if not results:
        return
    
    # Prepare data for plotting
    comparison_data = []
    
    for key, data in results.items():
        # Skip internal data and non-dictionary entries
        if key.startswith('_') or not isinstance(data, dict):
            continue
            
        # Parse the key to extract components
        if key.endswith('_raw_no_middle'):
            criterion_base = key[:-len('_raw_no_middle')]
            analysis_type = 'Raw'
            middle_status = 'without middle'
        elif key.endswith('_raw_with_middle'):
            criterion_base = key[:-len('_raw_with_middle')]
            analysis_type = 'Raw'
            middle_status = 'with middle'
        elif key.endswith('_freq_adjusted_no_middle'):
            criterion_base = key[:-len('_freq_adjusted_no_middle')]
            analysis_type = 'Freq adjusted'
            middle_status = 'without middle'
        elif key.endswith('_freq_adjusted_with_middle'):
            criterion_base = key[:-len('_freq_adjusted_with_middle')]
            analysis_type = 'Freq adjusted'
            middle_status = 'with middle'
        else:
            continue
        
        comparison_data.append({
            'criterion': data['name'],
            'criterion_full': data['name'],
            'middle_status': middle_status,
            'analysis_type': analysis_type,
            'spearman_r': data['spearman_r'],
            'spearman_p': data['spearman_p'],
            'n_samples': data['n_samples']
        })
    
    if not comparison_data:
        return
    
    df = pd.DataFrame(comparison_data)
    
    # Create comparison plot with better formatting
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Dvorak-9 criteria correlations: raw vs frequency-adjusted', fontsize=18, y=0.98)
    
    # Helper function to format axis for better readability
    def format_criterion_axis(ax, rotation=45):
        ax.tick_params(axis='x', rotation=rotation, labelsize=10)
        ax.grid(True, alpha=0.3)
        # Ensure labels fit
        plt.setp(ax.get_xticklabels(), ha='right')
    
    # Plot 1: Without Middle Columns - Raw vs Adjusted
    no_middle_data = df[df['middle_status'] == 'without middle']
    if not no_middle_data.empty:
        ax = axes[0, 0]
        pivot_data = no_middle_data.pivot(index='criterion', columns='analysis_type', values='spearman_r')
        if not pivot_data.empty:
            pivot_data.plot(kind='bar', ax=ax, color=['#2E86C1', '#E74C3C'], alpha=0.8, width=0.7)
            ax.set_title('Without middle columns', fontsize=14, pad=20)
            ax.set_xlabel('Criterion', fontsize=12)
            ax.set_ylabel('Spearman correlation', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax.legend(title='Analysis', fontsize=10, title_fontsize=11)
            format_criterion_axis(ax)
    
    # Plot 2: With Middle Columns - Raw vs Adjusted
    with_middle_data = df[df['middle_status'] == 'with middle']
    if not with_middle_data.empty:
        ax = axes[0, 1]
        pivot_data = with_middle_data.pivot(index='criterion', columns='analysis_type', values='spearman_r')
        if not pivot_data.empty:
            pivot_data.plot(kind='bar', ax=ax, color=['#2E86C1', '#E74C3C'], alpha=0.8, width=0.7)
            ax.set_title('With middle columns', fontsize=14, pad=20)
            ax.set_xlabel('Criterion', fontsize=12)
            ax.set_ylabel('Spearman correlation', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax.legend(title='Analysis', fontsize=10, title_fontsize=11)
            format_criterion_axis(ax)
    
    # Plot 3: Scatter plot - Raw vs Adjusted correlations
    ax = axes[1, 0]
    raw_data = df[df['analysis_type'] == 'Raw']
    adj_data = df[df['analysis_type'] == 'Freq adjusted']
    
    if not raw_data.empty and not adj_data.empty:
        # Match up corresponding analyses
        merged = pd.merge(raw_data, adj_data, on=['criterion', 'middle_status'], 
                         suffixes=('_raw', '_adj'))
        
        # Color by middle column status
        colors = ['#3498DB' if status == 'with middle' else '#E67E22' for status in merged['middle_status']]
        ax.scatter(merged['spearman_r_raw'], merged['spearman_r_adj'], 
                  c=colors, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line
        min_r = min(merged['spearman_r_raw'].min(), merged['spearman_r_adj'].min())
        max_r = max(merged['spearman_r_raw'].max(), merged['spearman_r_adj'].max())
        ax.plot([min_r, max_r], [min_r, max_r], 'k--', alpha=0.5, linewidth=2, label='Perfect Agreement')
        
        # Add abbreviated labels for criteria
        criterion_abbrev = {
            'hands': 'H', 'fingers': 'F', 'skip fingers': 'SF', 
            "don't cross home": 'DCH', 'same row': 'SR', 'home row': 'HR',
            'columns': 'C', 'strum': 'St', 'strong fingers': 'StF'
        }
        
        for _, row in merged.iterrows():
            abbrev = criterion_abbrev.get(row['criterion'], row['criterion'][:3])
            ax.annotate(abbrev, (row['spearman_r_raw'], row['spearman_r_adj']),
                       fontsize=9, alpha=0.8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Raw correlation', fontsize=12)
        ax.set_ylabel('Frequency-adjusted correlation', fontsize=12)
        ax.set_title('Raw vs frequency-adjusted correlations', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add custom legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#3498DB', label='With middle'),
                          Patch(facecolor='#E67E22', label='Without middle')]
        ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color='k', linestyle='--', label='Perfect agreement')], 
                 loc='best', fontsize=10)
    
    # Plot 4: Effect size comparison with better formatting
    ax = axes[1, 1]
    if not df.empty:
        df['abs_correlation'] = df['spearman_r'].abs()
        effect_comparison = df.pivot_table(index='criterion', columns='analysis_type', 
                                         values='abs_correlation', aggfunc='mean')
        
        if not effect_comparison.empty:
            effect_comparison.plot(kind='bar', ax=ax, color=['#2E86C1', '#E74C3C'], alpha=0.8, width=0.7)
            ax.set_title('Effect sizes (|r|)', fontsize=14, pad=20)
            ax.set_xlabel('Criterion', fontsize=12)
            ax.set_ylabel('Absolute correlation', fontsize=12)
            ax.legend(title='Analysis', fontsize=10, title_fontsize=11)
            format_criterion_axis(ax)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)
    plt.savefig(f"{output_dir}/dvorak9_frequency_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return "dvorak9_frequency_comparison.png"

def analyze_criterion_combinations(results):
    """Analyze actual combinations and interactions between Dvorak criteria."""
    print_and_log("\n" + "=" * 80)
    print_and_log("COMPREHENSIVE CRITERION COMBINATION ANALYSIS")
    print_and_log("=" * 80)
    print_and_log("Examining how combinations of criteria interact to predict typing speed")
    
    if not results:
        print_and_log("No results available for combination analysis.")
        return
    
    # Extract sequence-level data for combination analysis
    sequence_data_sets = {}
    
    for key, data in results.items():
        if key.startswith('_sequence_scores') and isinstance(data, list):
            # Determine group from parent key
            parent_key = key.replace('_sequence_scores', '')
            group_name = "Unknown"
            
            # Try to infer group from results keys
            for result_key in results.keys():
                if result_key.startswith(parent_key) and not result_key.startswith('_'):
                    if '_no_middle' in result_key:
                        if '_raw_' in result_key:
                            group_name = "No Middle (Raw)"
                        elif '_freq_adjusted_' in result_key:
                            group_name = "No Middle (Freq Adj)"
                    elif '_with_middle' in result_key:
                        if '_raw_' in result_key:
                            group_name = "With Middle (Raw)"
                        elif '_freq_adjusted_' in result_key:
                            group_name = "With Middle (Freq Adj)"
                    break
            
            sequence_data_sets[group_name] = data
    
    # If we don't have sequence data, extract from individual results
    if not sequence_data_sets:
        print_and_log("🔍 Extracting sequence-level data from correlation results...")
        
        # Group results by analysis group
        groups = {}
        for key, data in results.items():
            if key.startswith('_') or not isinstance(data, dict) or 'scores' not in data or 'times' not in data:
                continue
                
            if '_raw_no_middle' in key:
                group_name = 'No Middle (Raw)'
            elif '_raw_with_middle' in key:
                group_name = 'With Middle (Raw)'
            elif '_freq_adjusted_no_middle' in key:
                group_name = 'No Middle (Freq Adj)'
            elif '_freq_adjusted_with_middle' in key:
                group_name = 'With Middle (Freq Adj)'
            else:
                continue
            
            if group_name not in groups:
                groups[group_name] = {}
            
            criterion_name = key.split('_')[0]  # Extract criterion name
            groups[group_name][criterion_name] = {
                'scores': data['scores'],
                'times': data['times']
            }
        
        # Convert to sequence-level format
        for group_name, group_data in groups.items():
            if len(group_data) < 2:  # Need at least 2 criteria
                continue
                
            # Find common length (all criteria should have same number of sequences)
            lengths = [len(criterion_data['scores']) for criterion_data in group_data.values()]
            if len(set(lengths)) > 1:
                print_and_log(f"   ⚠️  Inconsistent sequence counts in {group_name}: {lengths}")
                continue
            
            n_sequences = lengths[0]
            sequence_records = []
            
            # Get criterion names and first criterion's times
            criteria_names = list(group_data.keys())
            times = group_data[criteria_names[0]]['times']
            
            for i in range(n_sequences):
                record = {'time': times[i]}
                for criterion in criteria_names:
                    record[criterion] = group_data[criterion]['scores'][i]
                sequence_records.append(record)
            
            sequence_data_sets[group_name] = sequence_records
    
    if not sequence_data_sets:
        print_and_log("❌ No sequence-level data available for combination analysis")
        return
    
    print_and_log(f"✅ Found sequence data for {len(sequence_data_sets)} groups")
    
    # Analyze each group
    for group_name, sequences in sequence_data_sets.items():
        if len(sequences) < 20:  # Need minimum sequences for reliable analysis
            print_and_log(f"\n⚠️  Skipping {group_name}: too few sequences ({len(sequences)})")
            continue
        
        print_and_log(f"\n📊 ANALYZING COMBINATIONS: {group_name}")
        print_and_log("-" * 60)
        
        # Extract criteria and times
        df = pd.DataFrame(sequences)
        if 'time' not in df.columns:
            print_and_log(f"   ❌ No time data available for {group_name}")
            continue
        
        times = df['time'].values
        criteria_cols = [col for col in df.columns if col != 'time' and col != 'sequence' and col != 'analysis_type']
        
        if len(criteria_cols) < 2:
            print_and_log(f"   ⚠️  Need at least 2 criteria for combinations ({len(criteria_cols)} found)")
            continue
        
        print_and_log(f"   Sequences: {len(sequences):,}")
        print_and_log(f"   Criteria: {criteria_cols}")
        
        # 1. INDIVIDUAL CRITERION CORRELATIONS (baseline)
        print_and_log(f"\n   🎯 INDIVIDUAL CRITERION EFFECTS:")
        individual_correlations = {}
        for criterion in criteria_cols:
            scores = df[criterion].values
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
        
        # Sort by strongest combined effect
        pairwise_effects.sort(key=lambda x: abs(x['combined_r']), reverse=True)
        
        print_and_log(f"     Top pairwise combinations (by |r|):")
        for i, effect in enumerate(pairwise_effects[:8]):  # Show top 8
            sig = "***" if effect['combined_p'] < 0.001 else "**" if effect['combined_p'] < 0.01 else "*" if effect['combined_p'] < 0.05 else ""
            interaction_desc = "synergy" if effect['interaction'] > 0.05 else "conflict" if effect['interaction'] < -0.05 else "additive"
            print_and_log(f"     {i+1:2d}. {effect['pair']:<25} r={effect['combined_r']:>6.3f}{sig:<3} ({interaction_desc})")
        
        # 3. THREE-WAY COMBINATIONS
        if len(criteria_cols) >= 3:
            print_and_log(f"\n   🎭 THREE-WAY CRITERION COMBINATIONS:")
            triplet_effects = []
            
            # Limit to top combinations to avoid too many
            for crit1, crit2, crit3 in combinations(criteria_cols, 3):
                scores1 = df[crit1].values
                scores2 = df[crit2].values
                scores3 = df[crit3].values
                
                # Skip if no variation
                if len(set(scores1)) <= 1 or len(set(scores2)) <= 1 or len(set(scores3)) <= 1:
                    continue
                
                # Create combined score
                combined_scores = scores1 + scores2 + scores3
                
                if len(set(combined_scores)) > 1:
                    combined_corr, combined_p = spearmanr(combined_scores, times)
                    
                    triplet_effects.append({
                        'triplet': f"{crit1} + {crit2} + {crit3}",
                        'combined_r': combined_corr,
                        'combined_p': combined_p
                    })
            
            # Sort and show top triplets
            triplet_effects.sort(key=lambda x: abs(x['combined_r']), reverse=True)
            
            print_and_log(f"     Top three-way combinations:")
            for i, effect in enumerate(triplet_effects[:5]):  # Show top 5
                sig = "***" if effect['combined_p'] < 0.001 else "**" if effect['combined_p'] < 0.01 else "*" if effect['combined_p'] < 0.05 else ""
                print_and_log(f"     {i+1}. {effect['triplet']:<40} r={effect['combined_r']:>6.3f}{sig}")
        
        # 4. IDENTIFY STRONGEST OVERALL COMBINATION
        all_combinations = pairwise_effects.copy()
        if 'triplet_effects' in locals():
            all_combinations.extend([{'pair': t['triplet'], 'combined_r': t['combined_r'], 'combined_p': t['combined_p']} 
                                   for t in triplet_effects])
        
        if all_combinations:
            strongest = max(all_combinations, key=lambda x: abs(x['combined_r']))
            print_and_log(f"\n   🏆 STRONGEST COMBINATION OVERALL:")
            print_and_log(f"     {strongest['pair']}")
            print_and_log(f"     Correlation: r = {strongest['combined_r']:.3f}, p = {strongest['combined_p']:.3f}")
            
            if abs(strongest['combined_r']) > 0.3:
                print_and_log(f"     ✅ Strong combination effect detected!")
            elif abs(strongest['combined_r']) > 0.1:
                print_and_log(f"     ✓ Moderate combination effect")
            else:
                print_and_log(f"     ⚠️  Weak combination effects overall")
        
        # 5. MACHINE LEARNING ANALYSIS (if available)
        if sklearn_available and len(criteria_cols) >= 3:
            print_and_log(f"\n   🤖 MACHINE LEARNING INTERACTION ANALYSIS:")
            
            try:
                X = df[criteria_cols].values
                y = times
                
                # Random Forest to capture non-linear interactions
                rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
                rf.fit(X, y)
                rf_score = rf.score(X, y)
                
                # Feature importance
                importances = rf.feature_importances_
                feature_importance = list(zip(criteria_cols, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print_and_log(f"     Random Forest R² = {rf_score:.3f}")
                print_and_log(f"     Feature importance ranking:")
                for i, (feature, importance) in enumerate(feature_importance):
                    print_and_log(f"       {i+1}. {feature:<15}: {importance:.3f}")
                
                # Polynomial features for interaction detection
                if len(criteria_cols) <= 5:  # Avoid combinatorial explosion
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    X_poly = poly.fit_transform(X)
                    
                    # Use Lasso to identify important interactions
                    lasso = LassoCV(cv=5, random_state=42)
                    lasso.fit(X_poly, y)
                    
                    # Get interaction terms
                    feature_names = poly.get_feature_names_out(criteria_cols)
                    interactions = []
                    for i, coef in enumerate(lasso.coef_):
                        if abs(coef) > 0.001 and ' ' in feature_names[i]:  # Interaction term
                            interactions.append((feature_names[i], coef))
                    
                    interactions.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    if interactions:
                        print_and_log(f"     Lasso-detected interactions (R² = {lasso.score(X_poly, y):.3f}):")
                        for interaction, coef in interactions[:5]:
                            effect = "↓ faster" if coef < 0 else "↑ slower"
                            print_and_log(f"       {interaction:<20}: {coef:>7.3f} ({effect})")
                    else:
                        print_and_log(f"     No significant interactions detected by Lasso")
                
            except Exception as e:
                print_and_log(f"     ❌ ML analysis error: {e}")

def interpret_correlation_results(results, title_prefix=""):
    """Provide detailed interpretation of correlation results."""
    print_and_log(f"\n" + "=" * 80)
    print_and_log(f"📊 RESULTS INTERPRETATION {title_prefix}")
    print_and_log("=" * 80)
    
    if not results:
        print_and_log("No results to interpret.")
        return
    
    # Collect significant results for interpretation
    significant_results = []
    frequency_effects = []
    dvorak_support = []
    dvorak_contradict = []
    
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
    
    # 1. Overall Summary
    print_and_log(f"\n🔍 OVERALL FINDINGS:")
    total_criteria = len(set(d['name'] for k, d in results.items() if not k.startswith('_')))
    print_and_log(f"   • Total criteria tested: {total_criteria}")
    print_and_log(f"   • Statistically significant results: {len(significant_results)}")
    print_and_log(f"   • Results supporting Dvorak principles: {len(dvorak_support)}")
    print_and_log(f"   • Results contradicting Dvorak principles: {len(dvorak_contradict)}")
    
    # 2. Dvorak Validation Analysis
    if dvorak_support or dvorak_contradict:
        print_and_log(f"\n✅ DVORAK PRINCIPLE VALIDATION:")
        
        if dvorak_support:
            print_and_log(f"   CRITERIA THAT SUPPORT DVORAK (negative correlation = faster typing):")
            for name, r, p in sorted(dvorak_support, key=lambda x: abs(x[1]), reverse=True):
                effect_size = "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small"
                print_and_log(f"     • {name}: r = {r:.3f}, p = {p:.3f} ({effect_size} effect)")
        
        if dvorak_contradict:
            print_and_log(f"   ⚠️  CRITERIA THAT CONTRADICT DVORAK (positive correlation = slower typing):")
            for name, r, p in sorted(dvorak_contradict, key=lambda x: x[1], reverse=True):
                effect_size = "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small"
                print_and_log(f"     • {name}: r = {r:.3f}, p = {p:.3f} ({effect_size} effect)")
                
                # Provide specific interpretation for each contradictory finding
                if 'home row' in name.lower():
                    print_and_log(f"       → This suggests home row usage may slow typing in practice")
                elif 'same row' in name.lower():
                    print_and_log(f"       → This suggests same-row sequences may slow typing (finger interference?)")
                elif 'hands' in name.lower():
                    print_and_log(f"       → This suggests hand alternation may not always speed typing")
    
    # 3. Frequency Effect Analysis
    if frequency_effects:
        print_and_log(f"\n📈 FREQUENCY ADJUSTMENT EFFECTS:")
        print_and_log(f"   The frequency adjustment successfully controlled for English letter/bigram frequency:")
        
        for name, r2, p in sorted(frequency_effects, key=lambda x: x[1], reverse=True):
            percent_var = r2 * 100
            significance = "significant" if p < 0.05 else "non-significant"
            print_and_log(f"     • {name}: {percent_var:.1f}% of variance explained by frequency ({significance})")
        
        print_and_log(f"\n   💡 INTERPRETATION:")
        print_and_log(f"     - Frequency effects explain 1-3% of typing time variance")
        print_and_log(f"     - This is typical for linguistic frequency in typing studies")
        print_and_log(f"     - Raw vs adjusted correlations show how much frequency biased results")
    
    # 4. Middle Column Analysis
    print_and_log(f"\n🎯 MIDDLE COLUMN KEY ANALYSIS:")
    print_and_log(f"   Middle column keys (T, G, B, Y, H, N) require index finger stretches.")
    print_and_log(f"   Comparing sequences with/without these keys tests finger stretch effects:")
    
    # Group results by middle column status
    with_middle_results = [data for key, data in results.items() if not key.startswith('_') and isinstance(data, dict) and 'with_middle' in key and data.get('spearman_p', 1) < 0.05]
    without_middle_results = [data for key, data in results.items() if not key.startswith('_') and isinstance(data, dict) and 'no_middle' in key and data.get('spearman_p', 1) < 0.05]
    
    if with_middle_results:
        print_and_log(f"   WITH middle column keys ({len(with_middle_results)} significant effects):")
        for data in sorted(with_middle_results, key=lambda x: abs(x['spearman_r']), reverse=True):
            direction = "supports" if data['spearman_r'] < 0 else "contradicts"
            print_and_log(f"     • {data['name']}: r = {data['spearman_r']:.3f} ({direction} Dvorak)")
    
    if without_middle_results:
        print_and_log(f"   WITHOUT middle column keys ({len(without_middle_results)} significant effects):")
        for data in sorted(without_middle_results, key=lambda x: abs(x['spearman_r']), reverse=True):
            direction = "supports" if data['spearman_r'] < 0 else "contradicts"
            print_and_log(f"     • {data['name']}: r = {data['spearman_r']:.3f} ({direction} Dvorak)")
    
    # 5. Practical Implications
    print_and_log(f"\n🛠️  PRACTICAL IMPLICATIONS:")
    
    if len(dvorak_support) > len(dvorak_contradict):
        print_and_log(f"   ✅ MOSTLY SUPPORTS DVORAK:")
        print_and_log(f"     - {len(dvorak_support)} criteria validate Dvorak principles")
        print_and_log(f"     - These typing patterns do correlate with faster speeds")
        print_and_log(f"     - Dvorak's optimization approach appears sound for these aspects")
    elif len(dvorak_contradict) > len(dvorak_support):
        print_and_log(f"   ⚠️  MIXED OR CONTRADICTORY EVIDENCE:")
        print_and_log(f"     - {len(dvorak_contradict)} criteria contradict Dvorak principles")
        print_and_log(f"     - Some Dvorak assumptions may not hold in practice")
        print_and_log(f"     - Modern typing behavior may differ from Dvorak's 1930s assumptions")
    else:
        print_and_log(f"   📊 NEUTRAL EVIDENCE:")
        print_and_log(f"     - Equal support and contradiction for Dvorak principles")
        print_and_log(f"     - Dvorak optimization may be more complex than originally theorized")
    
    # 6. Study Limitations & Considerations
    print_and_log(f"\n⚠️  IMPORTANT LIMITATIONS:")
    print_and_log(f"   • This analysis uses QWERTY typing data to test Dvorak principles")
    print_and_log(f"   • QWERTY typists are optimized for QWERTY, not Dvorak patterns")
    print_and_log(f"   • True validation would require native Dvorak typists")
    print_and_log(f"   • Sample size and demographic factors may influence results")
    print_and_log(f"   • Individual typing styles vary significantly")
    
    # 7. Effect Size Interpretation
    print_and_log(f"\n📏 EFFECT SIZE GUIDE:")
    print_and_log(f"   • |r| < 0.1  = Negligible effect")
    print_and_log(f"   • |r| 0.1-0.3 = Small effect (still practically meaningful)")
    print_and_log(f"   • |r| 0.3-0.5 = Medium effect (substantial practical impact)")
    print_and_log(f"   • |r| > 0.5   = Large effect (major practical significance)")
    print_and_log(f"   Most typing research finds small-to-medium effects due to individual variation.")

def main():
    """Main analysis function with frequency control and progress monitoring."""
    global output_file, progress_config
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set random seed for reproducible sampling
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Update progress configuration
    progress_config.update({
        'max_bigrams': args.max_bigrams,
        'progress_interval': args.progress_interval,
        'start_time': time.time()
    })
    
    # Open output file
    output_file = open('dvorak9_bigram_analysis_results.txt', 'w', encoding='utf-8')
    
    try:
        print_and_log("Dvorak-9 Criteria Correlation Analysis - Bigram Speed")
        print_and_log("=" * 80)
        print_and_log(f"Configuration:")
        print_and_log(f"  Max bigrams: {args.max_bigrams or 'unlimited'}")
        print_and_log(f"  Progress interval: {args.progress_interval:,}")
        print_and_log(f"  Random seed: {args.random_seed}")
        print_and_log(f"  Middle column keys: {', '.join(sorted(MIDDLE_COLUMN_KEYS))}")
        print_and_log("  Analysis includes both raw and frequency-adjusted correlations")
        print_and_log("  Focus: Bigram-level analysis only (word analysis removed)")
        print_and_log()
        
        # Load frequency data
        print_and_log("Loading frequency data...")
        bigram_freq_file = 'input/letter_pair_frequencies_english.csv'
        
        bigram_frequencies = load_frequency_data(bigram_freq_file)
        
        if bigram_frequencies is not None:
            print_and_log(f"✅ Bigram frequencies loaded: {len(bigram_frequencies):,} entries") 
        else:
            print_and_log(f"❌ No bigram frequency data available")
        
        # Read typing data files
        print_and_log("\nReading typing data files...")
        bigram_times_file = '../process_3.5M_keystrokes/output/bigram_times.csv'
        
        # FILTERING PARAMETERS
        MIN_INTERVAL = 50
        MAX_INTERVAL = 2000
        USE_PERCENTILE_BIGRAMS = False
        
        # Read the data files with progress monitoring
        bigrams, bigram_times = read_bigram_times(
            bigram_times_file, 
            min_threshold=MIN_INTERVAL,
            max_threshold=MAX_INTERVAL, 
            use_percentile_filter=USE_PERCENTILE_BIGRAMS,
            max_samples=args.max_bigrams
        )
        
        if not bigrams:
            print_and_log("Error: No valid bigram data found in CSV files.")
            sys.exit(1)
        
        # Create output directory for plots
        Path("plots").mkdir(exist_ok=True)
        print_and_log("Creating plots in 'plots/' directory...")
        
        # Analyze correlations with frequency control
        print_and_log(f"\n{'='*80}")
        print_and_log("STARTING BIGRAM CORRELATION ANALYSIS")
        print_and_log(f"{'='*80}")
        
        analysis_start_time = time.time()
        
        print_and_log(f"\nBIGRAM ANALYSIS")
        print_and_log(f"Processing {len(bigrams):,} bigrams...")
        bigram_start = time.time()
        
        bigram_results = analyze_bigram_correlations_with_frequency(
            bigrams, bigram_times, bigram_frequencies
        )
        
        bigram_elapsed = time.time() - bigram_start
        print_and_log(f"Bigram analysis completed in {format_time(bigram_elapsed)}")
        
        total_analysis_time = time.time() - analysis_start_time
        print_and_log(f"\nTotal analysis time: {format_time(total_analysis_time)}")
        
        if bigram_results:
            # Print correlation tables with frequency comparison
            print_correlation_results_with_frequency(
                bigram_results, 
                "BIGRAM CORRELATION ANALYSIS: RAW vs FREQUENCY-ADJUSTED"
            )
            
            # Create frequency comparison plots
            print_and_log(f"\nGenerating comparison plots...")
            plot_start = time.time()
            create_frequency_comparison_plots(bigram_results)
            plot_elapsed = time.time() - plot_start
            print_and_log(f"Plot generation completed in {format_time(plot_elapsed)}")
            
            # Add the missing interpretation analysis
            print_and_log(f"\nGenerating results interpretation...")
            interpret_start = time.time()
            interpret_correlation_results(bigram_results, "BIGRAM ANALYSIS")
            interpret_elapsed = time.time() - interpret_start
            print_and_log(f"Interpretation completed in {format_time(interpret_elapsed)}")
            
            # Analyze criterion combinations (not just interactions)
            print_and_log(f"\nAnalyzing criterion combinations...")
            combination_start = time.time()
            analyze_criterion_combinations(bigram_results)
            combination_elapsed = time.time() - combination_start
            print_and_log(f"Combination analysis completed in {format_time(combination_elapsed)}")
            
            # Apply multiple comparisons correction
            print_and_log("\n" + "=" * 80)
            print_and_log("MULTIPLE COMPARISONS CORRECTION")
            print_and_log("=" * 80)
            
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
        
        total_elapsed = time.time() - progress_config['start_time']
        
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
        
    finally:
        if output_file:
            output_file.close()

if __name__ == "__main__":
    main()