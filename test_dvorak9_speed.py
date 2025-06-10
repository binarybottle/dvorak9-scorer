#!/usr/bin/env python3
"""
Test correlations between Dvorak-9 criteria and typing speed with frequency control.

This script analyzes how well each of the 9 Dvorak criteria correlates with actual
typing speed data, including frequency regression to control for letter/bigram 
frequency effects in English.

Features:
- Frequency regression for both word-level and bigram-level analyses
- Split analysis by middle column inclusion (T,G,B,Y,H,N)
- Both raw and frequency-adjusted correlation analyses
- Enhanced statistical reporting with multiple comparison correction
- Interaction analysis between criteria
- Progress monitoring and sample limiting for large datasets

Usage:
    python test_dvorak9_speed.py [--max-bigrams N] [--max-words N] [--progress-interval N]
    
Arguments:
    --max-bigrams N       Limit bigram analysis to N samples (default: unlimited)
    --max-words N         Limit word analysis to N samples (default: unlimited)
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
    'max_words': None,
    'progress_interval': 1000,
    'start_time': None
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Dvorak-9 criteria correlation with typing speed')
    parser.add_argument('--max-bigrams', type=int, default=None,
                       help='Limit bigram analysis to N samples (default: unlimited)')
    parser.add_argument('--max-words', type=int, default=None,
                       help='Limit word analysis to N samples (default: unlimited)')
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

def load_frequency_data(letter_freq_path=None, bigram_freq_path=None):
    """Load frequency data from CSV files for regression analysis."""
    letter_frequencies = None
    bigram_frequencies = None
    
    print_and_log("🔍 Loading frequency data for regression analysis...")
    print_and_log("   (This uses pre-calculated English language frequencies, NOT sample frequencies)")
    
    try:
        if letter_freq_path and os.path.exists(letter_freq_path):
            letter_frequencies = pd.read_csv(letter_freq_path)
            print_and_log(f"✅ Loaded letter frequency data: {len(letter_frequencies):,} entries")
            print_and_log(f"   Columns: {list(letter_frequencies.columns)}")
            
            # Show some examples
            if not letter_frequencies.empty:
                print_and_log("   Sample letter frequencies:")
                for i, row in letter_frequencies.head(3).iterrows():
                    if 'item' in row and 'score' in row:
                        print_and_log(f"     '{row['item']}': {row['score']}")
                    else:
                        print_and_log(f"     Row {i}: {dict(row)}")
        else:
            print_and_log(f"⚠️  Letter frequency file not found: {letter_freq_path}")
    except Exception as e:
        print_and_log(f"❌ Error loading letter frequency data: {str(e)}")
    
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
    
    return letter_frequencies, bigram_frequencies

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
            'intercept': model.params[0],
            'slope': model.params[1],
            'p_value': model.pvalues[1] if len(model.pvalues) > 1 else None,
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
        
        if avg_adjustment < 1.0:
            print_and_log(f"  ⚠️  Small adjustment magnitude may explain similar raw/adjusted results")
        
        print_and_log(f"  ✅ Adjusted {sequence_type} times for frequency")
        
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
                
                for criterion in criteria_names.keys():
                    score = scores[criterion]
                    criterion_scores[criterion].append(score)
            
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
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(scores_list, valid_times)
                    
                    # Spearman correlation (rank-based, more robust)
                    spearman_r, spearman_p = spearmanr(scores_list, valid_times)
                    
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
                    continue
        
        print_and_log(f"    Correlation analysis complete")
    
    return results

def read_bigram_times(filename, min_threshold=50, max_threshold=2000, use_percentile_filter=False, max_samples=None):
    """Read bigram times from CSV file with optional filtering and progress monitoring."""
    bigrams = []
    times = []
    
    try:
        # First pass: count total lines
        print_and_log(f"Reading bigram data from {filename}...")
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            total_lines = sum(1 for _ in file) - 1  # Subtract header
        
        print_and_log(f"Found {total_lines:,} bigrams in file")
        
        # Determine sample size
        if max_samples and max_samples < total_lines:
            print_and_log(f"Will randomly sample {max_samples:,} bigrams")
            # Read all lines first, then sample
            all_data = []
            with open(filename, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    bigram = row['bigram'].lower().strip()
                    time = float(row['interkey_interval'])
                    
                    # Only include bigrams with characters in our layout
                    if len(bigram) == 2 and all(c in QWERTY_ITEMS.lower() for c in bigram):
                        all_data.append((bigram, time))
            
            # Random sample
            random.shuffle(all_data)
            sample_data = all_data[:max_samples]
            
            for bigram, time in sample_data:
                bigrams.append(bigram)
                times.append(time)
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
    
    # Apply filtering
    original_count = len(times)
    print_and_log(f"Loaded {original_count:,} valid bigrams")
    
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

def read_word_times(filename, max_threshold=None, use_percentile_filter=False, max_samples=None):
    """Read word times from CSV file with optional filtering and progress monitoring."""
    words = []
    times = []
    
    try:
        # First pass: count total lines
        print_and_log(f"Reading word data from {filename}...")
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            total_lines = sum(1 for _ in file) - 1  # Subtract header
        
        print_and_log(f"Found {total_lines:,} words in file")
        
        # Determine sample size
        if max_samples and max_samples < total_lines:
            print_and_log(f"Will randomly sample {max_samples:,} words")
            # Read all lines first, then sample
            all_data = []
            with open(filename, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    word = row['word'].strip()
                    time = float(row['time'])
                    all_data.append((word, time))
            
            # Random sample
            random.shuffle(all_data)
            sample_data = all_data[:max_samples]
            
            for word, time in sample_data:
                words.append(word)
                times.append(time)
        else:
            # Read all data with progress monitoring
            with open(filename, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                if TQDM_AVAILABLE:
                    reader = tqdm(reader, total=total_lines, desc="Reading words")
                
                processed = 0
                start_time = time.time()
                
                for row in reader:
                    word = row['word'].strip()
                    time_val = float(row['time'])
                    
                    words.append(word)
                    times.append(time_val)
                    
                    processed += 1
                    if not TQDM_AVAILABLE and processed % progress_config['progress_interval'] == 0:
                        print_progress(processed, total_lines, "Reading words", start_time)
    
    except FileNotFoundError:
        print_and_log(f"Error: {filename} not found")
        return [], []
    except Exception as e:
        print_and_log(f"Error reading {filename}: {e}")
        return [], []
    
    if not times or max_threshold is None:
        print_and_log(f"Loaded {len(words):,} words")
        return words, times
    
    # Apply filtering 
    original_count = len(times)
    print_and_log(f"Loaded {original_count:,} words")
    
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
        print_and_log(f"Filtered {removed_count:,}/{original_count:,} words using {filter_method}")
        print_and_log(f"  Kept {len(filtered_times):,} words ({len(filtered_times)/original_count*100:.1f}%)")
        print_and_log(f"  Time range: {min(filtered_times):.1f} - {max(filtered_times):.1f}ms")
    
    return filtered_words, filtered_times

def analyze_bigram_correlations_with_frequency(bigrams, times, bigram_freq_data=None):
    """Analyze correlations between all 9 criteria and bigram times with frequency control."""
    layout_mapping = create_qwerty_mapping()
    
    # All 9 Dvorak criteria
    criteria_names = {
        'hands': 'Hands (alternating)',
        'fingers': 'Fingers (different)',
        'skip_fingers': 'Skip Fingers',
        'dont_cross_home': "Don't Cross Home",
        'same_row': 'Same Row',
        'home_row': 'Home Row',
        'columns': 'Columns',
        'strum': 'Strum (inward)',
        'strong_fingers': 'Strong Fingers'
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

def analyze_word_correlations_with_frequency(words, times, letter_freq_data=None):
    """Analyze correlations between all 9 criteria and word times with frequency control."""
    layout_mapping = create_qwerty_mapping()
    
    # All 9 Dvorak criteria - analyzed at WORD level
    criteria_names = {
        'hands': 'Hands (alternating)',
        'fingers': 'Fingers (different)',
        'skip_fingers': 'Skip Fingers',
        'dont_cross_home': "Don't Cross Home",
        'same_row': 'Same Row',
        'home_row': 'Home Row',
        'columns': 'Columns',
        'strum': 'Strum (inward)',
        'strong_fingers': 'Strong Fingers'
    }
    
    print_and_log("Analyzing word correlations with frequency control...")
    print_and_log("NOTE: All 9 criteria are measured at WORD level")
    print_and_log("for correlation with word typing speed.")
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
        group_results = analyze_criteria_for_group_with_frequency(
            without_middle['sequences'], 
            without_middle['times'], 
            layout_mapping, 
            criteria_names, 
            "Words (No Middle Columns)",
            freq_data=letter_freq_data,
            sequence_type='word'
        )
        for criterion, data in group_results.items():
            results[f"{criterion}_no_middle"] = data
    
    # Group 2: Words WITH middle column keys
    if with_middle['sequences']:
        group_results = analyze_criteria_for_group_with_frequency(
            with_middle['sequences'], 
            with_middle['times'], 
            layout_mapping, 
            criteria_names, 
            "Words (With Middle Columns)",
            freq_data=letter_freq_data,
            sequence_type='word'
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
            s_sig = "***" if sp < 0.001 else "**" if sp < 0.01 else "*" if sp < 0.05 else ""
            effect = "Large" if abs(sr) >= 0.5 else "Med" if abs(sr) >= 0.3 else "Small" if abs(sr) >= 0.1 else "None"
            
            print_and_log(f"{'Raw':<15} {raw_data['n_samples']:<6} {sr:>7.3f}{s_sig:<4} {sp:<8.3f} {effect:<8} {'N/A':<12}")
            
            # Frequency-adjusted analysis
            sr_adj = adj_data['spearman_r']
            sp_adj = adj_data['spearman_p']
            s_sig_adj = "***" if sp_adj < 0.001 else "**" if sp_adj < 0.01 else "*" if sp_adj < 0.05 else ""
            effect_adj = "Large" if abs(sr_adj) >= 0.5 else "Med" if abs(sr_adj) >= 0.3 else "Small" if abs(sr_adj) >= 0.1 else "None"
            
            freq_r2 = adj_data.get('frequency_model', {}).get('r_squared', 0)
            freq_r2_str = f"{freq_r2:.3f}" if freq_r2 else "N/A"
            
            print_and_log(f"{'Freq-Adjusted':<15} {adj_data['n_samples']:<6} {sr_adj:>7.3f}{s_sig_adj:<4} {sp_adj:<8.3f} {effect_adj:<8} {freq_r2_str:<12}")
            
            # Show change in correlation
            change = abs(sr_adj) - abs(sr)
            change_direction = "↑" if change > 0.05 else "↓" if change < -0.05 else "≈"
            print_and_log(f"{'Change':<15} {'':<6} {change:>+7.3f} {change_direction:<8}")

def create_frequency_comparison_plots(results, output_dir="plots"):
    """Create plots comparing raw vs frequency-adjusted correlations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    if not results:
        return
    
    # Prepare data for plotting
    comparison_data = []
    
    for key, data in results.items():
        # Parse the key to extract components
        if key.endswith('_raw_no_middle'):
            criterion_base = key[:-len('_raw_no_middle')]
            analysis_type = 'Raw'
            middle_status = 'Without Middle'
        elif key.endswith('_raw_with_middle'):
            criterion_base = key[:-len('_raw_with_middle')]
            analysis_type = 'Raw'
            middle_status = 'With Middle'
        elif key.endswith('_freq_adjusted_no_middle'):
            criterion_base = key[:-len('_freq_adjusted_no_middle')]
            analysis_type = 'Freq Adjusted'
            middle_status = 'Without Middle'
        elif key.endswith('_freq_adjusted_with_middle'):
            criterion_base = key[:-len('_freq_adjusted_with_middle')]
            analysis_type = 'Freq Adjusted'
            middle_status = 'With Middle'
        else:
            continue
        
        comparison_data.append({
            'criterion': data['name'],
            'middle_status': middle_status,
            'analysis_type': analysis_type,
            'spearman_r': data['spearman_r'],
            'spearman_p': data['spearman_p'],
            'n_samples': data['n_samples']
        })
    
    if not comparison_data:
        return
    
    df = pd.DataFrame(comparison_data)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dvorak-9 Criteria Correlations: Raw vs Frequency-Adjusted', fontsize=16)
    
    # Plot 1: Without Middle Columns - Raw vs Adjusted
    no_middle_data = df[df['middle_status'] == 'Without Middle']
    if not no_middle_data.empty:
        ax = axes[0, 0]
        pivot_data = no_middle_data.pivot(index='criterion', columns='analysis_type', values='spearman_r')
        if not pivot_data.empty:
            pivot_data.plot(kind='bar', ax=ax, color=['blue', 'red'], alpha=0.7)
            ax.set_title('Without Middle Columns')
            ax.set_xlabel('Criterion')
            ax.set_ylabel('Spearman Correlation')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.legend(title='Analysis')
            ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: With Middle Columns - Raw vs Adjusted
    with_middle_data = df[df['middle_status'] == 'With Middle']
    if not with_middle_data.empty:
        ax = axes[0, 1]
        pivot_data = with_middle_data.pivot(index='criterion', columns='analysis_type', values='spearman_r')
        if not pivot_data.empty:
            pivot_data.plot(kind='bar', ax=ax, color=['blue', 'red'], alpha=0.7)
            ax.set_title('With Middle Columns')
            ax.set_xlabel('Criterion')
            ax.set_ylabel('Spearman Correlation')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.legend(title='Analysis')
            ax.tick_params(axis='x', rotation=45)
    
    # Plot 3: Scatter plot - Raw vs Adjusted correlations
    ax = axes[1, 0]
    raw_data = df[df['analysis_type'] == 'Raw']
    adj_data = df[df['analysis_type'] == 'Freq Adjusted']
    
    if not raw_data.empty and not adj_data.empty:
        # Match up corresponding analyses
        merged = pd.merge(raw_data, adj_data, on=['criterion', 'middle_status'], 
                         suffixes=('_raw', '_adj'))
        
        ax.scatter(merged['spearman_r_raw'], merged['spearman_r_adj'], alpha=0.7)
        
        # Add diagonal line
        min_r = min(merged['spearman_r_raw'].min(), merged['spearman_r_adj'].min())
        max_r = max(merged['spearman_r_raw'].max(), merged['spearman_r_adj'].max())
        ax.plot([min_r, max_r], [min_r, max_r], 'k--', alpha=0.5)
        
        # Add labels
        for _, row in merged.iterrows():
            ax.annotate(f"{row['criterion'][:3]}", 
                       (row['spearman_r_raw'], row['spearman_r_adj']),
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Raw Correlation')
        ax.set_ylabel('Frequency-Adjusted Correlation')
        ax.set_title('Raw vs Frequency-Adjusted Correlations')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Effect size comparison
    ax = axes[1, 1]
    if not df.empty:
        df['abs_correlation'] = df['spearman_r'].abs()
        effect_comparison = df.pivot_table(index='criterion', columns='analysis_type', 
                                         values='abs_correlation', aggfunc='mean')
        
        if not effect_comparison.empty:
            effect_comparison.plot(kind='bar', ax=ax, color=['blue', 'red'], alpha=0.7)
            ax.set_title('Effect Sizes (|r|)')
            ax.set_xlabel('Criterion')
            ax.set_ylabel('Absolute Correlation')
            ax.legend(title='Analysis')
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dvorak9_frequency_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return "dvorak9_frequency_comparison.png"

def analyze_criterion_interactions(results, output_dir="plots"):
    """Analyze interactions between criteria using machine learning techniques."""
    if not sklearn_available:
        print_and_log("\nSkipping interaction analysis - scikit-learn not available")
        return
    
    print_and_log("\n" + "=" * 80)
    print_and_log("CRITERION INTERACTION ANALYSIS")
    print_and_log("=" * 80)
    
    # Collect data for interaction analysis
    interaction_data = []
    
    for key, data in results.items():
        if 'scores' in data and 'times' in data and len(data['scores']) > 10:
            # Extract criterion name and conditions
            criterion_name = key.split('_')[0]  # First part is criterion name
            
            for score, time in zip(data['scores'], data['times']):
                interaction_data.append({
                    'criterion': criterion_name,
                    'score': score,
                    'time': time,
                    'group': data['group']
                })
    
    if len(interaction_data) < 100:
        print_and_log("Insufficient data for interaction analysis")
        return
    
    df = pd.DataFrame(interaction_data)
    
    # Create feature matrix for each group
    groups = df['group'].unique()
    
    for group in groups:
        group_data = df[df['group'] == group]
        
        if len(group_data) < 50:
            continue
        
        print_and_log(f"\nAnalyzing interactions for: {group}")
        
        # Pivot to get features (criteria scores) for each observation
        feature_data = group_data.pivot_table(
            index=group_data.index, 
            columns='criterion', 
            values='score', 
            aggfunc='first'
        )
        
        if feature_data.shape[1] < 3:  # Need at least 3 criteria
            continue
        
        # Get corresponding times
        times = group_data.groupby(group_data.index)['time'].first()
        
        # Align indices
        common_indices = feature_data.index.intersection(times.index)
        X = feature_data.loc[common_indices].fillna(0)
        y = times.loc[common_indices]
        
        if len(X) < 20:
            continue
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Linear model with interactions
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = poly.fit_transform(X_scaled)
            
            # Lasso regression to identify important interactions
            lasso = LassoCV(cv=5, random_state=42)
            lasso.fit(X_poly, y)
            
            # Get feature names
            feature_names = poly.get_feature_names_out(X.columns)
            
            # Find significant interactions (non-zero coefficients)
            important_features = []
            for i, coef in enumerate(lasso.coef_):
                if abs(coef) > 0.001:  # Threshold for importance
                    important_features.append((feature_names[i], coef))
            
            # Sort by absolute coefficient value
            important_features.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print_and_log(f"  Model R² = {lasso.score(X_poly, y):.3f}")
            print_and_log(f"  Important features (top 10):")
            
            for feature_name, coef in important_features[:10]:
                effect = "↓ Faster" if coef < 0 else "↑ Slower"
                if ' ' in feature_name:  # Interaction term
                    print_and_log(f"    {feature_name:<25} {coef:>8.3f} {effect}")
                else:  # Main effect
                    print_and_log(f"    {feature_name:<25} {coef:>8.3f} {effect} (main)")
            
        except Exception as e:
            print_and_log(f"  Error in interaction analysis: {e}")

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
        'max_words': args.max_words,
        'progress_interval': args.progress_interval,
        'start_time': time.time()
    })
    
    # Open output file
    output_file = open('test_dvorak9_speed_analysis.txt', 'w', encoding='utf-8')
    
    try:
        print_and_log("Dvorak-9 Criteria Correlation Analysis with Frequency Control")
        print_and_log("=" * 80)
        print_and_log(f"Configuration:")
        print_and_log(f"  Max bigrams: {args.max_bigrams or 'unlimited'}")
        print_and_log(f"  Max words: {args.max_words or 'unlimited'}")
        print_and_log(f"  Progress interval: {args.progress_interval:,}")
        print_and_log(f"  Random seed: {args.random_seed}")
        print_and_log(f"  Middle column keys: {', '.join(sorted(MIDDLE_COLUMN_KEYS))}")
        print_and_log("  Analysis includes both raw and frequency-adjusted correlations")
        print_and_log()
        
        # Load frequency data
        print_and_log("Loading frequency data...")
        letter_freq_file = 'input/letter_frequencies_english.csv'
        bigram_freq_file = 'input/letter_pair_frequencies_english.csv'
        
        letter_frequencies, bigram_frequencies = load_frequency_data(
            letter_freq_file, bigram_freq_file
        )
        
        # Verify we're using language frequencies, not sample frequencies
        print_and_log(f"\n🔬 FREQUENCY DATA VERIFICATION")
        print_and_log(f"{'='*50}")
        print_and_log()
        if letter_frequencies is not None:
            print_and_log(f"✅ Letter frequencies loaded: {len(letter_frequencies):,} entries")
        else:
            print_and_log(f"❌ No letter frequency data available")
            
        if bigram_frequencies is not None:
            print_and_log(f"✅ Bigram frequencies loaded: {len(bigram_frequencies):,} entries") 
        else:
            print_and_log(f"❌ No bigram frequency data available")
        print_and_log(f"{'='*50}")
        print_and_log()
        
        # Read typing data files
        print_and_log("\nReading typing data files...")
        bigram_times_file = '../process_3.5M_keystrokes/output/bigram_times.csv'
        word_times_file = '../process_3.5M_keystrokes/output/word_times.csv'
        
        # FILTERING PARAMETERS
        MIN_INTERVAL = 50
        MAX_INTERVAL = 2000
        USE_PERCENTILE_BIGRAMS = False
        
        MAX_WORD_TIME = None
        USE_PERCENTILE_WORDS = False
        
        # Read the data files with progress monitoring
        bigrams, bigram_times = read_bigram_times(
            bigram_times_file, 
            min_threshold=MIN_INTERVAL,
            max_threshold=MAX_INTERVAL, 
            use_percentile_filter=USE_PERCENTILE_BIGRAMS,
            max_samples=args.max_bigrams
        )
        
        words, word_times = read_word_times(
            word_times_file,
            max_threshold=MAX_WORD_TIME,
            use_percentile_filter=USE_PERCENTILE_WORDS,
            max_samples=args.max_words
        )
        
        if not bigrams and not words:
            print_and_log("Error: No valid data found in CSV files.")
            sys.exit(1)
        
        # Create output directory for plots
        Path("plots").mkdir(exist_ok=True)
        print_and_log("Creating plots in 'plots/' directory...")
        
        # Analyze correlations with frequency control
        print_and_log(f"\n{'='*80}")
        print_and_log("STARTING CORRELATION ANALYSIS")
        print_and_log(f"{'='*80}")
        
        analysis_start_time = time.time()
        bigram_results = {}
        word_results = {}
        
        if bigrams:
            print_and_log(f"\nBIGRAM ANALYSIS")
            print_and_log(f"Processing {len(bigrams):,} bigrams...")
            bigram_start = time.time()
            
            bigram_results = analyze_bigram_correlations_with_frequency(
                bigrams, bigram_times, bigram_frequencies
            )
            
            bigram_elapsed = time.time() - bigram_start
            print_and_log(f"Bigram analysis completed in {format_time(bigram_elapsed)}")
        
        if words:
            print_and_log(f"\nWORD ANALYSIS")
            print_and_log(f"Processing {len(words):,} words...")
            word_start = time.time()
            
            word_results = analyze_word_correlations_with_frequency(
                words, word_times, letter_frequencies
            )
            
            word_elapsed = time.time() - word_start
            print_and_log(f"Word analysis completed in {format_time(word_elapsed)}")
        
        # Combine all results
        all_results = {}
        all_results.update(bigram_results)
        all_results.update(word_results)
        
        total_analysis_time = time.time() - analysis_start_time
        print_and_log(f"\nTotal analysis time: {format_time(total_analysis_time)}")
        
        if all_results:
            # Print correlation tables with frequency comparison
            print_correlation_results_with_frequency(
                all_results, 
                "CORRELATION ANALYSIS: RAW vs FREQUENCY-ADJUSTED"
            )
            
            # Create frequency comparison plots
            print_and_log(f"\nGenerating comparison plots...")
            plot_start = time.time()
            create_frequency_comparison_plots(all_results)
            plot_elapsed = time.time() - plot_start
            print_and_log(f"Plot generation completed in {format_time(plot_elapsed)}")
            
            # Analyze criterion interactions
            print_and_log(f"\nAnalyzing criterion interactions...")
            interaction_start = time.time()
            analyze_criterion_interactions(all_results)
            interaction_elapsed = time.time() - interaction_start
            print_and_log(f"Interaction analysis completed in {format_time(interaction_elapsed)}")
            
            # Apply multiple comparisons correction
            print_and_log("\n" + "=" * 80)
            print_and_log("MULTIPLE COMPARISONS CORRECTION")
            print_and_log("=" * 80)
            
            # Extract p-values for both raw and adjusted analyses
            p_values_raw = []
            p_values_adj = []
            keys_raw = []
            keys_adj = []
            
            for key, data in all_results.items():
                if 'spearman_p' in data:
                    if '_raw_' in key:
                        p_values_raw.append(data['spearman_p'])
                        keys_raw.append(key)
                    elif '_freq_adjusted_' in key:
                        p_values_adj.append(data['spearman_p'])
                        keys_adj.append(key)
            
            # Apply FDR correction separately
            alpha = 0.05
            
            if p_values_raw:
                rejected_raw, p_adj_raw, _, _ = multipletests(p_values_raw, alpha=alpha, method='fdr_bh')
                print_and_log(f"\nRaw Analysis - Significant after FDR correction (α = {alpha}):")
                any_sig_raw = False
                for i, key in enumerate(keys_raw):
                    if rejected_raw[i]:
                        any_sig_raw = True
                        data = all_results[key]
                        direction = "↓ Faster" if data['spearman_r'] < 0 else "↑ Slower"
                        print_and_log(f"  {data['name']} ({data['group']}): r={data['spearman_r']:.3f}, p_adj={p_adj_raw[i]:.3f} {direction}")
                if not any_sig_raw:
                    print_and_log("  None significant after correction")
            
            if p_values_adj:
                rejected_adj, p_adj_adj, _, _ = multipletests(p_values_adj, alpha=alpha, method='fdr_bh')
                print_and_log(f"\nFrequency-Adjusted Analysis - Significant after FDR correction (α = {alpha}):")
                any_sig_adj = False
                for i, key in enumerate(keys_adj):
                    if rejected_adj[i]:
                        any_sig_adj = True
                        data = all_results[key]
                        direction = "↓ Faster" if data['spearman_r'] < 0 else "↑ Slower"
                        print_and_log(f"  {data['name']} ({data['group']}): r={data['spearman_r']:.3f}, p_adj={p_adj_adj[i]:.3f} {direction}")
                if not any_sig_adj:
                    print_and_log("  None significant after correction")
        
        total_elapsed = time.time() - progress_config['start_time']
        
        print_and_log(f"\n" + "=" * 80)
        print_and_log("ANALYSIS COMPLETE")
        print_and_log("=" * 80)
        print_and_log(f"Total runtime: {format_time(total_elapsed)}")
        print_and_log(f"Key outputs saved:")
        print_and_log(f"- Text output: dvorak9_analysis_results_with_frequency.txt")
        print_and_log(f"- Comparison plots: plots/dvorak9_frequency_comparison.png")
        print_and_log(f"- Each criterion analyzed with and without frequency control")
        
        if args.max_bigrams or args.max_words:
            print_and_log(f"\nNote: Analysis used sample limits:")
            if args.max_bigrams:
                print_and_log(f"- Bigrams: {args.max_bigrams:,} (from {len(bigrams):,} processed)")
            if args.max_words:
                print_and_log(f"- Words: {args.max_words:,} (from {len(words):,} processed)")
        
    finally:
        if output_file:
            output_file.close()

if __name__ == "__main__":
    main()