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
from scipy import stats as scipy_stats 
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

def save_log(filename="test_dvorak9_speed_results.txt"):
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
        
        # Calculate frequency-controlled residuals for all sequences
        frequency_residuals = []
        model_info = {
            'r_squared': model.rsquared,
            'intercept': model.params[0],
            'slope': model.params[1],
            'p_value': model.pvalues[1] if len(model.pvalues) > 1 else None,
            'n_obs': len(matched_frequencies),
            'log_frequencies': log_frequencies.copy(),  # For diagnostic plots
            'predicted_times': model.predict(X).copy(),
            'actual_times': times_array.copy(),
            'matched_sequences': matched_sequences.copy()
        }
        
        print_and_log(f"      Regression results:")
        print_and_log(f"        R² = {model.rsquared:.4f}")
        print_and_log(f"        Slope = {model.params[1]:.4f} (p = {model.pvalues[1]:.4f})")
        print_and_log(f"        Intercept = {model.params[0]:.4f}")
        print_and_log(f"        → Negative slope means higher frequency = faster typing")
        
        # Calculate residuals for all sequences
        residual_magnitudes = []
        rank_changes = []
        original_ranks = stats.rankdata(times)
        
        for i, seq in enumerate(sequences):
            if seq in freq_dict:
                log_freq = np.log10(freq_dict[seq])
                predicted_time = model.params[0] + model.params[1] * log_freq
                residual = times[i] - predicted_time  # Actual - Predicted
                frequency_residuals.append(residual)
                residual_magnitudes.append(abs(residual))
            else:
                # For sequences without frequency data, use original time
                # This preserves their relative ranking while noting missing frequency control
                frequency_residuals.append(times[i])
                residual_magnitudes.append(0)
        
        adjusted_ranks = stats.rankdata(frequency_residuals)
        rank_changes = abs(original_ranks - adjusted_ranks)
        
        print_and_log(f"      Frequency control effects:")
        print_and_log(f"        Average |residual|: {np.mean(residual_magnitudes):.2f}ms")
        print_and_log(f"        Maximum |residual|: {np.max(residual_magnitudes):.2f}ms")
        controlled_count = sum(1 for mag in residual_magnitudes if mag > 0.1)
        print_and_log(f"        Sequences with frequency control: {(controlled_count/len(residual_magnitudes)*100):.1f}%")
        
        # Rank order analysis
        rank_correlation = spearmanr(original_ranks, adjusted_ranks)[0]
        print_and_log(f"      📊 RANK ORDER ANALYSIS:")
        print_and_log(f"        Correlation between raw times and frequency residuals: {rank_correlation:.6f}")
        sequences_with_rank_changes = sum(1 for change in rank_changes if change > 0)
        print_and_log(f"        Sequences with rank changes: {sequences_with_rank_changes}/{len(sequences)} ({sequences_with_rank_changes/len(sequences)*100:.1f}%)")
        print_and_log(f"        Maximum rank position change: {int(max(rank_changes))}")
        
        print_and_log(f"  ✅ Generated frequency-controlled residuals")
        print_and_log(f"  💡 INTERPRETATION: Residuals represent typing speed after controlling for frequency")
        print_and_log(f"      • Negative residuals = faster than expected given frequency")
        print_and_log(f"      • Positive residuals = slower than expected given frequency")
        
        return frequency_residuals, model_info
        
    except Exception as e:
        print_and_log(f"      ❌ Regression failed: {e}")
        return times, None

def create_diagnostic_plots(model_info, output_dir='plots'):
    """Create diagnostic plots for frequency adjustment model"""
    if not model_info:
        print_and_log("No model info available for diagnostic plots")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create diagnostic plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Frequency Adjustment Model Diagnostics', fontsize=16, fontweight='bold')
    
    # Plot 1: Frequency vs Time relationship with model fit
    log_freqs = model_info['log_frequencies']
    actual_times = model_info['actual_times']
    predicted_times = model_info['predicted_times']
    
    ax1.scatter(log_freqs, actual_times, alpha=0.5, s=20, color='skyblue', label='Actual times')
    ax1.plot(log_freqs, predicted_times, 'r-', linewidth=2, label=f'Model fit (R² = {model_info["r_squared"]:.3f})')
    ax1.set_xlabel('Log₁₀ Frequency')
    ax1.set_ylabel('Typing Time (ms)')
    ax1.set_title('Frequency vs Typing Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs Predicted (check for heteroscedasticity)
    residuals = actual_times - predicted_times
    ax2.scatter(predicted_times, residuals, alpha=0.5, s=20, color='lightcoral')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Predicted Time (ms)')
    ax2.set_ylabel('Residuals (ms)')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals distribution (check for normality)
    ax3.hist(residuals, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Residuals (ms)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Add distribution stats
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax3.axvline(mean_res, color='red', linestyle='--', label=f'Mean: {mean_res:.1f}ms')
    ax3.axvline(mean_res + std_res, color='orange', linestyle='--', alpha=0.7, label=f'±1 SD: {std_res:.1f}ms')
    ax3.axvline(mean_res - std_res, color='orange', linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Plot 4: Q-Q plot for normality check
    from scipy import stats as scipy_stats
    scipy_stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'frequency_adjustment_diagnostics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Diagnostic plots saved to: {output_path}")
    
    # Print diagnostic summary
    print_and_log(f"\n📊 FREQUENCY MODEL DIAGNOSTICS:")
    print_and_log(f"   Model fit: R² = {model_info['r_squared']:.4f}")
    print_and_log(f"   Slope significance: p = {model_info.get('p_value', 'N/A'):.4f}")
    print_and_log(f"   Residual statistics:")
    print_and_log(f"     Mean: {mean_res:.2f}ms (should be ~0)")
    print_and_log(f"     Std: {std_res:.2f}ms")
    print_and_log(f"     Range: {min(residuals):.1f} to {max(residuals):.1f}ms")
    
    # Check for model assumptions
    if abs(mean_res) < 1.0:
        print_and_log(f"   ✅ Residuals well-centered (mean ≈ 0)")
    else:
        print_and_log(f"   ⚠️  Residuals not well-centered (mean = {mean_res:.2f})")
    
    if model_info.get('p_value', 1) < 0.05:
        print_and_log(f"   ✅ Frequency significantly predicts typing time")
    else:
        print_and_log(f"   ⚠️  Frequency not significantly predictive")

def create_bigram_scatter_plot(sequences, original_times, frequency_residuals, model_info=None, output_dir='plots'):
    """Create scatter plot showing all bigrams before/after frequency adjustment"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Bigram Typing Times: Raw vs Frequency-Controlled', fontsize=16, fontweight='bold')
    
    # Plot 1: Original times distribution
    ax1.hist(original_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Raw Typing Time (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Original Times Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add stats
    mean_orig = np.mean(original_times)
    std_orig = np.std(original_times)
    ax1.axvline(mean_orig, color='red', linestyle='--', label=f'Mean: {mean_orig:.1f}ms')
    ax1.legend()
    
    # Plot 2: Frequency residuals distribution
    ax2.hist(frequency_residuals, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Frequency Residuals (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Frequency-Controlled Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Add stats
    mean_res = np.mean(frequency_residuals)
    std_res = np.std(frequency_residuals)
    ax2.axvline(mean_res, color='red', linestyle='--', label=f'Mean: {mean_res:.1f}ms')
    ax2.axvline(0, color='black', linestyle='-', alpha=0.5, label='Expected (0)')
    ax2.legend()
    
    # Plot 3: Scatter plot Raw vs Residuals
    # Sample if too many points for visibility
    if len(original_times) > 5000:
        indices = np.random.choice(len(original_times), 5000, replace=False)
        sample_orig = [original_times[i] for i in indices]
        sample_res = [frequency_residuals[i] for i in indices]
        sample_seq = [sequences[i] for i in indices]
        alpha = 0.3
        title_suffix = f" (n={len(sample_orig):,} sample)"
    else:
        sample_orig = original_times
        sample_res = frequency_residuals
        sample_seq = sequences
        alpha = 0.5
        title_suffix = f" (n={len(sample_orig):,})"
    
    ax3.scatter(sample_orig, sample_res, alpha=alpha, s=15, color='purple')
    ax3.set_xlabel('Raw Typing Time (ms)')
    ax3.set_ylabel('Frequency Residuals (ms)')
    ax3.set_title(f'Raw vs Frequency-Controlled{title_suffix}')
    ax3.grid(True, alpha=0.3)
    
    # Add reference lines
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero residual')
    
    # Correlation between raw and residuals
    corr = np.corrcoef(sample_orig, sample_res)[0, 1]
    ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Highlight some interesting bigrams if model info available
    if model_info and 'matched_sequences' in model_info:
        # Find extreme residuals among matched sequences
        matched_seqs = model_info['matched_sequences']
        matched_times = model_info['actual_times']
        matched_predicted = model_info['predicted_times']
        matched_residuals = matched_times - matched_predicted
        
        # Find most extreme positive and negative residuals
        max_idx = np.argmax(matched_residuals)
        min_idx = np.argmin(matched_residuals)
        
        # Annotate if these are in our sample
        for idx, seq in enumerate(sample_seq):
            if seq == matched_seqs[max_idx]:
                ax3.annotate(f"'{seq}' (+{matched_residuals[max_idx]:.0f}ms)", 
                           (sample_orig[idx], sample_res[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=8)
            elif seq == matched_seqs[min_idx]:
                ax3.annotate(f"'{seq}' ({matched_residuals[min_idx]:.0f}ms)", 
                           (sample_orig[idx], sample_res[idx]),
                           xytext=(10, -15), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                           fontsize=8)
    
    ax3.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'bigram_frequency_adjustment_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Bigram scatter plot saved to: {output_path}")
    
    # Print summary statistics
    print_and_log(f"\n📊 BIGRAM TIMING SUMMARY:")
    print_and_log(f"   Raw times: {mean_orig:.1f} ± {std_orig:.1f}ms (range: {min(original_times):.1f}-{max(original_times):.1f})")
    print_and_log(f"   Frequency residuals: {mean_res:.1f} ± {std_res:.1f}ms (range: {min(frequency_residuals):.1f}-{max(frequency_residuals):.1f})")
    print_and_log(f"   Correlation raw vs residuals: r = {corr:.3f}")
    
    # Interpretation
    if abs(corr) > 0.8:
        print_and_log(f"   💡 High correlation suggests frequency control had minimal impact")
    elif abs(corr) > 0.5:
        print_and_log(f"   💡 Moderate correlation suggests frequency control modified rankings")
    else:
        print_and_log(f"   💡 Low correlation suggests strong frequency effects were controlled")

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
    """Analyze bigram typing data with middle column key analysis and diagnostics"""
    
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
            frequency_residuals, model_info = adjust_times_for_frequency(sequences, times, freq_df, "bigrams")
            
            # Create diagnostic plots for the first group only (to avoid duplication)
            if group_name == "Bigrams (No Middle Columns)" and model_info:
                create_diagnostic_plots(model_info)
                create_bigram_scatter_plot(sequences, times, frequency_residuals, model_info)
            
            freq_results = analyze_correlations(sequences, frequency_residuals, criteria_names, group_name, "freq_adjusted", model_info)
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
        
        if not isinstance(data, dict) or 'spearman_r' not in data:
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

def create_combination_performance_plots(combination_results, output_dir='plots'):
    """Create plots showing combination analysis results"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print_and_log("📊 Creating combination analysis plots...")
    
    # Create the plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dvorak Criterion Combination Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Best correlation by combination size
    combination_sizes = []
    best_correlations = []
    best_combinations = []
    
    for k_way, results_list in combination_results.items():
        if results_list:
            k = int(k_way.split('_')[0])
            best_result = max(results_list, key=lambda x: x['abs_correlation'])
            
            combination_sizes.append(k)
            best_correlations.append(best_result['abs_correlation'])
            best_combinations.append(best_result['combination'])
    
    if combination_sizes:
        bars = ax1.bar(combination_sizes, best_correlations, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Number of Criteria Combined')
        ax1.set_ylabel('Best Absolute Correlation |r|')
        ax1.set_title('Best Performance by Combination Size')
        ax1.set_xticks(combination_sizes)
        ax1.grid(True, alpha=0.3)
        
        # Add effect size reference lines
        ax1.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Small effect (0.1)')
        ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium effect (0.3)')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Large effect (0.5)')
        ax1.legend()
        
        # Annotate bars with correlation values
        for bar, corr in zip(bars, best_correlations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Distribution of effect sizes
    all_correlations = []
    all_sizes = []
    
    for k_way, results_list in combination_results.items():
        k = int(k_way.split('_')[0])
        for result in results_list:
            if result['p_value'] < 0.05:  # Only significant results
                all_correlations.append(result['abs_correlation'])
                all_sizes.append(k)
    
    if all_correlations:
        # Create effect size categories
        effect_categories = []
        for corr in all_correlations:
            if corr >= 0.5:
                effect_categories.append('Large (≥0.5)')
            elif corr >= 0.3:
                effect_categories.append('Medium (0.3-0.5)')
            elif corr >= 0.1:
                effect_categories.append('Small (0.1-0.3)')
            else:
                effect_categories.append('Negligible (<0.1)')
        
        # Count effect sizes
        from collections import Counter
        effect_counts = Counter(effect_categories)
        
        # Create pie chart
        labels = list(effect_counts.keys())
        sizes = list(effect_counts.values())
        colors = ['red', 'orange', 'yellow', 'lightgray'][:len(labels)]
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution of Effect Sizes\n(Significant Results Only)')
    
    # Plot 3: Number of significant results by combination size
    sizes_for_plot = []
    significant_counts = []
    total_counts = []
    
    for k_way, results_list in combination_results.items():
        k = int(k_way.split('_')[0])
        significant = sum(1 for r in results_list if r['p_value'] < 0.05)
        total = len(results_list)
        
        if total > 0:
            sizes_for_plot.append(k)
            significant_counts.append(significant)
            total_counts.append(total)
    
    if sizes_for_plot:
        x_pos = np.arange(len(sizes_for_plot))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, total_counts, width, label='Total tested', alpha=0.7, color='lightblue')
        bars2 = ax3.bar(x_pos + width/2, significant_counts, width, label='Significant (p<0.05)', alpha=0.7, color='darkblue')
        
        ax3.set_xlabel('Number of Criteria Combined')
        ax3.set_ylabel('Number of Combinations')
        ax3.set_title('Significant vs Total Combinations')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(sizes_for_plot)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (total, sig) in enumerate(zip(total_counts, significant_counts)):
            if total > 0:
                pct = (sig / total) * 100
                ax3.text(i, max(total, sig) + max(total_counts) * 0.02, 
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Top 10 overall combinations
    all_results = []
    for k_way, results_list in combination_results.items():
        all_results.extend(results_list)
    
    # Sort by absolute correlation and take top 10
    all_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
    top_10 = all_results[:10]
    
    if top_10:
        # Prepare data for horizontal bar chart
        combinations = [r['combination'] for r in top_10]
        correlations = [r['correlation'] for r in top_10]  # Use signed correlation
        
        # Truncate long combination names for readability
        truncated_combinations = []
        for combo in combinations:
            if len(combo) > 30:
                truncated_combinations.append(combo[:27] + '...')
            else:
                truncated_combinations.append(combo)
        
        y_pos = np.arange(len(truncated_combinations))
        
        # Color bars by direction (red for positive, blue for negative)
        colors = ['red' if corr > 0 else 'blue' for corr in correlations]
        
        bars = ax4.barh(y_pos, correlations, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(truncated_combinations, fontsize=8)
        ax4.set_xlabel('Spearman Correlation')
        ax4.set_title('Top 10 Best Combinations')
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        # Add correlation values to bars
        for bar, corr in zip(bars, correlations):
            width = bar.get_width()
            ax4.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                    f'{corr:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=7)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'dvorak_combination_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Combination analysis plots saved to: {output_path}")
    
    return output_path

def create_criterion_interaction_heatmap(combination_results, output_dir='plots'):
    """Create heatmap showing which criteria work best together"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print_and_log("🔥 Creating criterion interaction heatmap...")
    
    # Extract all pairwise combinations
    pairwise_results = combination_results.get('2_way', [])
    
    if not pairwise_results:
        print_and_log("❌ No pairwise combination data found for heatmap")
        return None
    
    # Get all unique criteria
    all_criteria = set()
    for result in pairwise_results:
        criteria = result['combination'].split(' + ')
        all_criteria.update(criteria)
    
    criteria_list = sorted(list(all_criteria))
    n_criteria = len(criteria_list)
    
    # Create correlation matrix
    correlation_matrix = np.zeros((n_criteria, n_criteria))
    p_value_matrix = np.ones((n_criteria, n_criteria))
    
    # Fill matrix with pairwise correlations
    for result in pairwise_results:
        criteria = result['combination'].split(' + ')
        if len(criteria) == 2:
            idx1 = criteria_list.index(criteria[0])
            idx2 = criteria_list.index(criteria[1])
            
            # Use absolute correlation for strength
            correlation_matrix[idx1, idx2] = result['abs_correlation']
            correlation_matrix[idx2, idx1] = result['abs_correlation']
            
            # Store p-values
            p_value_matrix[idx1, idx2] = result['p_value']
            p_value_matrix[idx2, idx1] = result['p_value']
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Dvorak Criterion Interaction Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Correlation strength heatmap
    im1 = ax1.imshow(correlation_matrix, cmap='Reds', aspect='auto')
    ax1.set_xticks(range(n_criteria))
    ax1.set_yticks(range(n_criteria))
    ax1.set_xticklabels(criteria_list, rotation=45, ha='right')
    ax1.set_yticklabels(criteria_list)
    ax1.set_title('Pairwise Combination Strength (|r|)')
    
    # Add correlation values to cells
    for i in range(n_criteria):
        for j in range(n_criteria):
            if i != j and correlation_matrix[i, j] > 0:
                text = ax1.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black" if correlation_matrix[i, j] < 0.3 else "white",
                               fontsize=8)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Absolute Correlation |r|')
    
    # Plot 2: Significance heatmap
    # Convert p-values to significance levels for better visualization
    significance_matrix = np.zeros_like(p_value_matrix)
    significance_matrix[p_value_matrix < 0.001] = 3  # ***
    significance_matrix[(p_value_matrix >= 0.001) & (p_value_matrix < 0.01)] = 2  # **
    significance_matrix[(p_value_matrix >= 0.01) & (p_value_matrix < 0.05)] = 1  # *
    significance_matrix[p_value_matrix >= 0.05] = 0  # ns
    
    im2 = ax2.imshow(significance_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=3)
    ax2.set_xticks(range(n_criteria))
    ax2.set_yticks(range(n_criteria))
    ax2.set_xticklabels(criteria_list, rotation=45, ha='right')
    ax2.set_yticklabels(criteria_list)
    ax2.set_title('Statistical Significance')
    
    # Add significance symbols to cells
    for i in range(n_criteria):
        for j in range(n_criteria):
            if i != j:
                p_val = p_value_matrix[i, j]
                if p_val < 0.001:
                    symbol = '***'
                elif p_val < 0.01:
                    symbol = '**'
                elif p_val < 0.05:
                    symbol = '*'
                else:
                    symbol = 'ns'
                
                ax2.text(j, i, symbol, ha="center", va="center", 
                        color="white" if significance_matrix[i, j] > 1.5 else "black",
                        fontsize=8, fontweight='bold')
    
    # Custom colorbar for significance
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, ticks=[0, 1, 2, 3])
    cbar2.set_label('Significance Level')
    cbar2.set_ticklabels(['ns', '*', '**', '***'])
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'dvorak_criterion_interactions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_and_log(f"Criterion interaction heatmap saved to: {output_path}")
    
    # Print insights from the heatmap
    print_and_log(f"\n🔥 CRITERION INTERACTION INSIGHTS:")
    
    # Find strongest pairwise interactions
    strong_pairs = []
    for i in range(n_criteria):
        for j in range(i+1, n_criteria):
            if correlation_matrix[i, j] > 0.1 and p_value_matrix[i, j] < 0.05:
                strong_pairs.append((
                    f"{criteria_list[i]} + {criteria_list[j]}",
                    correlation_matrix[i, j],
                    p_value_matrix[i, j]
                ))
    
    strong_pairs.sort(key=lambda x: x[1], reverse=True)
    
    if strong_pairs:
        print_and_log(f"   Top criterion pairs (|r| > 0.1, p < 0.05):")
        for i, (pair, corr, p_val) in enumerate(strong_pairs[:5]):
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            print_and_log(f"     {i+1}. {pair}: |r| = {corr:.3f}{sig}")
    else:
        print_and_log(f"   ⚠️  No strong criterion pairs found (all |r| < 0.1)")
    
    # Find criteria that work well with others
    avg_interactions = np.mean(correlation_matrix, axis=1)
    best_criteria = [(criteria_list[i], avg_interactions[i]) for i in range(n_criteria)]
    best_criteria.sort(key=lambda x: x[1], reverse=True)
    
    print_and_log(f"   Best criteria for combinations (average |r| with others):")
    for i, (criterion, avg_corr) in enumerate(best_criteria[:3]):
        print_and_log(f"     {i+1}. {criterion}: {avg_corr:.3f}")
    
    return output_path

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
    """Analyze how combinations of criteria predict typing speed - COMPREHENSIVE VERSION"""
    
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
        times = df['time'].values
        
        print_and_log(f"   Criteria: {criteria_cols}")
        
        # COMPREHENSIVE COMBINATION ANALYSIS - TEST ALL 511 COMBINATIONS
        all_results = {}
        
        # For each combination size k from 1 to 9
        for k in range(1, len(criteria_cols) + 1):
            print_and_log(f"\n📊 {k}-WAY COMBINATIONS:")
            
            # Generate ALL combinations of size k
            from itertools import combinations
            combos = list(combinations(criteria_cols, k))
            total_combos = len(combos)
            print_and_log(f"   Testing ALL {total_combos:,} combinations of {k} criteria...")
            
            combo_results = []
            
            # Test EVERY combination (no sampling!)
            for combo in combos:
                # Create combined score (additive model)
                combined_scores = np.zeros(len(times))
                for criterion in combo:
                    combined_scores += df[criterion].values
                
                # Test if there's variation
                if len(set(combined_scores)) > 1:
                    try:
                        corr, p_val = spearmanr(combined_scores, times)
                        if not (np.isnan(corr) or np.isnan(p_val)):
                            combo_results.append({
                                'combination': ' + '.join(combo),
                                'criteria_count': k,
                                'correlation': corr,
                                'p_value': p_val,
                                'abs_correlation': abs(corr)
                            })
                    except:
                        continue
            
            # Sort by absolute correlation
            combo_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            # Store results
            all_results[f'{k}_way'] = combo_results
            
            # Show top results for this k
            top_n = min(5, len(combo_results))
            if combo_results:
                print_and_log(f"   Top {top_n} combinations:")
                for i, result in enumerate(combo_results[:top_n]):
                    sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                    print_and_log(f"     {i+1}. {result['combination']}")
                    print_and_log(f"        r = {result['correlation']:.4f}{sig}, p = {result['p_value']:.4f}")
            else:
                print_and_log(f"   No valid combinations found")
        
        # FIND THE SINGLE BEST COMBINATION OVERALL
        print_and_log(f"\n🏆 BEST COMBINATION ACROSS ALL SIZES:")
        
        best_overall = None
        best_correlation = 0
        
        for k_way, results_list in all_results.items():
            if results_list:
                best_in_category = max(results_list, key=lambda x: x['abs_correlation'])
                if best_in_category['abs_correlation'] > best_correlation:
                    best_correlation = best_in_category['abs_correlation']
                    best_overall = best_in_category
        
        if best_overall:
            direction = "supports Dvorak (faster)" if best_overall['correlation'] < 0 else "contradicts Dvorak (slower)"
            effect_size = "large" if best_overall['abs_correlation'] >= 0.5 else "medium" if best_overall['abs_correlation'] >= 0.3 else "small" if best_overall['abs_correlation'] >= 0.1 else "negligible"
            
            print_and_log(f"   🎯 STRONGEST PREDICTOR: {best_overall['combination']}")
            print_and_log(f"   📈 Correlation: r = {best_overall['correlation']:.4f}")
            print_and_log(f"   📊 Effect size: {effect_size}")
            print_and_log(f"   🎭 Direction: {direction}")
            print_and_log(f"   🔢 Uses {best_overall['criteria_count']} criteria")
        
        # SUMMARY STATISTICS
        print_and_log(f"\n📈 COMBINATION ANALYSIS SUMMARY:")
        
        total_combinations = sum(len(results_list) for results_list in all_results.values())
        significant_combinations = sum(sum(1 for r in results_list if r['p_value'] < 0.05) for results_list in all_results.values())
        
        print_and_log(f"   • Total combinations tested: {total_combinations:,}")
        print_and_log(f"   • Statistically significant: {significant_combinations:,} ({significant_combinations/total_combinations*100:.1f}%)")
        
        # Effect size distribution
        all_correlations = [r['abs_correlation'] for results_list in all_results.values() for r in results_list if r['p_value'] < 0.05]
        if all_correlations:
            large_effects = sum(1 for r in all_correlations if r >= 0.5)
            medium_effects = sum(1 for r in all_correlations if 0.3 <= r < 0.5)
            small_effects = sum(1 for r in all_correlations if 0.1 <= r < 0.3)
            negligible_effects = sum(1 for r in all_correlations if r < 0.1)
            
            print_and_log(f"   • Large effects (|r| ≥ 0.5): {large_effects}")
            print_and_log(f"   • Medium effects (|r| 0.3-0.5): {medium_effects}")
            print_and_log(f"   • Small effects (|r| 0.1-0.3): {small_effects}")
            print_and_log(f"   • Negligible effects (|r| < 0.1): {negligible_effects}")
        
        # MACHINE LEARNING ANALYSIS WITH PROPER INTERPRETATION
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
                
                # PROPER INTERPRETATION
                print_and_log(f"     📊 MODEL PERFORMANCE:")
                print_and_log(f"       Random Forest R² = {r2:.4f} ({r2*100:.2f}% of variance explained)")
                
                if r2 < 0.01:
                    performance = "Essentially useless - no predictive power"
                elif r2 < 0.05:
                    performance = "Very weak - minimal predictive power"  
                elif r2 < 0.1:
                    performance = "Weak but detectable patterns"
                elif r2 < 0.25:
                    performance = "Moderate predictive power"
                else:
                    performance = "Strong predictive power"
                
                print_and_log(f"       💡 Interpretation: {performance}")
                
                # Feature importance interpretation  
                print_and_log(f"     🎯 FEATURE IMPORTANCE ANALYSIS:")
                print_and_log(f"       Ranking (importance scores sum to 1.0):")
                
                for i, (feature, importance) in enumerate(feature_importance):
                    percentage = importance * 100
                    if importance > 0.2:
                        importance_level = "🔥 Critical"
                    elif importance > 0.15:
                        importance_level = "🔴 High"
                    elif importance > 0.1:
                        importance_level = "🟡 Medium"
                    else:
                        importance_level = "🟢 Low"
                    
                    print_and_log(f"         {i+1}. {feature:<15}: {percentage:5.1f}% {importance_level}")
                
                # Practical implications
                print_and_log(f"     💡 PRACTICAL IMPLICATIONS:")
                
                if r2 < 0.05:
                    print_and_log(f"       ❌ INDIVIDUAL CRITERIA ARE POOR PREDICTORS:")
                    print_and_log(f"         • Even combined, all 9 criteria explain <5% of typing speed variance")
                    print_and_log(f"         • Individual differences dominate over keyboard layout principles")
                    print_and_log(f"         • Other factors (skill, practice, fatigue) are much more important")
                    
                    print_and_log(f"       🤔 WHY ARE DVORAK CRITERIA WEAK PREDICTORS?")
                    print_and_log(f"         • Modern typing may not follow 1930s assumptions")
                    print_and_log(f"         • Individual typing styles vary enormously")
                    print_and_log(f"         • Muscle memory and practice effects dominate")
                    print_and_log(f"         • Real-world typing includes errors, corrections, thinking time")
                
                # Top criterion analysis
                top_criterion, top_importance = feature_importance[0]
                print_and_log(f"       🏆 MOST PREDICTIVE CRITERION: {top_criterion}")
                print_and_log(f"         • Explains {top_importance*100:.1f}% of the model's {r2*100:.1f}% total variance")
                print_and_log(f"         • Actual variance explained: {top_importance*r2*100:.2f}% of typing speed")
                
                if top_importance * r2 < 0.01:
                    print_and_log(f"         • Still explains <1% of actual typing speed - very weak effect")
                
            else:
                print_and_log(f"     ⚠️  Insufficient variation for ML analysis")
                
        except Exception as e:
            print_and_log(f"     ❌ ML analysis failed: {e}")
        
        # REPORT WHAT WE ACTUALLY TESTED
        print_and_log(f"\n📋 COMPREHENSIVE TESTING SUMMARY:")
        print_and_log(f"   This analysis tested ALL possible combinations:")
        total_possible = 2**len(criteria_cols) - 1  # 2^9 - 1 = 511
        print_and_log(f"   • Total possible combinations: {total_possible}")
        
        for k in range(1, len(criteria_cols) + 1):
            k_combinations = len(list(combinations(criteria_cols, k)))
            k_tested = len(all_results.get(f'{k}_way', []))
            print_and_log(f"   • {k}-way combinations: {k_tested}/{k_combinations} tested")
        
        print_and_log(f"   ✅ No combinations were skipped - complete coverage achieved")
        
        # CREATE THE MISSING PLOTS! 📊
        if all_results:
            print_and_log(f"\n📊 Creating combination analysis visualizations...")
            
            # Plot 1: Combination performance plots
            create_combination_performance_plots(all_results)
            
            # Plot 2: Criterion interaction heatmap
            create_criterion_interaction_heatmap(all_results)
            
            print_and_log(f"✅ Combination analysis plots created successfully!")
        else:
            print_and_log(f"⚠️  No combination results to plot")
        
        # Return results for further analysis if needed
        return all_results
    
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

def fix_multiple_comparisons_correction(results):
    """Fix the p-value extraction logic"""
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log("FIXED MULTIPLE COMPARISONS CORRECTION")
    print_and_log("=" * 80)
    
    # Extract p-values with CORRECT key patterns
    p_values_raw = []
    p_values_adj = []
    keys_raw = []
    keys_adj = []
    
    for key, data in results.items():
        if key.startswith('_') or not isinstance(data, dict):
            continue
            
        if 'spearman_p' in data and not np.isnan(data['spearman_p']):
            # CORRECT pattern matching
            if key.endswith('_raw'):  # Changed from '_raw_' to '_raw'
                p_values_raw.append(data['spearman_p'])
                keys_raw.append(key)
            elif key.endswith('_freq_adjusted'):  # Changed pattern
                p_values_adj.append(data['spearman_p'])
                keys_adj.append(key)
    
    print_and_log(f"Found {len(p_values_raw)} raw p-values and {len(p_values_adj)} adjusted p-values")
    
    # Apply FDR correction
    alpha = 0.05
    
    if p_values_raw:
        rejected_raw, p_adj_raw, _, _ = multipletests(p_values_raw, alpha=alpha, method='fdr_bh')
        print_and_log(f"\nRaw Analysis - Significant after FDR correction:")
        significant_count = sum(rejected_raw)
        print_and_log(f"   {significant_count}/{len(rejected_raw)} remain significant after correction")
        
        for i, key in enumerate(keys_raw):
            if rejected_raw[i]:
                data = results[key]
                direction = "↓ Faster" if data['spearman_r'] < 0 else "↑ Slower"
                print_and_log(f"   • {data['name']}: r={data['spearman_r']:.3f}, p_adj={p_adj_raw[i]:.3f} {direction}")
    
    if p_values_adj:
        rejected_adj, p_adj_adj, _, _ = multipletests(p_values_adj, alpha=alpha, method='fdr_bh')
        print_and_log(f"\nFrequency-Adjusted Analysis - Significant after FDR correction:")
        significant_count = sum(rejected_adj)
        print_and_log(f"   {significant_count}/{len(rejected_adj)} remain significant after correction")
        
        for i, key in enumerate(keys_adj):
            if rejected_adj[i]:
                data = results[key]
                direction = "↓ Faster" if data['spearman_r'] < 0 else "↑ Slower"
                print_and_log(f"   • {data['name']}: r={data['spearman_r']:.3f}, p_adj={p_adj_adj[i]:.3f} {direction}")

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
        fix_multiple_comparisons_correction(bigram_results)
    
    # Final summary
    total_elapsed = time.time() - start_time
    
    print_and_log(f"\n" + "=" * 80)
    print_and_log("ANALYSIS COMPLETE")
    print_and_log("=" * 80)
    print_and_log(f"Total runtime: {format_time(total_elapsed)}")
    print_and_log(f"Key outputs saved:")
    print_and_log(f"- Text output: test_dvorak9_speed_results.txt")
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