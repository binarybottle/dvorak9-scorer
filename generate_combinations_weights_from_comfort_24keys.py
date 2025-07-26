#!/usr/bin/env python3
"""
Generate Dvorak-9 empirical weights based on comfort scores for 24 keys (home blocks).

This adapts the speed-based analysis to use subjective comfort ratings for 
position pairs. Higher comfort scores = better layouts.

Key differences from speed analysis:
- Uses comfort scores instead of typing times
- Positive correlation = good (higher comfort)
- No frequency adjustment needed (comfort is independent of linguistic frequency)
- May include uncertainty weighting

Usage:
    python generate_combinations_weights_from_comfort.py
    python generate_combinations_weights_from_comfort.py --min-uncertainty 0.1
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from scipy.stats import spearmanr
from scipy import stats
from collections import defaultdict, Counter
from itertools import combinations
import argparse
from statsmodels.stats.multitest import multipletests

# Import the canonical scoring function
from dvorak9_scorer import score_bigram_dvorak9

def load_comfort_data(comfort_file="input/estimated_bigram_scores_24keys.csv"):
    """Load comfort scores for position pairs"""
    print(f"Loading comfort data from {comfort_file}...")
    
    try:
        df = pd.read_csv(comfort_file)
        print(f"✅ Loaded {len(df)} position pairs with comfort scores")
        print(f"   Columns: {list(df.columns)}")
        
        # Show sample data
        print("   Sample comfort scores:")
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            print(f"     '{row['position_pair']}': {row['score']:.3f} (±{row['uncertainty']:.3f})")
        
        # Data quality checks
        print(f"   Comfort score range: {df['score'].min():.3f} to {df['score'].max():.3f}")
        print(f"   Average uncertainty: {df['uncertainty'].mean():.3f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading comfort data: {e}")
        return None

def filter_by_uncertainty(df, min_uncertainty=None, max_uncertainty=None):
    """Filter comfort data by uncertainty thresholds"""
    if min_uncertainty is None and max_uncertainty is None:
        return df
    
    original_count = len(df)
    
    if min_uncertainty is not None:
        df = df[df['uncertainty'] >= min_uncertainty]
    
    if max_uncertainty is not None:
        df = df[df['uncertainty'] <= max_uncertainty]
    
    filtered_count = len(df)
    removed_count = original_count - filtered_count
    
    print(f"Filtered {removed_count}/{original_count} pairs by uncertainty")
    print(f"  Kept {filtered_count} pairs ({filtered_count/original_count*100:.1f}%)")
    
    return df

def get_hand_for_position(pos):
    """Get hand (L/R) for a QWERTY position"""
    left_positions = {'Q', 'W', 'E', 'R', 'T', 'A', 'S', 'D', 'F', 'G', 'Z', 'X', 'C', 'V', 'B'}
    right_positions = {'Y', 'U', 'I', 'O', 'P', 'H', 'J', 'K', 'L', ';', 'N', 'M', ',', '.', '/'}
    
    if pos in left_positions:
        return 'L'
    elif pos in right_positions:
        return 'R'
    else:
        return None

def analyze_comfort_correlations(comfort_df):
    """Analyze correlations between Dvorak criteria and comfort scores"""
    
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
    
    print(f"\nAnalyzing comfort correlations for {len(comfort_df)} position pairs...")
    print("Important constraints:")
    print("  • Only 24 keys (home row + adjacent rows, no middle columns)")
    print("  • Different hands assumed to have comfort score = 1.0")
    print("  • Only same-hand bigrams have variable comfort scores")
    
    # Convert position pairs to sequences for Dvorak scoring
    sequences = []
    comfort_scores = []
    uncertainties = []
    same_hand_sequences = []
    same_hand_comfort = []
    
    different_hand_count = 0
    same_hand_count = 0
    
    for _, row in comfort_df.iterrows():
        pos_pair = row['position_pair']
        
        if len(pos_pair) == 2:
            pos1, pos2 = pos_pair[0], pos_pair[1]
            
            # Check if different hands (if so, comfort = 1.0)
            hand1 = get_hand_for_position(pos1)
            hand2 = get_hand_for_position(pos2)
            
            if hand1 and hand2:
                if hand1 != hand2:
                    # Different hands: comfort = 1.0 (maximum)
                    sequences.append(pos_pair.lower())
                    comfort_scores.append(1.0)  # Maximum comfort for different hands
                    uncertainties.append(0.0)   # No uncertainty for this assumption
                    different_hand_count += 1
                else:
                    # Same hand: use actual comfort score
                    sequences.append(pos_pair.lower())
                    comfort_scores.append(row['score'])
                    uncertainties.append(row['uncertainty'])
                    
                    # Also store for same-hand only analysis
                    same_hand_sequences.append(pos_pair.lower())
                    same_hand_comfort.append(row['score'])
                    same_hand_count += 1
    
    print(f"✅ Processed {len(sequences)} total bigrams:")
    print(f"    • Different hands (comfort = 1.0): {different_hand_count}")
    print(f"    • Same hand (variable comfort): {same_hand_count}")
    print(f"✅ Same-hand subset: {len(same_hand_sequences)} bigrams for correlation analysis")
    
    # Calculate Dvorak scores for ALL sequences (including different hands)
    print("\nCalculating Dvorak criterion scores for ALL sequences...")
    
    all_criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    all_valid_indices = []
    
    for i, seq in enumerate(sequences):
        # Calculate Dvorak scores using canonical function
        scores = score_bigram_dvorak9(seq)
        
        # Validate scores
        if all(isinstance(score, (int, float)) and not np.isnan(score) for score in scores.values()):
            all_valid_indices.append(i)
            for criterion in criteria_names.keys():
                all_criterion_scores[criterion].append(scores[criterion])
    
    all_valid_sequences = [sequences[i] for i in all_valid_indices]
    all_valid_comfort = [comfort_scores[i] for i in all_valid_indices]
    all_valid_uncertainty = [uncertainties[i] for i in all_valid_indices]
    
    print(f"✅ Valid sequences (ALL): {len(all_valid_sequences)}")
    
    # Calculate Dvorak scores for SAME-HAND sequences only
    print("Calculating Dvorak criterion scores for SAME-HAND sequences only...")
    
    same_criterion_scores = {criterion: [] for criterion in criteria_names.keys()}
    same_valid_indices = []
    
    for i, seq in enumerate(same_hand_sequences):
        scores = score_bigram_dvorak9(seq)
        
        if all(isinstance(score, (int, float)) and not np.isnan(score) for score in scores.values()):
            same_valid_indices.append(i)
            for criterion in criteria_names.keys():
                same_criterion_scores[criterion].append(scores[criterion])
    
    same_valid_sequences = [same_hand_sequences[i] for i in same_valid_indices]
    same_valid_comfort = [same_hand_comfort[i] for i in same_valid_indices]
    
    print(f"✅ Valid sequences (SAME-HAND only): {len(same_valid_sequences)}")
    
    # Calculate correlations for ALL sequences
    print("\n📊 ANALYSIS 1: ALL SEQUENCES (including different hands = 1.0)")
    results = {}
    
    for criterion, scores_list in all_criterion_scores.items():
        if len(scores_list) >= 3:
            try:
                # Check for constant values (will cause NaN correlation)
                unique_scores = len(set(scores_list))
                unique_comfort = len(set(all_valid_comfort))
                
                if unique_scores <= 1:
                    print(f"    {criterion}: constant scores (all {scores_list[0]:.3f})")
                    results[f"{criterion}_all"] = {
                        'name': f"{criteria_names[criterion]} (all sequences)",
                        'correlation': float('nan'),
                        'p_value': float('nan'),
                        'abs_correlation': float('nan'),
                        'n_samples': len(scores_list),
                        'supports_dvorak': None,
                        'constant_scores': True,
                        'analysis_type': 'all_sequences'
                    }
                elif unique_comfort <= 1:
                    print(f"    {criterion}: constant comfort scores")
                    results[f"{criterion}_all"] = {
                        'name': f"{criteria_names[criterion]} (all sequences)",
                        'correlation': float('nan'),
                        'p_value': float('nan'),
                        'abs_correlation': float('nan'),
                        'n_samples': len(scores_list),
                        'supports_dvorak': None,
                        'constant_comfort': True,
                        'analysis_type': 'all_sequences'
                    }
                else:
                    # Spearman correlation (rank-based, robust)
                    spearman_r, spearman_p = spearmanr(scores_list, all_valid_comfort)
                    
                    # Note: For comfort, POSITIVE correlation = good (higher Dvorak score = more comfortable)
                    results[f"{criterion}_all"] = {
                        'name': f"{criteria_names[criterion]} (all sequences)",
                        'correlation': spearman_r,
                        'p_value': spearman_p,
                        'abs_correlation': abs(spearman_r),
                        'n_samples': len(scores_list),
                        'supports_dvorak': spearman_r > 0,  # Positive = supports Dvorak for comfort
                        'scores': scores_list.copy(),
                        'comfort_scores': all_valid_comfort.copy(),
                        'analysis_type': 'all_sequences'
                    }
                    
                    print(f"    {criterion}: r = {spearman_r:.3f}, p = {spearman_p:.3f}")
                
            except Exception as e:
                print(f"    Error calculating correlation for {criterion}: {e}")
                continue
    
    # Calculate correlations for SAME-HAND sequences only
    print("\n📊 ANALYSIS 2: SAME-HAND SEQUENCES ONLY (variable comfort)")
    
    for criterion, scores_list in same_criterion_scores.items():
        if len(scores_list) >= 3:
            try:
                unique_scores = len(set(scores_list))
                unique_comfort = len(set(same_valid_comfort))
                
                if unique_scores <= 1:
                    print(f"    {criterion}: constant scores (same-hand)")
                    results[f"{criterion}_same_hand"] = {
                        'name': f"{criteria_names[criterion]} (same-hand only)",
                        'correlation': float('nan'),
                        'p_value': float('nan'),
                        'abs_correlation': float('nan'),
                        'n_samples': len(scores_list),
                        'supports_dvorak': None,
                        'constant_scores': True,
                        'analysis_type': 'same_hand_only'
                    }
                elif unique_comfort <= 1:
                    print(f"    {criterion}: constant comfort scores (same-hand)")
                    results[f"{criterion}_same_hand"] = {
                        'name': f"{criteria_names[criterion]} (same-hand only)",
                        'correlation': float('nan'),
                        'p_value': float('nan'),
                        'abs_correlation': float('nan'),
                        'n_samples': len(scores_list),
                        'supports_dvorak': None,
                        'constant_comfort': True,
                        'analysis_type': 'same_hand_only'
                    }
                else:
                    spearman_r, spearman_p = spearmanr(scores_list, same_valid_comfort)
                    
                    results[f"{criterion}_same_hand"] = {
                        'name': f"{criteria_names[criterion]} (same-hand only)",
                        'correlation': spearman_r,
                        'p_value': spearman_p,
                        'abs_correlation': abs(spearman_r),
                        'n_samples': len(scores_list),
                        'supports_dvorak': spearman_r > 0,
                        'scores': scores_list.copy(),
                        'comfort_scores': same_valid_comfort.copy(),
                        'analysis_type': 'same_hand_only'
                    }
                    
                    print(f"    {criterion}: r = {spearman_r:.3f}, p = {spearman_p:.3f}")
                
            except Exception as e:
                print(f"    Error calculating correlation for {criterion} (same-hand): {e}")
                continue
    
    # Store sequence data for combination analysis (use ALL sequences for now)
    all_sequence_data = []
    for i, seq in enumerate(all_valid_sequences):
        record = {
            'sequence': seq,
            'comfort_score': all_valid_comfort[i],
            'uncertainty': all_valid_uncertainty[i],
            'analysis_type': 'all_sequences'
        }
        
        # Add criterion scores
        for criterion in criteria_names.keys():
            if i < len(all_criterion_scores[criterion]):
                record[criterion] = all_criterion_scores[criterion][i]
        
        all_sequence_data.append(record)
    
    # Store same-hand sequence data for combination analysis
    same_hand_sequence_data = []
    for i, seq in enumerate(same_valid_sequences):
        record = {
            'sequence': seq,
            'comfort_score': same_valid_comfort[i],
            'uncertainty': 0.0,  # Will be filled from original data if needed
            'analysis_type': 'same_hand_only'
        }
        
        # Add criterion scores
        for criterion in criteria_names.keys():
            if i < len(same_criterion_scores[criterion]):
                record[criterion] = same_criterion_scores[criterion][i]
        
        same_hand_sequence_data.append(record)
    
    # Store both datasets for analysis
    results['_all_sequence_data'] = all_sequence_data
    results['_same_hand_sequence_data'] = same_hand_sequence_data
    
    print(f"\n✅ Data summary:")
    print(f"    • All sequences dataset: {len(all_sequence_data)} bigrams")
    print(f"    • Same-hand only dataset: {len(same_hand_sequence_data)} bigrams")
    print(f"    • Different hands assumed comfort = 1.0")
    
    return results

def analyze_comfort_combinations(all_sequence_data, same_hand_data):
    """Analyze combinations of criteria for comfort prediction"""
    
    print(f"\n" + "=" * 80)
    print("CRITERION COMBINATION ANALYSIS")
    print("=" * 80)
    print("Testing combinations on both datasets:")
    print(f"  1. ALL sequences ({len(all_sequence_data)} bigrams) - includes different hands = 1.0")
    print(f"  2. SAME-HAND only ({len(same_hand_data)} bigrams) - variable comfort scores only")
    
    results = {}
    
    # Analyze ALL sequences first
    print(f"\n📊 ANALYSIS 1: ALL SEQUENCES")
    print("-" * 50)
    all_results = analyze_single_dataset_combinations(all_sequence_data, "all_sequences")
    results.update({f"all_{k}": v for k, v in all_results.items()})
    
    # Analyze SAME-HAND sequences
    print(f"\n📊 ANALYSIS 2: SAME-HAND SEQUENCES ONLY")
    print("-" * 50)
    same_results = analyze_single_dataset_combinations(same_hand_data, "same_hand_only")
    results.update({f"same_{k}": v for k, v in same_results.items()})
    
    return results

def analyze_single_dataset_combinations(sequence_data, dataset_name):
    """Analyze combinations for a single dataset"""
    
    if len(sequence_data) < 10:
        print(f"❌ Too few sequences for {dataset_name} analysis ({len(sequence_data)})")
        return {}
    
    print(f"Analyzing {dataset_name}: {len(sequence_data)} sequences")
    
    # Convert to DataFrame
    df = pd.DataFrame(sequence_data)
    
    # Get criteria columns
    exclude_cols = {'sequence', 'comfort_score', 'uncertainty', 'analysis_type'}
    criteria_cols = [col for col in df.columns if col not in exclude_cols]
    
    comfort_scores = df['comfort_score'].values
    
    # Check for variation in comfort scores
    unique_comfort = len(set(comfort_scores))
    if unique_comfort <= 1:
        print(f"⚠️  No variation in comfort scores for {dataset_name} (all = {comfort_scores[0]:.3f})")
        print("Cannot calculate meaningful correlations")
        return {}
    
    print(f"Comfort score range: {min(comfort_scores):.3f} to {max(comfort_scores):.3f}")
    print(f"Testing all combinations of {len(criteria_cols)} criteria...")
    
    # Test all combinations
    all_results = {}
    
    for k in range(1, len(criteria_cols) + 1):
        print(f"\n  {k}-way combinations:")
        
        combos = list(combinations(criteria_cols, k))
        combo_results = []
        
        for combo in combos:
            # Create combined score (additive model)
            combined_scores = np.zeros(len(comfort_scores))
            for criterion in combo:
                combined_scores += df[criterion].values
            
            # Test correlation with comfort
            if len(set(combined_scores)) > 1:
                try:
                    corr, p_val = spearmanr(combined_scores, comfort_scores)
                    if not (np.isnan(corr) or np.isnan(p_val)):
                        combo_results.append({
                            'combination': ' + '.join(combo),
                            'criteria_count': k,
                            'correlation': corr,
                            'p_value': p_val,
                            'abs_correlation': abs(corr),
                            'supports_dvorak': corr > 0,  # Positive = supports for comfort
                            'dataset': dataset_name
                        })
                except:
                    continue
        
        # Sort by absolute correlation
        combo_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
        all_results[f'{k}_way'] = combo_results
        
        # Show top results
        if combo_results:
            print(f"     Top 3 combinations:")
            for i, result in enumerate(combo_results[:3]):
                sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                direction = "supports" if result['supports_dvorak'] else "contradicts"
                print(f"       {i+1}. {result['combination']}")
                print(f"          r = {result['correlation']:.4f}{sig}, {direction} Dvorak")
        else:
            print(f"     No valid combinations found")
    
    return all_results

def apply_fdr_correction(individual_results, combination_results):
    """Apply FDR correction to all results"""
    
    print(f"\n" + "=" * 80)
    print("COMFORT-BASED FDR ANALYSIS")
    print("=" * 80)
    print("Analyzing both datasets:")
    print("  1. ALL sequences (including different hands = 1.0)")
    print("  2. SAME-HAND sequences only (variable comfort)")
    
    # PART 1: Individual criteria - analyze both datasets
    print(f"\n📊 INDIVIDUAL CRITERIA ANALYSIS")
    print("=" * 60)
    
    # Separate individual results by analysis type
    all_individual = []
    same_hand_individual = []
    
    for key, data in individual_results.items():
        if key.startswith('_'):
            continue
        if isinstance(data, dict) and 'correlation' in data:
            if key.endswith('_all'):
                all_individual.append((key, data))
            elif key.endswith('_same_hand'):
                same_hand_individual.append((key, data))
    
    # Analyze ALL sequences individual results
    print(f"\n1. ALL SEQUENCES ({len(all_individual)} criteria):")
    print("   (Including different hands = 1.0 comfort)")
    
    if all_individual:
        all_p_values = [data['p_value'] for _, data in all_individual if not np.isnan(data['p_value'])]
        valid_all_individual = [(key, data) for key, data in all_individual if not np.isnan(data['p_value'])]
        
        if all_p_values:
            rejected, p_adj, _, _ = multipletests(all_p_values, alpha=0.05, method='fdr_bh')
            
            print("Criterion              r      p-val    FDR p-val  Significant  Dvorak")
            print("-" * 75)
            
            for i, (key, data) in enumerate(valid_all_individual):
                sig_marker = "✅" if rejected[i] else "❌"
                dvorak_marker = "✅ Support" if data.get('supports_dvorak') else "❌ Contradict"
                
                print(f"{data['name'][:18]:<18} {data['correlation']:>6.3f}  "
                      f"{data['p_value']:>6.3f}  {p_adj[i]:>8.3f}  {sig_marker:<11}  {dvorak_marker}")
        else:
            print("   No valid correlations for ALL sequences analysis")
    
    # Analyze SAME-HAND sequences individual results  
    print(f"\n2. SAME-HAND SEQUENCES ONLY ({len(same_hand_individual)} criteria):")
    print("   (Variable comfort scores only)")
    
    significant_same_hand = []
    if same_hand_individual:
        same_p_values = [data['p_value'] for _, data in same_hand_individual if not np.isnan(data['p_value'])]
        valid_same_individual = [(key, data) for key, data in same_hand_individual if not np.isnan(data['p_value'])]
        
        if same_p_values:
            rejected, p_adj, _, _ = multipletests(same_p_values, alpha=0.05, method='fdr_bh')
            
            print("Criterion              r      p-val    FDR p-val  Significant  Dvorak")
            print("-" * 75)
            
            for i, (key, data) in enumerate(valid_same_individual):
                sig_marker = "✅" if rejected[i] else "❌"
                dvorak_marker = "✅ Support" if data.get('supports_dvorak') else "❌ Contradict"
                
                if rejected[i]:
                    significant_same_hand.append(data)
                
                print(f"{data['name'][:18]:<18} {data['correlation']:>6.3f}  "
                      f"{data['p_value']:>6.3f}  {p_adj[i]:>8.3f}  {sig_marker:<11}  {dvorak_marker}")
        else:
            print("   No valid correlations for SAME-HAND analysis")
    
    # PART 2: Combination analysis for SAME-HAND sequences (most meaningful)
    print(f"\n📊 COMBINATION ANALYSIS (SAME-HAND SEQUENCES)")
    print("=" * 60)
    print("Using same-hand data for meaningful comfort variation")
    
    # Extract same-hand combination results
    same_hand_combinations = []
    for key, results_list in combination_results.items():
        if key.startswith('same_') and results_list:
            for result in results_list:
                same_hand_combinations.append({
                    'combination': result['combination'],
                    'k_way': result['criteria_count'],
                    'correlation': result['correlation'],
                    'p_value': result['p_value'],
                    'abs_correlation': result['abs_correlation'],
                    'supports_dvorak': result['supports_dvorak'],
                    'dataset': 'same_hand_only'
                })
    
    significant_combinations = []
    if same_hand_combinations:
        p_values = [r['p_value'] for r in same_hand_combinations]
        rejected, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        # Add FDR results
        for i, result in enumerate(same_hand_combinations):
            result['p_fdr_corrected'] = p_adj[i]
            result['significant_after_fdr'] = rejected[i]
        
        # Save all results
        df = pd.DataFrame(same_hand_combinations)
        df.to_csv('output/combinations_weights_from_comfort_24keys.csv', index=False)
        print(f"💾 ALL COMBINATIONS SAVED TO: output/combinations_weights_from_comfort_24keys.csv")
        
        # Filter significant
        significant_combinations = [r for r in same_hand_combinations if r['significant_after_fdr']]
        significant_combinations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        print(f"Significant after FDR: {len(significant_combinations)}/{len(same_hand_combinations)} "
              f"({len(significant_combinations)/len(same_hand_combinations)*100:.1f}%)")
        
        if significant_combinations:
            print(f"\nTop 10 significant combinations (same-hand only):")
            print("K  Combination                           r       FDR p-val  Dvorak")
            print("-" * 75)
            
            for i, result in enumerate(significant_combinations[:10]):
                dvorak_marker = "✅" if result['supports_dvorak'] else "❌"
                combo_short = result['combination'][:35] + "..." if len(result['combination']) > 35 else result['combination']
                print(f"{result['k_way']}  {combo_short:<35} {result['correlation']:>7.3f}  "
                      f"{result['p_fdr_corrected']:>8.3f}  {dvorak_marker}")
            
            # Save significant results
            sig_df = pd.DataFrame(significant_combinations)
            sig_df.to_csv('output/combinations_weights_from_comfort_significant_24keys.csv', index=False)
            print(f"💾 SIGNIFICANT COMBINATIONS SAVED TO: output/combinations_weights_from_comfort_significant_24keys.csv")
        else:
            print("❌ No combinations survived FDR correction!")
    else:
        print("❌ No combination results found for same-hand analysis!")
    
    # SUMMARY
    print(f"\n📊 SUMMARY")
    print("=" * 40)
    print(f"Key findings:")
    if significant_same_hand:
        print(f"  • Individual criteria (same-hand): {len(significant_same_hand)} significant")
        best_individual = max(significant_same_hand, key=lambda x: x['abs_correlation'])
        print(f"    Best: {best_individual['name']} (r = {best_individual['correlation']:.3f})")
    
    if significant_combinations:
        print(f"  • Combinations (same-hand): {len(significant_combinations)} significant")
        best_combo = significant_combinations[0]
        print(f"    Best: {best_combo['combination'][:40]}...")
        print(f"          r = {best_combo['correlation']:.3f}")
        print(f"          Uses {best_combo['k_way']} criteria")
    
    print(f"  • Same-hand analysis focuses on meaningful comfort variation")
    print(f"  • Different hands assumed maximum comfort (score = 1.0)")
    
    return significant_combinations

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Generate Dvorak-9 weights based on comfort scores')
    parser.add_argument('--comfort-file', default='input/estimated_bigram_scores_24keys.csv',
                       help='Path to comfort scores CSV file')
    parser.add_argument('--min-uncertainty', type=float,
                       help='Minimum uncertainty threshold for filtering')
    parser.add_argument('--max-uncertainty', type=float,
                       help='Maximum uncertainty threshold for filtering')
    args = parser.parse_args()
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    print("Dvorak-9 Comfort-Based Weight Generation")
    print("=" * 50)
    print(f"Input file: {args.comfort_file}")
    if args.min_uncertainty:
        print(f"Min uncertainty: {args.min_uncertainty}")
    if args.max_uncertainty:
        print(f"Max uncertainty: {args.max_uncertainty}")
    print()
    
    # Load comfort data
    comfort_df = load_comfort_data(args.comfort_file)
    if comfort_df is None:
        return
    
    # Filter by uncertainty if specified
    if args.min_uncertainty or args.max_uncertainty:
        comfort_df = filter_by_uncertainty(comfort_df, args.min_uncertainty, args.max_uncertainty)
    
    # Analyze individual criteria correlations
    individual_results = analyze_comfort_correlations(comfort_df)
    
    # Analyze combinations
    all_sequence_data = individual_results.get('_all_sequence_data', [])
    same_hand_data = individual_results.get('_same_hand_sequence_data', [])
    combination_results = analyze_comfort_combinations(all_sequence_data, same_hand_data)
    
    # Apply FDR correction
    significant_combinations = apply_fdr_correction(individual_results, combination_results)
    
    print(f"\n" + "=" * 50)
    print("COMFORT ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"✅ Generated comfort-based weights for Dvorak-9 scoring")
    print(f"✅ Use 'output/combinations_weights_from_comfort_significant_24keys.csv' with dvorak9_scorer.py")
    print(f"✅ Interpretation: Positive correlation = supports Dvorak (more comfortable)")

if __name__ == "__main__":
    main()