# Dvorak-9 scorer: Theory, implementation, and empirical validation
A comprehensive implementation and validation of August Dvorak's 9 evaluation criteria from his 1936 work "Typewriter Keyboard" and patent. This project bridges theoretical keyboard design principles with modern empirical analysis using real typing speed data and subjective comfort ratings.

**Repository**: https://github.com/binarybottle/dvorak9-scorer.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

## Dvorak's 9 evaluation criteria (0-1 scale, higher = better):
1. Hands - favor alternating hands over same hand sequences
2. Fingers - avoid same finger repetition  
3. Skip fingers - favor non-adjacent fingers over adjacent
4. Don't cross home - avoid crossing over the home row
5. Same row - favor typing within the same row
6. Home row - favor using the home row
7. Columns - favor fingers staying in their designated columns
8. Strum - favor inward rolls over outward rolls
9. Strong fingers - favor stronger fingers over weaker ones

## Scoring Modes

### Unweighted Scoring (Pure Dvorak Theory)
- **Method**: Simple average of all 9 individual criteria (0-1 scale)
- **Use case**: Testing pure Dvorak theoretical principles
- **Advantages**: No external dependencies, interpretable scores
- **Command**: `--no-weights`

### Weighted Scoring (Empirically Validated)
- **Method**: Combination-specific weights derived from real data
- **Use case**: Practical layout evaluation with empirical validation
- **Advantages**: Accounts for real-world typing patterns and preferences
- **Files required**: Speed or comfort weights CSV

## Required Files (for Weighted Scoring)

### Speed-Based Analysis
- `combinations_weights_from_speed_significant.csv` - Empirical combination weights from speed analysis
- `bigram_times.csv` - Real typing performance data (from [process_3.5M_keystrokes](https://github.com/binarybottle/process_3.5M_keystrokes))
- `letter_pair_frequencies_english.csv` - English bigram frequencies for linguistic control

### Comfort-Based Analysis  
- `estimated_bigram_scores.csv` - Original comfort scores for 24-key dataset
- `estimated_bigram_scores_extended.csv` - Extended comfort scores for 30-key coverage
- `combinations_weights_from_comfort_significant.csv` - Empirical weights from comfort analysis

## Data Sources

### Typing Performance Data
Real typing data from the 136M Keystrokes dataset:
- **Source**: Correctly typed bigrams from correctly typed words only
- **Coverage**: 30 standard typing keys (qwertyuiopasdfghjkl;zxcvbnm,./)
- **Processing**: See [process_3.5M_keystrokes](https://github.com/binarybottle/process_3.5M_keystrokes)
- **Reference**: Dhakal et al. Observations on Typing from 136 Million Keystrokes. CHI 2018.

### Comfort Rating Data
Subjective comfort preferences from typing preference studies:
- **Source**: [typing_preferences_to_comfort_scores](https://github.com/binarybottle/typing_preferences_to_comfort_scores)
- **Original Coverage**: 24 keys (home row + adjacent rows, excluding middle columns)
- **Extended Coverage**: 30 keys (includes middle columns t,g,y,h,b,n via extrapolation)

### Linguistic Frequency Data
English bigram frequencies for frequency adjustment and regression analysis.

## Scripts Overview

### Core Scoring
#### `dvorak9_scorer.py`
**Purpose**: Calculate Dvorak-9 score for keyboard layouts
**Key Features**:
- **Dual scoring modes**: Unweighted (pure theory) and weighted (empirical)
- Supports both speed-based and comfort-based weights
- Filters to letters only (removes punctuation from analysis)
- Respects word boundaries in bigram extraction
- Outputs detailed breakdowns and CSV exports

**Usage**:
```bash
# Unweighted scoring (pure Dvorak theory)
python dvorak9_scorer.py --items "etaoinshrlcu" --positions "FDESGJWXRTYZ" --no-weights --details

# Weighted scoring with speed-based weights
python dvorak9_scorer.py --items "etaoinshrlcu" --positions "FDESGJWXRTYZ" \
  --weights-csv "combinations_weights_from_speed_significant.csv"

# Weighted scoring with comfort-based weights  
python dvorak9_scorer.py --items "etaoinshrlcu" --positions "FDESGJWXRTYZ" \
  --weights-csv "combinations_weights_from_comfort_significant.csv"

# CSV output (unweighted)
python dvorak9_scorer.py --items "etaoinshrlcu" --positions "FDESGJWXRTYZ" --no-weights --csv
```

### Speed-Based Analysis
#### `generate_combinations_weights_from_speed.py`
**Purpose**: Generate empirical weights from typing speed data
**Assumptions**:
- Negative correlation = good (faster typing)
- Includes frequency adjustment (controls for English bigram frequency)
- Tests both middle column and non-middle column bigrams
- Uses 136M+ keystroke dataset with correctly typed bigrams only

**Usage**:
```bash
python generate_combinations_weights_from_speed.py
```

#### `generate_combinations_weights_from_comfort.py`
**Purpose**: Generate empirical weights from comfort preference data
**Assumptions**:
- Positive correlation = good (more comfortable)
- No frequency adjustment needed (comfort independent of linguistic frequency)
- Uses extended 30-key dataset with real + extrapolated comfort scores
- Tests all 511 combinations with FDR correction

**Usage**:
```bash
python generate_combinations_weights_from_comfort.py --comfort-file input/estimated_bigram_scores_extended.csv
```

### Validation and Testing
#### `validate_dvorak9_all_bigrams.py`
**Purpose**: Comprehensive validation testing
- Tests all 841 possible QWERTY bigram combinations
- Validates expected criterion behaviors
- Tests both weighted and unweighted scoring
- Outputs diagnostic CSV files

**Usage**:
```bash
# Test unweighted scoring
python validate_dvorak9_all_bigrams.py

# Test with speed weights
python validate_dvorak9_all_bigrams.py --weights-csv weights/combinations_weights_from_speed_significant.csv

# Test with comfort weights
python validate_dvorak9_all_bigrams.py --weights-csv weights/combinations_weights_from_comfort_significant.csv
```

## Scoring Mode Comparison

| Aspect | Unweighted Scoring | Speed-Based Weights | Comfort-Based Weights |
|--------|-------------------|-------------------|---------------------|
| **Data Source** | Dvorak's theory only | 136M keystroke dataset | Subjective preference ratings |
| **Calculation** | Average of 9 criteria | Empirical combination weights | Empirical combination weights |
| **Dependencies** | None | Speed weights CSV | Comfort weights CSV |
| **Correlation Direction** | N/A (theoretical) | Negative = good (faster) | Positive = good (more comfortable) |
| **Frequency Control** | N/A | Yes (linguistic frequency adjustment) | No (comfort independent of frequency) |
| **Coverage** | All layouts | 30 keys (natural typing data) | 24→30 keys (extended via extrapolation) |
| **Use Case** | Pure theory testing | Performance optimization | Ergonomic optimization |
| **Interpretability** | High (direct Dvorak scores) | Medium (empirically weighted) | Medium (empirically weighted) |

### When to Use Each Mode

- **Unweighted (`--no-weights`)**: 
  - Testing pure Dvorak theoretical principles
  - Educational purposes or research into historical claims
  - When no empirical data is available
  - Quick layout comparisons without external dependencies

- **Speed-based weights**: 
  - Optimizing layouts for typing speed
  - Research comparing performance across layouts
  - When empirical validation against real typing data is needed

- **Comfort-based weights**:
  - Optimizing layouts for user comfort and ergonomics
  - When subjective user preferences are the priority
  - Research into comfort vs. performance trade-offs

## Analysis Approaches

### Extension Methodology
For comfort analysis, missing middle column keys (t,g,y,h,b,n) are assigned the median comfort score from existing "same finger, adjacent row" bigrams, representing similar awkward finger movements.

### Statistical Methods
- **Frequency Adjustment**: `log10(frequency)` regression for speed data
- **Multiple Testing**: Benjamini-Hochberg FDR correction (511 combinations tested)
- **Middle Column Analysis**: Separates analysis by lateral index finger movement requirements
- **Correlation Method**: Spearman rank correlation (robust to outliers)

## Implementation Notes
- **Letter-Only Analysis**: Automatically filters to letters, ignoring punctuation
- **Word Boundary Respect**: Spaces break bigram chains (prevents artificial cross-word bigrams)
- **Empirical Weighting**: Uses combination-specific weights derived from real data
- **Score Interpretation**: Higher scores = better layouts (consistent across all modes)

## Integration Examples

### Basic Unweighted Usage
```python
from dvorak9_scorer import Dvorak9Scorer

# Simple unweighted scoring
layout_mapping = {'e': 'D', 't': 'K', 'a': 'A', 'o': 'S'}
text = "the quick brown fox"

scorer = Dvorak9Scorer(layout_mapping, text, weights_csv=None)
results = scorer.calculate_scores()

print(f"Unweighted score: {results['layout_score']:.3f}")
print(f"Individual scores: {results['individual_scores']}")
```

### Comparing Scoring Modes
```python
from dvorak9_scorer import Dvorak9Scorer

layout_mapping = {'e': 'D', 't': 'K', 'a': 'A', 'o': 'S'}
text = "the quick brown fox"

# Unweighted scoring
unweighted_scorer = Dvorak9Scorer(layout_mapping, text, weights_csv=None)
unweighted_results = unweighted_scorer.calculate_scores()

# Speed-based scoring
speed_scorer = Dvorak9Scorer(layout_mapping, text, "combinations_weights_from_speed_significant.csv")
speed_results = speed_scorer.calculate_scores()

# Comfort-based scoring
comfort_scorer = Dvorak9Scorer(layout_mapping, text, "combinations_weights_from_comfort_significant.csv")
comfort_results = comfort_scorer.calculate_scores()

print(f"Unweighted score: {unweighted_results['layout_score']:.3f}")
print(f"Speed score: {speed_results['layout_score']:.3f}")
print(f"Comfort score: {comfort_results['layout_score']:.3f}")
```

### Command Line Comparison
```bash
# Compare all three scoring modes for the same layout
ITEMS="etaoinshrlcu"; POSITIONS="FDESGJWXRTYZ"

# Unweighted (pure theory)
python dvorak9_scorer.py --items "$ITEMS" --positions "$POSITIONS" --no-weights --ten-scores

# Speed-optimized (empirical)
python dvorak9_scorer.py --items "$ITEMS" --positions "$POSITIONS" --weights-csv weights/combinations_weights_from_speed_significant.csv --ten-scores

# Comfort-optimized (empirical)  
python dvorak9_scorer.py --items "$ITEMS" --positions "$POSITIONS" --weights-csv weights/combinations_weights_from_comfort_significant.csv --ten-scores
```