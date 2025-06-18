# Dvorak-9 scorer: Theory, implementation, and empirical validation
A comprehensive implementation and validation of August Dvorak's 9 evaluation criteria from his 1936 work "Typewriter Keyboard" and patent. This project bridges theoretical keyboard design principles with modern empirical analysis using real typing data.

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

## Required File
- `dvorak9_weights.csv` 
  - Empirical combination weights from statistical analysis. This file should contain 'combination' and 'correlation' columns mapping feature combinations to their empirical correlations with typing speed.

## Data sources

### Typing Performance Data
Real typing data from the 136M Keystrokes dataset:
- **Source**: Correctly typed bigrams from correctly typed words only
- **Processing**: See [process_3.5M_keystrokes](https://github.com/binarybottle/process_3.5M_keystrokes)
- **Reference**: 
Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, Antti Oulasvirta. 
Observations on Typing from 136 Million Keystrokes. 
In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, ACM, 2018.

### Linguistic Frequency Data
English bigram frequencies for frequency adjustment and regression analysis.

## Scripts Overview

### `test_dvorak9_all_bigrams.py`
Purpose: Comprehensive validation testing
- Tests all 841 possible QWERTY bigram combinations
- Validates that each criterion produces expected scores
- Outputs detailed CSV files for analysis
- Includes specific test cases and debugging output

**Usage**:
`python test_dvorak9_all_bigrams.py`

**Outputs**:
dvorak9_scores_all_bigrams.csv - All bigrams with scores
dvorak9_scores_unique_all_bigrams.csv - Unique score patterns

### `test_dvorak9_speed.py`
Purpose: Empirical analysis of criteria vs. actual typing speed
- Correlates Dvorak criteria with real typing performance
- Controls for English bigram frequency effects
- Analyzes middle column key effects (lateral index movements)
- Tests all 511 possible criterion combinations
- Applies rigorous statistical correction (FDR)

**Usage**:
```bash
# Full analysis
python test_dvorak9_speed.py

# Limit to subset for faster testing
python test_dvorak9_speed.py --max-bigrams 100000

# Test the scoring system only
python test_dvorak9_speed.py --test-scorer
```

### `dvorak9_scorer.py`
Purpose: Calculate Dvorak-9 score for some text.

#### Implementation Notes
- **Word Boundary Handling**: Bigram extraction respects word boundaries - spaces and punctuation break bigram chains, preventing artificial cross-word bigrams like "o-w" from "hello world"
- **Empirical Weighting**: Uses combination-specific weights derived from 136M+ keystroke analysis rather than simple averaging
- **Score Interpretation**: Higher scores indicate better layouts (sign-flipped from original correlation data)

**Usage**:
```bash
# Basic layout scoring
python dvorak9_scorer.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --text "sample text"

# Score with detailed breakdown
python dvorak9_scorer.py --items "abc" --positions "FDJ" --details

# Export to CSV
python dvorak9_scorer.py --items "abc" --positions "FDJ" --csv > results.csv

# Get just the 10 numerical scores
python dvorak9_scorer.py --items "abc" --positions "FDJ" --ten-scores
```

```python 
from dvorak9_scorer import Dvorak9Scorer

# Define layout mapping (char -> QWERTY position)
layout_mapping = {
    'e': 'D', 't': 'K', 'a': 'A', 'o': 'S', 'i': 'F',
    'n': 'J', 's': 'R', 'h': 'U', 'r': 'L'
}

# Text to analyze
text = "the quick brown fox"

# Score using empirical weights
scorer = Dvorak9Scorer(layout_mapping, text, weights_csv="dvorak9_weights.csv")
results = scorer.calculate_scores()

print(f"Layout score: {results['average_weighted_score']:.3f}")
```

## Statistical methods

### Frequency Adjustment
Typing speed is heavily influenced by letter frequency. We control for this using:
```python
# Log-transform frequency (psycholinguistic standard)
log_freq = np.log10(frequency + 1)

# Regression model
time = intercept + slope * log_freq + residual

# Frequency-adjusted time = residual
adjusted_time = time - predicted_time
```

### Multiple Testing Correction
With 9 criteria × 2 groups + 511 combinations = 529 tests, we apply Benjamini-Hochberg FDR correction to control false discovery rate.

### Middle Column Analysis
Splits analysis by middle column key usage (b, g, h, n, t, y) to test whether lateral index finger movements affect criterion validity.

### Findings
The empirical analysis reveals which of Dvorak's theoretical principles actually correlate with typing speed in practice, providing evidence-based weights for keyboard evaluation.

