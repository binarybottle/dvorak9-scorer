# dvorak10-scoring
Score and analyze keyboard layouts based on Dvorak-10 criteria and keystroke data.

# Scoring
The score_dvorak10.py script scores keyboard layouts based on
10 evaluation criteria derived from Dvorak's work:

2-HAND CRITERIA (apply to all digraphs):
1. Use both hands equally
2. Alternate between hands  

SAME-HAND CRITERIA (apply only to same-hand digraphs):
3. Don't use the same finger (two fingers)
4. Use non-adjacent fingers (skip fingers)
5. Stay within the home block (home block)
6. Don't skip over the home row (don't skip home)
7. Stay in the same row (same row)
8. Use the home row (include home)
9. Strum inward (roll in, not out)
10. Use strong fingers

# Analysis
The test_dvorak10.py script analyzes whether Dvorak's 10 criteria 
correlate with actual typing performance, splitting the analysis 
by middle column key usage to understand if lateral index finger 
movements affect criterion validity, and regressing out letter 
frequency effects.

## Regressing out letter/bigram frequency
Log-transform frequency (psycholinguistic standard):
`log_freq = np.log10(frequency + 1)`
Regression: time ~ log_frequency:
`model = sm.OLS(time, log_freq).fit()`
Frequency-adjusted time = residual + mean:
`adjusted_time = time - predicted_time + overall_mean`

ANALYSIS APPROACH:
1. Word typing times (criteria 1-2: two-hand criteria)
   - Hand Balance: Equal left/right hand usage within each word
   - Hand Alternation: Alternating hands between consecutive keystrokes
   - Correlates word-level hand usage patterns with word typing speed

2. Bigram interkey intervals (criteria 3-10: same-hand criteria)  
   - Two Fingers: Avoid same finger repetition
   - Skip Fingers: Use non-adjacent fingers  
   - Home Block: Stay within easily accessible keys
   - Don't Skip Home: Avoid jumping over home row
   - Same Row: Type within same keyboard row
   - Include Home: Involve home row keys
   - Roll Inward: Move from outer to inner fingers
   - Strong Fingers: Prefer index/middle over ring/pinky fingers
   - Correlates bigram finger coordination with interkey timing

MIDDLE COLUMN ANALYSIS:
- Splits all sequences into two groups based on middle column key inclusion
- Middle column keys: T, G, B, Y, H, N (require lateral index finger stretches)
- Group 1: Sequences WITHOUT middle column keys (e.g., "as", "df", "kl")  
- Group 2: Sequences WITH middle column keys (e.g., "at", "the", "ng")
- Tests whether Dvorak criteria apply uniformly across keyboard regions

STATISTICAL FEATURES:
- Pearson and Spearman correlations for robustness
- Multiple comparisons correction (FDR) for statistical validity
- Effect size interpretation using Cohen's guidelines
- Filtering for realistic typing speeds (50-2000ms for bigrams)

OUTPUTS:
- Detailed text analysis: dvorak_analysis_results_split.txt
- Visual comparison plots: plots/middle_column_comparison.png
- Group-specific correlation tables and effect sizes
- Statistical significance testing with correction

INTERPRETATION:
- Negative correlation = better criterion score → faster typing (validates Dvorak)
- Positive correlation = better criterion score → slower typing (contradicts Dvorak)
- Compares correlation patterns between middle vs. non-middle column sequences

Usage:
    python test_dvorak10_correlations.py
    
Requirements:
    - bigram_times.csv: bigram,interkey_interval
    - word_times.csv: word,time
    - scipy, numpy, matplotlib, seaborn, sklearn (optional)

# Data
The analysis is run on correctly typed words and bigrams from the 136M Keystrokes dataset.
See the processing/filtering script (https://github.com/binarybottle/process_3.5M_keystrokes)
and the original study's article:

Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, Antti Oulasvirta. Observations on Typing from 136 Million Keystrokes. 
In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, ACM, 2018.
