# dvorak9-scoring
Dvorak-9 scoring model: From theoretical foundation to empirical validation.

This script implements August Dvorak's 9 evaluation criteria from his seminal work 
"Typing Behavior" and patent in 1936, enhanced with empirical weighting derived 
from analysis of 50,000+ typing sessions.

## Dvorak's 9 evaluation criteria:
1. Hands - favor alternating hands over same hand sequences
2. Fingers - avoid same finger repetition  
3. Skip fingers - favor non-adjacent fingers over adjacent (same hand)
4. Don't cross home - avoid crossing between non-adjacent rows
5. Same row - favor typing within the same row
6. Home row - favor using the home row
7. Columns - favor fingers staying in their designated columns
8. Strum - favor inward rolls over outward rolls (same hand)
9. Strong fingers - favor stronger fingers over weaker ones

## Data
The analysis is run on correctly typed bigrams from the 136M Keystrokes dataset.
See the processing/filtering script (https://github.com/binarybottle/process_3.5M_keystrokes)
and the original study's article:

Vivek Dhakal, Anna Maria Feit, Per Ola Kristensson, Antti Oulasvirta. 
Observations on Typing from 136 Million Keystrokes. 
In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, ACM, 2018.

## Analysis
The test_dvorak9_speed.py script analyzes whether Dvorak's 9 criteria 
correlate with actual typing performance, splitting the analysis 
by middle column key usage to understand if lateral index finger 
movements affect criterion validity, and regressing out letter 
frequency effects.

### Regressing out letter/bigram frequency
Log-transform frequency (psycholinguistic standard):
`log_freq = np.log10(frequency + 1)`
Regression: time ~ log_frequency:
`model = sm.OLS(time, log_freq).fit()`
Frequency-adjusted time = residual + mean:
`adjusted_time = time - predicted_time + overall_mean`

## Example usage:
python dvorak9_scorer.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details
