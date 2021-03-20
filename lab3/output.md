| Experiement | Accuracy | Confusion Matrix | Comment |
|-------------|----------|------------------|---------|
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |
| Solution 1   | 0.7760416666666666  | [[115  15] [ 28  34]] |  Features selected were ['bmi', 'glucose','age'] as they had the highest correlation to the label (0.29, 0.47, 0.24) |
| Solution 2   | 0.7760416666666666  | [[116  14] [ 29  33]] |  Features selected were ['bmi', 'glucose'] as they had a higher correlation to the label as compared to age. This is visible in a better confusion matrix (more correctly labelled values) |
| Solution 3   | 0.7864583333333334  | [[117  13]  [ 28  34]] |  The feature set this time was selected as ['bmi', 'glucose', 'age', 'pregnant']. after the previous 3, pregnant had the highest correlation (0.22).|
| Solution 4    | 0.7916666666666666 | [[115  15] [ 25  37]] | The final feature set had all the columns (except label) as the feature input. here we see a higher accuracy but lower True positive values and false negative and an increase in true negatives and false positives.|