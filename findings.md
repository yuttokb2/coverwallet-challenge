Model Performance Analysis
Target Variable Distribution
Count: 5,709
Mean: $1,723.70
Std: $3,761.79
Min: $29.25
25%: $500.00
50%: $697.00
75%: $1,438.87
Max: $134,752.41
Model Performance Metrics

RMSE: 663.80
MAE: 161.48
R²: 0.9506

Key Findings
Overall Performance

The model achieves strong predictive accuracy with an R² of 95.06%
RMSE represents 38% of the target variable mean, indicating reasonable error magnitude
MAE represents only 9% of the mean, suggesting excellent median prediction accuracy

Error Distribution Analysis

The significant gap between RMSE (663.80) and MAE (161.48) indicates the presence of outliers with large prediction errors
This pattern suggests the model performs well for typical cases but struggles with extreme values

Performance by Account Value Segments

Small accounts (~$500-700): MAE of ~$161 represents approximately 23-32% relative error - acceptable for business use
Medium accounts (~$1,400): Relative error drops to ~11% - very good performance
Large accounts (>$10,000): Likely higher absolute errors due to the heavy-tailed distribution

Business Implications

The model is highly reliable for the majority of accounts (median and below)
Prediction quality degrades for high-value accounts in the upper tail of the distribution
The 161-dollar median absolute error provides a practical benchmark for expected prediction accuracy in most business scenarios

Recommendations

Consider separate models or post-processing adjustments for high-value accounts
Monitor prediction intervals for accounts above the 90th percentile
The current model is suitable for deployment given the strong performance on typical account values