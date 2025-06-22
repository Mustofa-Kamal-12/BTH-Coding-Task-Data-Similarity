# Dataset Generator with Similarity

This project demonstrates how to analyze an existing dataset, extract its key characteristics, and generate new similar samples with comprehensive verification methods.

## Problem Statement

Given a synthetic dataset with mixed data types (categorical and numerical), the goal is to:
1. Analyze the original dataset characteristics
2. Generate new, similar samples without using the original sampling parameters
3. Verify that the new samples maintain similar statistical properties
4. Scale up the dataset size while preserving characteristics

## Original Dataset Characteristics

The provided dataset contains:
- **500 samples**
- **Category1**: Categorical variable with 5 levels (A, B, C, D, E)
  - Probabilities: A=0.2, B=0.4, C=0.2, D=0.1, E=0.1
- **Value1**: Continuous variable from normal distribution (μ=10, σ=2)
- **Value2**: Continuous variable from normal distribution (μ=20, σ=6)

## Solution Approach

### 1. Dataset Analysis
- Extract statistical properties of each variable
- Identify distribution types (categorical probabilities, normal parameters)
- Detect correlations between variables
- Store characteristics for replication

### 2. Similar Dataset Generation
- Use extracted characteristics instead of original parameters
- Generate larger dataset (1000 samples vs original 500)
- Use different random seed to ensure independence
- Maintain the same statistical properties

### 3. Similarity Verification
Multiple statistical tests to ensure similarity:

**For Categorical Variables:**
- Visual comparison of category distributions
- Chi-square test for distribution similarity

**For Numerical Variables:**
- Descriptive statistics comparison (mean, std, etc.)
- Two-sample t-test for mean similarity
- Levene's test for variance equality
- Kolmogorov-Smirnov test for distribution similarity

## Usage

```python
from dataset_generator import DatasetAnalyzer
import pandas as pd

# Load your original dataset
original_df = pd.read_csv("your_dataset.csv")

# Create analyzer
analyzer = DatasetAnalyzer(original_df)

# Generate similar dataset
new_df = analyzer.generate_similar_dataset(num_samples=1000, random_seed=123)

# Verify similarity
verification_results = analyzer.verify_similarity(new_df, plot=True)
```

## Running the Code

```bash
python dataset_generator.py
```

This will:
1. Generate the original dataset
2. Analyze its characteristics
3. Create a new similar dataset (1000 samples)
4. Verify similarity with statistical tests
5. Save both datasets as CSV files

## Output Files

- `original_dataset.csv`: The original 500-sample dataset
- `new_similar_dataset.csv`: The new 1000-sample similar dataset

## Verification Results Interpretation

**P-values > 0.05**: Indicates similarity (null hypothesis: distributions are the same)
**P-values < 0.05**: Indicates significant difference

For this use case, we want HIGH p-values as they indicate the new dataset successfully replicates the original characteristics.

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```
