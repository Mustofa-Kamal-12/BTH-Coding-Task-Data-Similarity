#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# In[2]:


class DatasetAnalyzer:
    """Analyzing and generating similar dataset"""
    
    def __init__(self, original_df):
        self.original_df = original_df
        self.characteristics = self._extract_characteristics()
    
    def _extract_characteristics(self):
        """Extracting key characteristics from the original dataset"""
        char = {}
        
        # Categorical variable analysis
        cat_cols = self.original_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            char[col] = {
                'type': 'categorical',
                'categories': list(self.original_df[col].unique()),
                'probabilities': self.original_df[col].value_counts(normalize=True).to_dict()
            }
        
        # Analysing numerical variables
        num_cols = self.original_df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            # normality checking
            _, p_value = stats.normaltest(self.original_df[col])
            is_normal = p_value > 0.05
            
            char[col] = {
                'type': 'numerical',
                'mean': self.original_df[col].mean(),
                'std': self.original_df[col].std(),
                'min': self.original_df[col].min(),
                'max': self.original_df[col].max(),
                'is_normal': is_normal,
                'distribution': 'normal' if is_normal else 'unknown'
            }
        
        # Checking correlations between numerical variables
        if len(num_cols) > 1:
            corr_matrix = self.original_df[num_cols].corr()
            char['correlations'] = corr_matrix.to_dict()
        
        return char
    
    def generate_similar_dataset(self, num_samples=1000, random_seed=None):
        """Generating new dataset with similar characteristics"""
        if random_seed:
            np.random.seed(random_seed)
        
        new_data = {}
        
        # Generating categorical variables
        for col, info in self.characteristics.items():
            if col == 'correlations':
                continue
            if info['type'] == 'categorical':
                categories = info['categories']
                probs = [info['probabilities'].get(cat, 0) for cat in categories]
                new_data[col] = np.random.choice(categories, num_samples, p=probs)
        
        # Generating numerical variables
        for col, info in self.characteristics.items():
            if col == 'correlations':  
                continue
            if info['type'] == 'numerical':
                if info['distribution'] == 'normal':
                    new_data[col] = np.random.normal(info['mean'], info['std'], num_samples)
                else:
                    # Fallback to normal distribution with original parameters
                    new_data[col] = np.random.normal(info['mean'], info['std'], num_samples)
        
        return pd.DataFrame(new_data)
    
    def verify_similarity(self, new_df, plot=True):
        """Verifing new dataset weather is similar to the original"""
        verification_results = {}
        
        #print("Dataset Similarity Verification")
        print(f"a. Original files shape: {self.original_df.shape}")
        print(f"b. New files shape: {new_df.shape}")
        
        # Verifying Categorical Variables
        for col in self.original_df.select_dtypes(include=['object']).columns:
            #print(f"Categorical Variable: {col}")
            
            orig_dist = self.original_df[col].value_counts(normalize=True).sort_index()
            new_dist = new_df[col].value_counts(normalize=True).sort_index()
            
            #print("Original distribution:")
            print(f"c. Original distribution - categorical variable: {col}")
            for cat, prob in orig_dist.items():
                print(f"  {cat}: {prob:.3f}")
            
            print(f"d. New distribution - categorical variable: {col}")
            for cat, prob in new_dist.items():
                print(f"  {cat}: {prob:.3f}")
            
            # Chi-square test for categorical distribution similarity check
            orig_counts = self.original_df[col].value_counts().sort_index()
            new_counts = new_df[col].value_counts().sort_index()
            
            # Aligning categories
            all_cats = sorted(set(orig_counts.index) | set(new_counts.index))
            orig_aligned = [orig_counts.get(cat, 0) for cat in all_cats]
            new_aligned = [new_counts.get(cat, 0) for cat in all_cats]
            
            # Scale new counts to match original total for fair comparison
            scale_factor = sum(orig_aligned) / sum(new_aligned)
            new_aligned_scaled = [count * scale_factor for count in new_aligned]
            
            try:
                chi2, p_val = stats.chisquare(new_aligned_scaled, orig_aligned)
                print(f"e. Chi-square test p-value: {p_val:.4f}")
                verification_results[f"{col}_chi2_pvalue"] = p_val
            except:
                print("Chi-square test failed")
            
            print()
        
        # Verifying numerical variables
        for col in self.original_df.select_dtypes(include=[np.number]).columns:
            print(f"Numerical variable test: {col}")
            #print("-" * 30)
            
            orig_stats = self.original_df[col].describe()
            new_stats = new_df[col].describe()
            
            print(f" Mean - original data: {orig_stats['mean']:.3f}, Std: {orig_stats['std']:.3f}")
            print(f" Mean - new data: {new_stats['mean']:.3f}, Std: {new_stats['std']:.3f}")
            
            # Two-sample t-test for mean similarity
            t_stat, t_p_val = stats.ttest_ind(self.original_df[col], new_df[col])
            print(f" T-test for mean similarity p-value: {t_p_val:.4f}")
            
            # F-test for variance similarity (Levene's test)
            f_stat, f_p_val = stats.levene(self.original_df[col], new_df[col])
            print(f" Levene's test for variance similarity p-value: {f_p_val:.4f}")
            
            # Kolmogorov-Smirnov test for distribution similarity
            ks_stat, ks_p_val = stats.ks_2samp(self.original_df[col], new_df[col])
            print(f" KS test for distribution similarity p-value: {ks_p_val:.4f}")
            
            verification_results[f"{col}_ttest_pvalue"] = t_p_val
            verification_results[f"{col}_levene_pvalue"] = f_p_val
            verification_results[f"{col}_ks_pvalue"] = ks_p_val
            
            print()
        
        if plot:
            self._plot_comparison(new_df)
        
        return verification_results
    
    def _plot_comparison(self, new_df):
        """Creating original and new datasets comparison plots"""
        n_cols = len(self.original_df.columns)
        fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(self.original_df.columns):
            if self.original_df[col].dtype == 'object':
                # Categorical variable
                orig_counts = self.original_df[col].value_counts().sort_index()
                new_counts = new_df[col].value_counts().sort_index()
                
                axes[0, i].bar(orig_counts.index, orig_counts.values, alpha=0.7, label='Original')
                axes[0, i].set_title(f'Original {col} Distribution')
                axes[0, i].set_ylabel('Count')
                
                axes[1, i].bar(new_counts.index, new_counts.values, alpha=0.7, label='New', color='orange')
                axes[1, i].set_title(f'New {col} Distribution')
                axes[1, i].set_ylabel('Count')
                
            else:
                # Numerical variable
                axes[0, i].hist(self.original_df[col], bins=30, alpha=0.7, label='Original', density=True)
                axes[0, i].set_title(f'Original {col} Distribution')
                axes[0, i].set_ylabel('Density')
                
                axes[1, i].hist(new_df[col], bins=30, alpha=0.7, label='New', color='orange', density=True)
                axes[1, i].set_title(f'New {col} Distribution')
                axes[1, i].set_ylabel('Density')
        
        plt.tight_layout()
        plt.show()

def main():
    
    # Step 1: Generate original dataset (as provided)
    print("Step 1 - Generating Original Dataset\n")
    
    np.random.seed(42)
    num_samples = 500
    
    original_df = pd.DataFrame({
        "Category1": np.random.choice(["A", "B", "C", "D", "E"], num_samples, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
        "Value1": np.random.normal(10, 2, num_samples),
        "Value2": np.random.normal(20, 6, num_samples),
    })
    
    print(f"a. Original dataset created with shape: {original_df.shape}")
    print("b. Original dataset preview:")
    print(original_df.head())
    print("c. Original dataset statistics:")
    print(original_df.describe())
    
    # Step 2: Analyze characteristics and generate new dataset
    print("\n\nStep 2 - Analyzing Dataset Characteristics\n")
    
    analyzer = DatasetAnalyzer(original_df)
    
    # print("Extracted characteristics:")
    for col, char in analyzer.characteristics.items():
        if col == 'correlations':
            print(f"Status: Correlations detected between numerical variables")
            continue
        if char['type'] == 'categorical':
            print(f"{col}: {char['type']} type with categories {char['categories']}")
            print(f"Probabilities: {char['probabilities']}")
        elif char['type'] == 'numerical':
            print(f"{col}: {char['type']} (mean={char['mean']:.2f}, std={char['std']:.2f})")
    
    # Step 3: Generate new similar dataset
    print("\n\nStep 3 - Generating New Similar Dataset\n")
    
    # Generate larger dataset without using original seed
    new_df = analyzer.generate_similar_dataset(num_samples=1000, random_seed=123)
    
    print(f"a. New dataset created with shape: {new_df.shape}")
    print("b. New dataset preview:")
    print(new_df.head())
    print("c. New dataset statistics:")
    print(new_df.describe())
    
    # Step 4: Verify similarity
    print("\n\nStep 4 - Verifying Similarity\n")
    
    verification_results = analyzer.verify_similarity(new_df, plot=True)
    
    # Step 5: Save datasets
    print("\n\nStep 5 - Saving Datasets\n")
    
    original_df.to_csv("original_dataset.csv", sep=";", index=False)
    new_df.to_csv("new_similar_dataset.csv", sep=";", index=False)
    
    #print("Datasets saved:")
    print("original_dataset.csv")
    print("new_similar_dataset.csv")
    
    # Summary
    print("\n\nStep 6 - Complete Summary\n")
    # print("=" * 20)
    print("a. Analyzed original dataset characteristics")
    print("b. Generated new dataset with similar properties")
    print("c. Verified similarity using statistical tests")
    print("d. New dataset has twice the samples (1000 vs 500)")
    print("e. Used different random seed to ensure independence")

if __name__ == "__main__":
    main()

