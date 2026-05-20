import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/statistical_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """
    Performs statistical analysis on cadmium toxicity study data.
    Includes ANOVA, SNK post-hoc tests, and regression analysis.
    """
    
    def __init__(self, processed_data_dir='data/processed_data', results_dir='results/tables'):
        self.processed_data_dir = Path(processed_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("StatisticalAnalyzer initialized")
    
    def load_data(self, filename):
        """Load processed data."""
        filepath = self.processed_data_dir / filename
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Loaded {filename}: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
    
    def perform_anova(self, data, group_column, value_column):
        """
        Perform one-way ANOVA test.
        
        Parameters:
        -----------
        data : DataFrame
            Data containing groups and values
        group_column : str
            Name of column containing group labels
        value_column : str
            Name of column containing values to analyze
        
        Returns:
        --------
        dict : ANOVA results (F-statistic, p-value)
        """
        try:
            groups = data[group_column].unique()
            group_data = [data[data[group_column] == group][value_column].values for group in groups]
            
            f_stat, p_value = f_oneway(*group_data)
            
            result = {
                'parameter': value_column,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': 'Yes' if p_value < 0.05 else 'No'
            }
            
            logger.info(f"ANOVA {value_column}: F={f_stat:.4f}, p={p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"ANOVA failed for {value_column}: {str(e)}")
            raise
    
    def perform_snk_test(self, data, group_column, value_column):
        """
        Perform Student-Newman-Keuls (SNK) post-hoc test.
        Uses Tukey HSD as an approximation.
        
        Parameters:
        -----------
        data : DataFrame
            Data containing groups and values
        group_column : str
            Name of column containing group labels
        value_column : str
            Name of column containing values
        
        Returns:
        --------
        DataFrame : Pairwise comparison results
        """
        try:
            groups = sorted(data[group_column].unique())
            group_data = [data[data[group_column] == group][value_column].values for group in groups]
            
            # Perform Tukey HSD test
            tukey_result = tukey_hsd(*group_data)
            
            # Create comparison DataFrame
            comparisons = []
            for i, group1 in enumerate(groups):
                for j, group2 in enumerate(groups):
                    if i < j:
                        p_value = tukey_result.pvalue[i, j]
                        comparisons.append({
                            'parameter': value_column,
                            'group1': group1,
                            'group2': group2,
                            'p_value': p_value,
                            'significant': 'Yes' if p_value < 0.05 else 'No'
                        })
            
            logger.info(f"SNK test completed for {value_column}")
            return pd.DataFrame(comparisons)
            
        except Exception as e:
            logger.error(f"SNK test failed for {value_column}: {str(e)}")
            raise
    
    def dose_response_regression(self, data, dose_column, response_column):
        """
        Perform linear regression for dose-response relationship.
        
        Parameters:
        -----------
        data : DataFrame
            Data containing dose and response values
        dose_column : str
            Name of column containing dose values
        response_column : str
            Name of column containing response values
        
        Returns:
        --------
        dict : Regression results (slope, intercept, R², p-value)
        """
        try:
            # Remove NaN values
            clean_data = data[[dose_column, response_column]].dropna()
            
            x = clean_data[dose_column].values
            y = clean_data[response_column].values
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            result = {
                'parameter': response_column,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err,
                'significant': 'Yes' if p_value < 0.05 else 'No'
            }
            
            logger.info(f"Dose-Response {response_column}: R²={r_value**2:.4f}, p={p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Regression failed for {response_column}: {str(e)}")
            raise
    
    def calculate_descriptive_stats(self, data, group_column, value_columns):
        """
        Calculate descriptive statistics by group.
        
        Parameters:
        -----------
        data : DataFrame
            Data to analyze
        group_column : str
            Name of column containing group labels
        value_columns : list
            List of column names to analyze
        
        Returns:
        --------
        DataFrame : Descriptive statistics
        """
        try:
            stats_list = []
            
            for col in value_columns:
                for group in sorted(data[group_column].unique()):
                    group_vals = data[data[group_column] == group][col]
                    
                    stats_list.append({
                        'parameter': col,
                        'group': group,
                        'n': len(group_vals),
                        'mean': group_vals.mean(),
                        'std': group_vals.std(),
                        'min': group_vals.min(),
                        'max': group_vals.max(),
                        'sem': group_vals.sem()
                    })
            
            logger.info("Descriptive statistics calculated")
            return pd.DataFrame(stats_list)
            
        except Exception as e:
            logger.error(f"Descriptive stats calculation failed: {str(e)}")
            raise
    
    def effect_size_cohen_d(self, group1, group2):
        """
        Calculate Cohen's d effect size between two groups.
        
        Parameters:
        -----------
        group1, group2 : array-like
            Data for the two groups
        
        Returns:
        --------
        float : Cohen's d value
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohen_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return cohen_d
    
    def save_results(self, results, filename):
        """Save results to CSV."""
        filepath = self.results_dir / filename
        try:
            if isinstance(results, pd.DataFrame):
                results.to_csv(filepath, index=False)
            else:
                pd.DataFrame([results]).to_csv(filepath, index=False)
            logger.info(f"Saved results: {filepath}")
        except Exception as e:
            logger.error(f"Error saving {filename}: {str(e)}")
            raise
    
    def complete_analysis(self, liver_data, blood_data, metadata):
        """
        Complete statistical analysis pipeline.
        """
        logger.info("Starting complete statistical analysis pipeline")
        
        try:
            # Merge metadata with data
            liver_merged = pd.merge(liver_data, metadata[['MouseID', 'Group']], on='MouseID', how='left')
            blood_merged = pd.merge(blood_data, metadata[['MouseID', 'Group']], on='MouseID', how='left')
            
            # Liver enzyme columns
            liver_enzymes = ['MDA', 'GPx', 'TBARS', 'LP', 'CAT', 'GSH', 'TAS', 'TOS']
            
            # Blood parameters
            blood_params = ['ALT', 'AST', 'Glucose', 'Cortisol']
            
            # --- ANOVA Analysis ---
            logger.info("\n=== ANOVA Analysis ===")
            anova_results = []
            
            for enzyme in liver_enzymes:
                if enzyme in liver_merged.columns:
                    result = self.perform_anova(liver_merged, 'Group', enzyme)
                    anova_results.append(result)
            
            for param in blood_params:
                if param in blood_merged.columns:
                    result = self.perform_anova(blood_merged, 'Group', param)
                    anova_results.append(result)
            
            anova_df = pd.DataFrame(anova_results)
            self.save_results(anova_df, 'anova_results.csv')
            
            # --- SNK Post-hoc Test ---
            logger.info("\n=== SNK Post-hoc Test ===")
            snk_results = []
            
            for enzyme in liver_enzymes:
                if enzyme in liver_merged.columns:
                    snk_df = self.perform_snk_test(liver_merged, 'Group', enzyme)
                    snk_results.append(snk_df)
            
            for param in blood_params:
                if param in blood_merged.columns:
                    snk_df = self.perform_snk_test(blood_merged, 'Group', param)
                    snk_results.append(snk_df)
            
            snk_combined = pd.concat(snk_results, ignore_index=True)
            self.save_results(snk_combined, 'snk_pairwise_comparisons.csv')
            
            # --- Dose-Response Regression ---
            logger.info("\n=== Dose-Response Regression ===")
            dose_map = {'Control': 0, '0.1 mg/kg': 0.1, '1.0 mg/kg': 1.0}
            liver_merged['Dose'] = liver_merged['Group'].map(dose_map)
            
            regression_results = []
            for enzyme in liver_enzymes:
                if enzyme in liver_merged.columns:
                    result = self.dose_response_regression(liver_merged, 'Dose', enzyme)
                    regression_results.append(result)
            
            regression_df = pd.DataFrame(regression_results)
            self.save_results(regression_df, 'dose_response_regression.csv')
            
            # --- Descriptive Statistics ---
            logger.info("\n=== Descriptive Statistics ===")
            liver_stats = self.calculate_descriptive_stats(liver_merged, 'Group', liver_enzymes)
            blood_stats = self.calculate_descriptive_stats(blood_merged, 'Group', blood_params)
            
            self.save_results(liver_stats, 'liver_descriptive_stats.csv')
            self.save_results(blood_stats, 'blood_descriptive_stats.csv')
            
            logger.info("Complete statistical analysis pipeline finished successfully")
            
            return {
                'anova': anova_df,
                'snk': snk_combined,
                'regression': regression_df,
                'liver_stats': liver_stats,
                'blood_stats': blood_stats
            }
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            raise

if __name__ == '__main__':
    analyzer = StatisticalAnalyzer()
    
    # Load preprocessed data
    liver_data = analyzer.load_data('liver_enzymes_processed.csv')
    blood_data = analyzer.load_data('blood_parameters_processed.csv')
    metadata = pd.read_csv('data/raw_data/experimental_metadata.csv')
    
    # Run complete analysis
    results = analyzer.complete_analysis(liver_data, blood_data, metadata)
    print("\n✓ Statistical analysis complete!")
    print(f"Results saved to {analyzer.results_dir}")
