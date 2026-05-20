import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles data loading, cleaning, and normalization for cadmium toxicity study.
    """
    
    def __init__(self, raw_data_dir='data/raw_data', processed_data_dir='data/processed_data'):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataPreprocessor initialized")
    
    def load_data(self, filename):
        """Load CSV data file."""
        filepath = self.raw_data_dir / filename
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Loaded {filename}: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def check_data_quality(self, data, filename):
        """Perform quality control checks on data."""
        logger.info(f"\n--- Quality Check: {filename} ---")
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
        else:
            logger.info("No missing values detected")
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        else:
            logger.info("No duplicate rows detected")
        
        # Check data types
        logger.info(f"Data types:\n{data.dtypes}")
        
        return data
    
    def normalize_liver_enzymes(self, data):
        """Normalize liver enzyme measurements."""
        logger.info("Normalizing liver enzyme data")
        
        enzyme_columns = ['MDA', 'GPx', 'TBARS', 'LP', 'CAT', 'GSH', 'TAS', 'TOS']
        
        normalized_data = data.copy()
        for col in enzyme_columns:
            if col in normalized_data.columns:
                mean = normalized_data[col].mean()
                std = normalized_data[col].std()
                if std > 0:
                    normalized_data[f'{col}_normalized'] = (normalized_data[col] - mean) / std
                    logger.info(f"Normalized {col}: mean={mean:.2f}, std={std:.2f}")
        
        return normalized_data
    
    def normalize_blood_parameters(self, data):
        """Normalize blood biochemical parameters."""
        logger.info("Normalizing blood parameter data")
        
        param_columns = ['ALT', 'AST', 'Glucose', 'Cortisol']
        
        normalized_data = data.copy()
        for col in param_columns:
            if col in normalized_data.columns:
                mean = normalized_data[col].mean()
                std = normalized_data[col].std()
                if std > 0:
                    normalized_data[f'{col}_normalized'] = (normalized_data[col] - mean) / std
                    logger.info(f"Normalized {col}: mean={mean:.2f}, std={std:.2f}")
        
        return normalized_data
    
    def calculate_descriptive_statistics(self, data):
        """Calculate descriptive statistics."""
        logger.info("\n--- Descriptive Statistics ---")
        stats = data.describe()
        logger.info(f"\n{stats}")
        return stats
    
    def save_processed_data(self, data, filename):
        """Save processed data to CSV."""
        filepath = self.processed_data_dir / filename
        try:
            data.to_csv(filepath, index=False)
            logger.info(f"Saved processed data: {filepath}")
        except Exception as e:
            logger.error(f"Error saving {filename}: {str(e)}")
            raise
    
    def process_all_data(self):
        """Complete data processing pipeline."""
        logger.info("Starting complete data processing pipeline")
        
        try:
            # Load data
            liver_data = self.load_data('liver_enzymes.csv')
            blood_data = self.load_data('blood_parameters.csv')
            metadata = self.load_data('experimental_metadata.csv')
            
            # Quality checks
            liver_data = self.check_data_quality(liver_data, 'liver_enzymes.csv')
            blood_data = self.check_data_quality(blood_data, 'blood_parameters.csv')
            metadata = self.check_data_quality(metadata, 'experimental_metadata.csv')
            
            # Normalize data
            liver_normalized = self.normalize_liver_enzymes(liver_data)
            blood_normalized = self.normalize_blood_parameters(blood_data)
            
            # Calculate statistics
            liver_stats = self.calculate_descriptive_statistics(liver_normalized)
            blood_stats = self.calculate_descriptive_statistics(blood_normalized)
            
            # Save processed data
            self.save_processed_data(liver_normalized, 'liver_enzymes_processed.csv')
            self.save_processed_data(blood_normalized, 'blood_parameters_processed.csv')
            
            logger.info("Data processing pipeline completed successfully")
            
            return {
                'liver_data': liver_normalized,
                'blood_data': blood_normalized,
                'metadata': metadata,
                'liver_stats': liver_stats,
                'blood_stats': blood_stats
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_data()
    print("\n✓ Data preprocessing complete!")
